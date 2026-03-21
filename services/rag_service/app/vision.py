from __future__ import annotations

import asyncio
import logging
from time import perf_counter
from io import BytesIO

logger = logging.getLogger(__name__)

_processor = None
_model = None
_load_error: str | None = None
_lock = asyncio.Lock()

_DESCRIBE_PROMPT = (
    "Кратко и фактически опиши изображение или схему для RAG-индексации. "
    "Перечисли только ключевые объекты, подписи и видимый текст. "
    "Не додумывай и уложись в 3-5 коротких предложений. "
    "Отвечай на русском языке."
)


def _load_vision_model(model_id: str) -> None:
    global _processor, _model
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    started_at = perf_counter()
    _processor = AutoProcessor.from_pretrained(model_id)
    _model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    _model.eval()
    logger.info("A-Vision model loaded: %s in %.2fs", model_id, perf_counter() - started_at)


async def _get_vision_model(model_id: str):
    global _model, _processor, _load_error
    if _load_error is not None:
        raise RuntimeError(_load_error)

    if _model is None:
        async with _lock:
            if _model is None:
                try:
                    await asyncio.to_thread(_load_vision_model, model_id)
                except Exception as exc:
                    _load_error = f"Unable to load vision model {model_id}: {exc}"
                    logger.warning(_load_error)
                    raise RuntimeError(_load_error) from exc
    return _processor, _model


def _resize_image(image, max_image_side: int):
    from PIL import Image

    if max_image_side <= 0 or max(image.size) <= max_image_side:
        return image

    resized = image.copy()
    resized.thumbnail((max_image_side, max_image_side), Image.Resampling.LANCZOS)
    return resized


def _describe_sync(image_bytes: bytes, max_new_tokens: int, max_image_side: int) -> str:
    from PIL import Image
    from qwen_vl_utils import process_vision_info
    import torch

    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image = _resize_image(image, max_image_side)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": _DESCRIBE_PROMPT},
            ],
        }
    ]

    text = _processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = _processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(_model.device)

    with torch.no_grad():
        generated_ids = _model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    trimmed = [
        out[len(inp):]
        for inp, out in zip(inputs.input_ids, generated_ids)
    ]
    result = _processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return result[0].strip() if result else ""


async def _describe_image(
    image_bytes: bytes,
    model_id: str,
    max_new_tokens: int,
    max_image_side: int,
) -> tuple[str, float]:
    started_at = perf_counter()
    try:
        await _get_vision_model(model_id)
        description = await asyncio.to_thread(
            _describe_sync,
            image_bytes,
            max_new_tokens,
            max_image_side,
        )
        return description, perf_counter() - started_at
    except Exception as exc:
        logger.warning("Vision model failed while describing image: %s", exc)
        return "", perf_counter() - started_at


def _extract_images_from_pdf(payload: bytes) -> list[bytes]:
    from pypdf import PdfReader

    reader = PdfReader(BytesIO(payload))
    images: list[bytes] = []
    for page in reader.pages:
        for img in page.images:
            images.append(img.data)
    return images


def _extract_images_from_docx(payload: bytes) -> list[bytes]:
    from docx import Document

    doc = Document(BytesIO(payload))
    images: list[bytes] = []
    for rel in doc.part.rels.values():
        if "image" in rel.reltype:
            try:
                images.append(rel.target_part.blob)
            except Exception:
                pass
    return images


def _build_warmup_image_bytes() -> bytes:
    from PIL import Image

    buffer = BytesIO()
    Image.new("RGB", (64, 64), color="white").save(buffer, format="PNG")
    return buffer.getvalue()


async def preload_vision_model(model_id: str, max_new_tokens: int, max_image_side: int) -> None:
    started_at = perf_counter()
    logger.info("Starting background preload of vision model: %s", model_id)
    try:
        await _get_vision_model(model_id)
        await asyncio.to_thread(
            _describe_sync,
            _build_warmup_image_bytes(),
            min(max_new_tokens, 16),
            min(max_image_side, 256),
        )
    except Exception as exc:
        logger.warning("Vision preload failed: %s", exc)
        return

    logger.info("Background vision preload finished in %.2fs", perf_counter() - started_at)


async def extract_image_descriptions(
    payload: bytes,
    suffix: str,
    model_id: str,
    max_new_tokens: int,
    max_image_side: int,
) -> list[str]:
    """Извлекает изображения из PDF/DOCX и возвращает их текстовые описания через A-Vision."""
    if suffix == ".pdf":
        image_bytes_list = _extract_images_from_pdf(payload)
    elif suffix == ".docx":
        image_bytes_list = _extract_images_from_docx(payload)
    else:
        return []

    if not image_bytes_list:
        return []

    started_at = perf_counter()
    logger.info("Found %d image(s) in document, describing via A-Vision...", len(image_bytes_list))

    descriptions: list[str] = []
    for i, img_bytes in enumerate(image_bytes_list, 1):
        desc, elapsed = await _describe_image(
            img_bytes,
            model_id,
            max_new_tokens,
            max_image_side,
        )
        if desc:
            descriptions.append(f"[Изображение {i}]: {desc}")
            logger.info(
                "Vision described image %d/%d in %.2fs.",
                i,
                len(image_bytes_list),
                elapsed,
            )
        else:
            logger.warning(
                "Vision returned an empty description for image %d/%d after %.2fs.",
                i,
                len(image_bytes_list),
                elapsed,
            )

    if not descriptions:
        logger.warning(
            "Found %d image(s) in document, but vision descriptions were not produced.",
            len(image_bytes_list),
        )
    else:
        logger.info(
            "Vision produced %d/%d image description(s) in %.2fs.",
            len(descriptions),
            len(image_bytes_list),
            perf_counter() - started_at,
        )

    return descriptions
