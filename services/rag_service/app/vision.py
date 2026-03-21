from __future__ import annotations

import asyncio
import hashlib
import logging
from collections import OrderedDict
from time import perf_counter
from io import BytesIO

logger = logging.getLogger(__name__)

_processor = None
_model = None
_load_error: str | None = None
_lock = asyncio.Lock()
_DESCRIPTION_CACHE: OrderedDict[str, str] = OrderedDict()
_DESCRIPTION_CACHE_LIMIT = 256

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


def _remember_description(image_hash: str, description: str) -> None:
    if not description:
        return

    _DESCRIPTION_CACHE[image_hash] = description
    _DESCRIPTION_CACHE.move_to_end(image_hash)
    while len(_DESCRIPTION_CACHE) > _DESCRIPTION_CACHE_LIMIT:
        _DESCRIPTION_CACHE.popitem(last=False)


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
    image_hash = hashlib.sha1(image_bytes).hexdigest()
    cached_description = _DESCRIPTION_CACHE.get(image_hash)
    if cached_description is not None:
        _DESCRIPTION_CACHE.move_to_end(image_hash)
        return cached_description, perf_counter() - started_at

    try:
        await _get_vision_model(model_id)
        description = await asyncio.to_thread(
            _describe_sync,
            image_bytes,
            max_new_tokens,
            max_image_side,
        )
        _remember_description(image_hash, description)
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


def _select_candidate_images(
    image_bytes_list: list[bytes],
    min_image_side: int,
    max_images_per_document: int,
) -> tuple[list[bytes], dict[str, int]]:
    from PIL import Image

    selected: list[tuple[int, bytes]] = []
    seen_hashes: set[str] = set()
    stats = {
        "total": len(image_bytes_list),
        "duplicates": 0,
        "too_small": 0,
        "invalid": 0,
        "trimmed": 0,
    }

    for image_bytes in image_bytes_list:
        image_hash = hashlib.sha1(image_bytes).hexdigest()
        if image_hash in seen_hashes:
            stats["duplicates"] += 1
            continue
        seen_hashes.add(image_hash)

        try:
            with Image.open(BytesIO(image_bytes)) as image:
                width, height = image.size
        except Exception:
            stats["invalid"] += 1
            continue

        if min(width, height) < min_image_side:
            stats["too_small"] += 1
            continue

        selected.append((width * height, image_bytes))

    selected.sort(key=lambda item: item[0], reverse=True)
    if max_images_per_document > 0 and len(selected) > max_images_per_document:
        stats["trimmed"] = len(selected) - max_images_per_document
        selected = selected[:max_images_per_document]

    return [image_bytes for _, image_bytes in selected], stats


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
    min_image_side: int,
    max_images_per_document: int,
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

    candidate_images, stats = _select_candidate_images(
        image_bytes_list,
        min_image_side=min_image_side,
        max_images_per_document=max_images_per_document,
    )
    if not candidate_images:
        logger.info(
            "Vision skipped all images after filtering: total=%d duplicates=%d too_small=%d invalid=%d",
            stats["total"],
            stats["duplicates"],
            stats["too_small"],
            stats["invalid"],
        )
        return []

    started_at = perf_counter()
    logger.info(
        "Found %d raw image(s), selected %d for vision: duplicates=%d too_small=%d invalid=%d trimmed=%d",
        stats["total"],
        len(candidate_images),
        stats["duplicates"],
        stats["too_small"],
        stats["invalid"],
        stats["trimmed"],
    )

    descriptions: list[str] = []
    for i, img_bytes in enumerate(candidate_images, 1):
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
                len(candidate_images),
                elapsed,
            )
        else:
            logger.warning(
                "Vision returned an empty description for image %d/%d after %.2fs.",
                i,
                len(candidate_images),
                elapsed,
            )

    if not descriptions:
        logger.warning(
            "Found %d selected image(s) in document, but vision descriptions were not produced.",
            len(candidate_images),
        )
    else:
        logger.info(
            "Vision produced %d/%d image description(s) in %.2fs.",
            len(descriptions),
            len(candidate_images),
            perf_counter() - started_at,
        )

    return descriptions
