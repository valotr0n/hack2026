import asyncio
import logging
import os
import uuid
from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pydub import AudioSegment
from ..llm import chat
from ..config import settings
from ..json_utils import candidate_sentences, parse_json_payload, safe_sample
from ..tts import SPEAKER_ALEX, SPEAKER_MARIA, synthesize

router = APIRouter()
logger = logging.getLogger(__name__)


class PodcastRequest(BaseModel):
    text: str
    tone: str = "popular"  # scientific | popular


class PodcastResponse(BaseModel):
    audio_url: str
    script: list[dict]


async def _tts(text: str, voice: str, output_path: str) -> None:
    await synthesize(text, voice, output_path)


def _fallback_script(text: str, tone: str) -> list[dict]:
    intro = (
        "Разберём материал последовательно и без лишних допущений."
        if tone == "scientific"
        else "Давай быстро разберём, о чём этот материал и что в нём главное."
    )
    sentences = candidate_sentences(text, min_length=30, max_items=8)
    if not sentences:
        sentences = ["В документе недостаточно связного текста для полноценного подкаста."]

    script: list[dict] = [
        {"speaker": "Алекс", "text": intro},
    ]
    speakers = ("Мария", "Алекс")
    for index, sentence in enumerate(sentences[:6], start=1):
        speaker = speakers[index % 2]
        script.append({"speaker": speaker, "text": sentence[:360]})

    while len(script) < 6:
        speaker = "Мария" if len(script) % 2 else "Алекс"
        script.append(
            {
                "speaker": speaker,
                "text": "Это ключевой фрагмент документа, который стоит дополнительно проверить в исходном тексте.",
            }
        )

    return script[:8]


def _normalize_script(data: object, fallback_text: str, tone: str) -> list[dict]:
    if not isinstance(data, dict):
        return _fallback_script(fallback_text, tone)

    raw_script = data.get("script")
    if not isinstance(raw_script, list):
        return _fallback_script(fallback_text, tone)

    script: list[dict] = []
    for index, item in enumerate(raw_script):
        if isinstance(item, dict):
            speaker = str(item.get("speaker") or item.get("role") or "").strip()
            text = str(item.get("text") or item.get("content") or item.get("line") or "").strip()
        else:
            speaker = ""
            text = str(item).strip()

        if not text:
            continue

        if speaker not in {"Алекс", "Мария"}:
            speaker = "Алекс" if index % 2 == 0 else "Мария"

        script.append({"speaker": speaker, "text": text[:360]})

    return script if len(script) >= 2 else _fallback_script(fallback_text, tone)


@router.post("/podcast", response_model=PodcastResponse)
async def generate_podcast(req: PodcastRequest) -> PodcastResponse:
    # Шаг 1 — генерируем сценарий диалога
    tone_prompt = (
        "в научном стиле с использованием терминологии"
        if req.tone == "scientific"
        else "в доступном популярном стиле, просто и интересно"
    )

    system = (
        "Ты — сценарист подкаста. "
        "Используй ТОЛЬКО информацию из предоставленного текста. "
        "Отвечай строго в формате JSON, без лишнего текста."
    )

    user = (
        f"Создай сценарий подкаста {tone_prompt} на основе текста ниже.\n"
        "Два ведущих: Алекс и Мария обсуждают материал, задают друг другу вопросы.\n"
        "Минимум 6 реплик. Верни JSON:\n"
        '{"script": [{"speaker": "Алекс", "text": "..."}, {"speaker": "Мария", "text": "..."}]}\n\n'
        f"Текст:\n{req.text}"
    )

    try:
        raw = await chat(system=system, user=user)
    except Exception as exc:
        logger.warning("Podcast generation failed before parsing: %s", exc)
        script = _fallback_script(req.text, req.tone)
    else:
        data = parse_json_payload(raw)
        if data is None:
            logger.warning("Podcast parse failed, using fallback. raw_sample=%r", safe_sample(raw))
            script = _fallback_script(req.text, req.tone)
        else:
            script = _normalize_script(data, req.text, req.tone)

    # Шаг 2 — озвучиваем каждую реплику через edge-tts
    os.makedirs(settings.audio_dir, exist_ok=True)
    session_id = str(uuid.uuid4())
    temp_files = []

    tasks = []
    for i, line in enumerate(script):
        voice = SPEAKER_ALEX if line["speaker"] == "Алекс" else SPEAKER_MARIA
        path = os.path.join(settings.audio_dir, f"{session_id}_{i}.mp3")
        temp_files.append(path)
        tasks.append(_tts(line["text"], voice, path))

    try:
        await asyncio.gather(*tasks)

        # Шаг 3 — склеиваем аудио в один файл
        combined = AudioSegment.empty()
        pause = AudioSegment.silent(duration=500)

        for path in temp_files:
            segment = AudioSegment.from_mp3(path)
            combined += segment + pause

        output_filename = f"{session_id}.mp3"
        output_path = os.path.join(settings.audio_dir, output_filename)
        combined.export(output_path, format="mp3")
    finally:
        for path in temp_files:
            try:
                os.remove(path)
            except FileNotFoundError:
                pass

    return PodcastResponse(
        audio_url=f"/audio/{output_filename}",
        script=script,
    )


@router.get("/audio/{filename}")
async def get_audio(filename: str):
    audio_dir = Path(settings.audio_dir).resolve()
    file_path = (audio_dir / filename).resolve()
    if not str(file_path).startswith(str(audio_dir) + os.sep) and file_path != audio_dir:
        raise HTTPException(status_code=400, detail="Недопустимое имя файла")
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Файл не найден")
    return FileResponse(str(file_path), media_type="audio/mpeg")
