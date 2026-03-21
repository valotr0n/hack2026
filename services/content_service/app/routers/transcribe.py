from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from pathlib import Path

import httpx
from fastapi import APIRouter, File, HTTPException, UploadFile, status

from ..config import settings

logger = logging.getLogger(__name__)
router = APIRouter()

AUDIO_EXTENSIONS = {".mp3", ".wav", ".ogg", ".m4a", ".flac", ".aac", ".opus"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}
ALL_EXTENSIONS = AUDIO_EXTENSIONS | VIDEO_EXTENSIONS

# ── Локальный Whisper (faster-whisper) ────────────────────────────────────────

_whisper_model = None
_whisper_lock = asyncio.Lock()


def _load_whisper() -> None:
    global _whisper_model
    from faster_whisper import WhisperModel
    _whisper_model = WhisperModel(
        settings.whisper_model_size,
        device="cpu",
        compute_type="int8",
    )
    logger.info("Whisper model '%s' loaded", settings.whisper_model_size)


async def _get_whisper():
    global _whisper_model
    if _whisper_model is None:
        async with _whisper_lock:
            if _whisper_model is None:
                await asyncio.to_thread(_load_whisper)
    return _whisper_model


def _transcribe_local_sync(audio_path: str) -> str:
    segments, _ = _whisper_model.transcribe(audio_path, language="ru", beam_size=5)
    return " ".join(seg.text.strip() for seg in segments).strip()


async def _transcribe_local(audio_path: str) -> str:
    await _get_whisper()
    text = await asyncio.to_thread(_transcribe_local_sync, audio_path)
    if not text:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Локальный Whisper вернул пустую транскрипцию.",
        )
    return text


# ── Хакатонный STT API ────────────────────────────────────────────────────────

async def _transcribe_api(audio_path: str, audio_filename: str) -> str:
    async with httpx.AsyncClient(verify=False, timeout=300.0) as client:
        with open(audio_path, "rb") as f:
            resp = await client.post(
                f"{settings.stt_base_url.rstrip('/')}/audio/transcriptions",
                headers={"Authorization": f"Bearer {settings.stt_api_key}"},
                files={"file": (audio_filename, f, "audio/mpeg")},
                data={"model": "whisper", "language": "ru"},
            )
    if resp.status_code != 200:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"STT API вернул {resp.status_code}: {resp.text}",
        )
    text = resp.json().get("text", "").strip()
    if not text:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="STT API вернул пустую транскрипцию.",
        )
    return text


# ── Диспетчер по режиму ───────────────────────────────────────────────────────

async def _transcribe(audio_path: str, audio_filename: str) -> str:
    mode = settings.stt_mode

    if mode == "local":
        return await _transcribe_local(audio_path)

    if mode == "api":
        return await _transcribe_api(audio_path, audio_filename)

    # auto: пробуем API, при ошибке фолбэк на локальный
    try:
        return await _transcribe_api(audio_path, audio_filename)
    except HTTPException as exc:
        logger.warning("STT API недоступен (%s), переключаемся на локальный Whisper", exc.detail)
        return await _transcribe_local(audio_path)


# ── ffmpeg: извлечение аудио из видео ────────────────────────────────────────

async def _extract_audio(video_path: str, audio_path: str) -> None:
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-acodec", "libmp3lame", "-q:a", "4",
        audio_path,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    await proc.wait()
    if proc.returncode != 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Не удалось извлечь аудио из видео. Проверьте формат файла.",
        )


# ── Эндпоинт ─────────────────────────────────────────────────────────────────

@router.post(
    "/transcribe",
    summary="Транскрибировать аудио/видео",
    description=f"""
Принимает аудио или видеофайл, извлекает речь и возвращает текст транскрипции.

**Поддерживаемые форматы:**
- Аудио: mp3, wav, ogg, m4a, flac, aac, opus
- Видео: mp4, avi, mov, mkv, webm, flv

**Режимы STT** (задаются через `STT_MODE` в окружении):
- `auto` — сначала хакатонный API, при недоступности — локальный Whisper
- `api` — только хакатонный API (открытый контур)
- `local` — только локальный Whisper (закрытый контур)

**Ответ:**
```json
{{"text": "Транскрибированный текст..."}}
```
    """,
)
async def transcribe(file: UploadFile = File(...)) -> dict[str, str]:
    filename = file.filename or "upload"
    suffix = Path(filename).suffix.lower()

    if suffix not in ALL_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Неподдерживаемый формат '{suffix}'. Поддерживаются: {', '.join(sorted(ALL_EXTENSIONS))}",
        )

    content = await file.read()
    if not content:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Файл пустой.")

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, f"input{suffix}")
        with open(input_path, "wb") as f:
            f.write(content)

        if suffix in VIDEO_EXTENSIONS:
            audio_path = os.path.join(tmpdir, "audio.mp3")
            await _extract_audio(input_path, audio_path)
        else:
            audio_path = input_path

        text = await _transcribe(audio_path, Path(audio_path).name)

    return {"text": text}
