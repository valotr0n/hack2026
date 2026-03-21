from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path

import httpx
from fastapi import APIRouter, File, HTTPException, UploadFile, status

from ..config import settings

router = APIRouter()

AUDIO_EXTENSIONS = {".mp3", ".wav", ".ogg", ".m4a", ".flac", ".aac", ".opus"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}
ALL_EXTENSIONS = AUDIO_EXTENSIONS | VIDEO_EXTENSIONS


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


async def _call_stt(audio_path: str, audio_filename: str) -> str:
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
    text = resp.json().get("text", "")
    if not text.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="STT API вернул пустую транскрипцию.",
        )
    return text


@router.post(
    "/transcribe",
    summary="Транскрибировать аудио/видео",
    description="""
Принимает аудио или видеофайл, извлекает речь и возвращает текст транскрипции.

**Поддерживаемые форматы:**
- Аудио: mp3, wav, ogg, m4a, flac, aac, opus
- Видео: mp4, avi, mov, mkv, webm, flv

**Ответ:**
```json
{"text": "Транскрибированный текст..."}
```

Транскрипция выполняется через Whisper STT API. Время обработки зависит от длины файла.
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
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Файл пустой.",
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, f"input{suffix}")
        with open(input_path, "wb") as f:
            f.write(content)

        if suffix in VIDEO_EXTENSIONS:
            audio_path = os.path.join(tmpdir, "audio.mp3")
            await _extract_audio(input_path, audio_path)
        else:
            audio_path = input_path

        text = await _call_stt(audio_path, Path(audio_path).name)

    return {"text": text}
