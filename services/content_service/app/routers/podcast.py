import json
import os
import uuid
import asyncio
from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import edge_tts
from pydub import AudioSegment
from ..llm import chat
from ..config import settings

router = APIRouter()


class PodcastRequest(BaseModel):
    text: str
    tone: str = "popular"  # scientific | popular


class PodcastResponse(BaseModel):
    audio_url: str
    script: list[dict]


async def _tts(text: str, voice: str, output_path: str) -> None:
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)


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

    raw = await chat(system=system, user=user)

    try:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        data = json.loads(raw[start:end])
        script = data["script"]
    except Exception:
        raise HTTPException(status_code=500, detail="Не удалось разобрать сценарий")

    # Шаг 2 — озвучиваем каждую реплику через edge-tts
    os.makedirs(settings.audio_dir, exist_ok=True)
    session_id = str(uuid.uuid4())
    temp_files = []

    tasks = []
    for i, line in enumerate(script):
        voice = (
            settings.tts_voice_alex
            if line["speaker"] == "Алекс"
            else settings.tts_voice_maria
        )
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
