import json
import os
import uuid
import asyncio
from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from pydub import AudioSegment
from ..llm import chat
from ..config import settings
from ..tts import AVAILABLE_VOICES, DEFAULT_SPEAKERS, synthesize

router = APIRouter()


class SpeakerConfig(BaseModel):
    name: str
    voice: str  # один из AVAILABLE_VOICES id


class PodcastRequest(BaseModel):
    text: str
    tone: str = "popular"  # scientific | popular
    speakers: list[SpeakerConfig] = Field(default_factory=list)

    def resolved_speakers(self) -> list[SpeakerConfig]:
        if self.speakers:
            return self.speakers[:2]  # максимум 2 ведущих
        return [SpeakerConfig(**s) for s in DEFAULT_SPEAKERS]


class PodcastResponse(BaseModel):
    audio_url: str
    script: list[dict]


@router.get("/voices", summary="Список доступных голосов TTS")
async def list_voices() -> list[dict]:
    return AVAILABLE_VOICES


async def _tts(text: str, voice: str, output_path: str) -> None:
    await synthesize(text, voice, output_path)


@router.post("/podcast", response_model=PodcastResponse)
async def generate_podcast(req: PodcastRequest) -> PodcastResponse:
    speakers = req.resolved_speakers()

    # Валидация голосов
    valid_ids = {v["id"] for v in AVAILABLE_VOICES}
    for sp in speakers:
        if sp.voice not in valid_ids:
            raise HTTPException(status_code=400, detail=f"Неизвестный голос: {sp.voice}")

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

    if len(speakers) == 1:
        sp = speakers[0]
        user = (
            f"Создай сценарий подкаста {tone_prompt} на основе текста ниже.\n"
            f"Ведущий один: {sp.name} рассказывает материал как монолог.\n"
            "Минимум 6 реплик (абзацев). Верни JSON:\n"
            f'{{"script": [{{"speaker": "{sp.name}", "text": "..."}}]}}\n\n'
            f"Текст:\n{req.text}"
        )
    else:
        a, b = speakers[0], speakers[1]
        user = (
            f"Создай сценарий подкаста {tone_prompt} на основе текста ниже.\n"
            f"Два ведущих: {a.name} и {b.name} обсуждают материал, задают друг другу вопросы.\n"
            "Минимум 6 реплик. Верни JSON:\n"
            f'{{"script": [{{"speaker": "{a.name}", "text": "..."}}, {{"speaker": "{b.name}", "text": "..."}}]}}\n\n'
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

    # Карта имя → голос
    voice_map = {sp.name: sp.voice for sp in speakers}
    # Если модель написала что-то не то — fallback на первого спикера
    fallback_voice = speakers[0].voice

    os.makedirs(settings.audio_dir, exist_ok=True)
    session_id = str(uuid.uuid4())
    temp_files = []

    tasks = []
    for i, line in enumerate(script):
        voice = voice_map.get(line["speaker"], fallback_voice)
        path = os.path.join(settings.audio_dir, f"{session_id}_{i}.mp3")
        temp_files.append(path)
        tasks.append(_tts(line["text"], voice, path))

    try:
        await asyncio.gather(*tasks)

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
