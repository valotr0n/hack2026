from __future__ import annotations

import asyncio
import os

import torch

# Голоса silero для русского v3_1_ru
SPEAKER_ALEX = "aidar"   # мужской
SPEAKER_MARIA = "xenia"  # женский
SAMPLE_RATE = 24000

_model = None
_lock = asyncio.Lock()


def _load_model() -> None:
    global _model
    model, _ = torch.hub.load(
        repo_or_dir="snakers4/silero-models",
        model="silero_tts",
        language="ru",
        speaker="v3_1_ru",
        trust_repo=True,
    )
    _model = model


async def _get_model():
    global _model
    if _model is None:
        async with _lock:
            if _model is None:
                await asyncio.to_thread(_load_model)
    return _model


def _synthesize_sync(text: str, speaker: str, output_path: str) -> None:
    audio = _model.apply_tts(text=text, speaker=speaker, sample_rate=SAMPLE_RATE)
    import torchaudio
    from pydub import AudioSegment

    wav_path = output_path.replace(".mp3", ".wav")
    torchaudio.save(wav_path, audio.unsqueeze(0), SAMPLE_RATE)
    AudioSegment.from_wav(wav_path).export(output_path, format="mp3")
    try:
        os.remove(wav_path)
    except FileNotFoundError:
        pass


async def synthesize(text: str, speaker: str, output_path: str) -> None:
    await _get_model()
    await asyncio.to_thread(_synthesize_sync, text, speaker, output_path)
