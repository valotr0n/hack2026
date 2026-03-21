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
    import io
    import numpy as np
    import scipy.io.wavfile
    from pydub import AudioSegment

    audio = _model.apply_tts(text=text, speaker=speaker, sample_rate=SAMPLE_RATE)
    audio_np = (audio.numpy() * 32767).astype(np.int16)

    buf = io.BytesIO()
    scipy.io.wavfile.write(buf, SAMPLE_RATE, audio_np)
    buf.seek(0)
    AudioSegment.from_wav(buf).export(output_path, format="mp3")


async def synthesize(text: str, speaker: str, output_path: str) -> None:
    await _get_model()
    await asyncio.to_thread(_synthesize_sync, text, speaker, output_path)
