FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TORCH_HOME=/app/models/torch \
    HF_HOME=/app/models/hf

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

# Torch CPU — отдельный слой (тяжёлый, кэшируется отдельно)
RUN pip install --no-cache-dir \
    torch==2.3.1 \
    torchaudio==2.3.1 \
    --index-url https://download.pytorch.org/whl/cpu

# Все остальные зависимости (включая omegaconf, requests, faster-whisper)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Предзагружаем silero TTS (~50MB) — офлайн TTS
RUN python -c "import torch; torch.hub.load('snakers4/silero-models', 'silero_tts', language='ru', speaker='v3_1_ru', trust_repo=True); print('Silero TTS OK')"

# Предзагружаем Whisper small (~244MB) — офлайн STT
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('small', device='cpu', compute_type='int8'); print('Whisper small OK')"

COPY app /app/app

EXPOSE 8002

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8002"]
