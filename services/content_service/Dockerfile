FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

# Torch CPU — ставим отдельно до остальных зависимостей (кэш слоя)
RUN pip install --no-cache-dir \
    torch==2.3.1 \
    torchaudio==2.3.1 \
    --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Предзагружаем silero модель в образ (офлайн работа при деплое)
RUN python -c "\
import torch; \
torch.hub.load('snakers4/silero-models', 'silero_tts', language='ru', speaker='v3_1_ru', trust_repo=True); \
print('Silero TTS model downloaded OK')"

COPY app /app/app

EXPOSE 8002

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8002"]
