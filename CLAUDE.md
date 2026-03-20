# Хакатон Банк Центр-Инвест — AI-платформа знаний

## Суть проекта
Альтернатива NotebookLM. AI-платформа глубокой переработки знаний и генерации мультимедийного контента.
Призовой фонд: 100 000 рублей.

Пользователь загружает документы (PDF, DOCX, TXT) → система преобразует их в:
- Чат с источниками (RAG)
- Аудиоподкасты (диалог 2 ведущих)
- Саммари / формализованные отчёты
- Mindmap (интерактивный граф)
- Flashcards (вопрос-ответ)
- Таблицы данных (опционально)

## Репозиторий
github.com/valotr0n/hack2026

## Ветки
- `main` — продакшн
- `dev` — интеграционная ветка (обновлена от main с кодом бэкендера 3)
- `feature/rag-and-chat` — Бэкендер 1
- `feature/content-generation` — Бэкендер 2 (пользователь, работает с Claude) ← ТЕКУЩАЯ
- `feature/api-gateway-orchestration` — Бэкендер 3 (ЗАВЕРШЕНО, смерджено в main)

## Хакатонный ландшафт (API)
- **LLM**: `https://hackai.centrinvest.ru:6630/v1` — OpenAI-совместимый, модель `gpt-oss-20b`
- **Embedder**: `https://hackai.centrinvest.ru:6620/v1` — модель `ru-en-RoSBERTa`
- **STT**: `https://hackai.centrinvest.ru:6640/v1` — Speech-to-Text (аудио → текст)
- **API ключ для всех**: `hackaton2026`
- **Важно**: SSL самоподписанный → использовать `verify=False` в httpx клиенте

## Команда

### Бэкендер 1 — `feature/rag-and-chat`
**Задача:** RAG-система + чат с источниками

- `POST /upload` — парсинг PDF/DOCX/TXT, чанкинг (512 токенов, overlap 50, LlamaIndex)
- `POST /chat` — similarity search top-5, ответ LLM + источники, SSE стриминг
- Embeddings: `ai-forever/ru-en-RoSBERTa` через embedder API (:6620)
- Векторная БД: Qdrant
- Структура: `services/rag_service/`

### Бэкендер 2 (ПОЛЬЗОВАТЕЛЬ) — `feature/content-generation`
**Задача:** Генерация контента — НАПИСАНО, ПРОТЕСТИРОВАНО ✅

**Что сделано:**
- `services/content_service/app/config.py` — настройки из env
- `services/content_service/app/llm.py` — клиент LLM (openai SDK + verify=False)
- `services/content_service/app/routers/summary.py` — `POST /summary` ✅ РАБОТАЕТ
- `services/content_service/app/routers/mindmap.py` — `POST /mindmap`
- `services/content_service/app/routers/flashcards.py` — `POST /flashcards`
- `services/content_service/app/routers/podcast.py` — `POST /podcast` + `GET /audio/{filename}`
- `services/content_service/app/main.py` — FastAPI app, все роутеры подключены

**Зависимости (requirements.txt):**
```
fastapi==0.115.13
uvicorn[standard]==0.34.0
openai==1.57.0
edge-tts==6.1.12
pydub==0.25.1
audioop-lts==0.2.1   ← нужен для Python 3.13
aiofiles==24.1.0
python-multipart==0.0.12
```

**Важные детали:**
- LLM клиент использует `httpx.AsyncClient(verify=False)` — SSL самоподписанный
- Модель: `gpt-oss-20b`
- TTS: edge-tts, голоса `ru-RU-DmitryNeural` (Алекс) и `ru-RU-SvetlanaNeural` (Мария)
- Подкаст требует **ffmpeg** для pydub (склейка аудио)
- Аудио файлы сохраняются в `AUDIO_DIR` (по умолчанию `/app/audio`)
- Сервис слушает на порту **8002**

**Что осталось:**
- Удалить папку `venv` из `services/content_service/` (занята процессом — закрыть терминал)
- Добавить `venv/` в `.gitignore`
- Запушить ветку `feature/content-generation`

### Бэкендер 3 — `feature/api-gateway-orchestration` ✅ ЗАВЕРШЕНО
Смерджено в main. Сделал:
- API Gateway (FastAPI прокси) на порту 8000/443
- docker-compose.yml (все 3 сервиса + сеть ai-platform)
- TLS через openssl скрипты
- `/health` агрегированный
- Makefile, .env.example

**Новые задачи для бэкендера 3:**
1. Добавить Qdrant в docker-compose
2. Поднять REQUEST_TIMEOUT_SECONDS до 120 (для подкастов)
3. Добавить volume для аудиофайлов (`./data/audio:/app/audio`)

## Технический стэк
- Язык: Python 3.13, FastAPI
- LLM: OpenAI-совместимый API (хакатонный ландшафт)
- TTS: edge-tts (онлайн, Microsoft)
- Векторная БД: Qdrant
- Embeddings: ru-en-RoSBERTa через API
- Все сервисы с Dockerfile
- `.gitignore` должен содержать: venv/, .env, data/

## .env.example (актуальный)
```
LLM_BASE_URL=https://hackai.centrinvest.ru:6630/v1
LLM_API_KEY=hackaton2026
LLM_MODEL=gpt-oss-20b
EMBEDDER_BASE_URL=https://hackai.centrinvest.ru:6620/v1
EMBEDDER_API_KEY=hackaton2026
STT_BASE_URL=https://hackai.centrinvest.ru:6640/v1
STT_API_KEY=hackaton2026
TTS_VOICE_ALEX=ru-RU-DmitryNeural
TTS_VOICE_MARIA=ru-RU-SvetlanaNeural
QDRANT_URL=http://qdrant:6333
AUDIO_DIR=/app/audio
CORS_ORIGINS=http://localhost:3000,http://localhost:5173
TLS_CERT_FILE=/app/certs/server.crt
TLS_KEY_FILE=/app/certs/server.key
REQUEST_TIMEOUT_SECONDS=120.0
```

## Критерии оценки (жюри)
1. Качество обработки русского языка
2. Интеграция модулей (текст → карта → тест)
3. UX/UI — удобство для нетехнического пользователя
4. Архитектурная гибкость (работа с открытыми моделями, офлайн)

## Нерешённые вопросы
- **ffmpeg** — нужен для подкастов (pydub склеивает аудио). В Dockerfile добавить: `RUN apt-get install -y ffmpeg`
- **Frontend** — делает отдельный фронтендер, независимо
