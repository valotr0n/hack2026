# CentralAI — AI-платформа переработки знаний

> Хакатон Банк Центр-Инвест 2026 · Альтернатива NotebookLM для русского языка

---

## Что это

KnowledgeAI превращает ваши документы в структурированные знания: задавайте вопросы, получайте саммари, mindmap, карточки для изучения и аудиоподкасты — всё это на русском языке, с поддержкой полностью офлайн-режима.

---

## Ключевые возможности

| Фича | Описание |
|---|---|
| **RAG-чат** | Диалог с содержимым блокнота, стриминг токен-по-токену (SSE) |
| **Саммари** | Краткое изложение всех загруженных документов |
| **Mindmap** | Визуальная карта знаний в формате JSON |
| **Flashcards** | Карточки вопрос–ответ для запоминания материала |
| **Подкаст** | Аудио-диалог двух голосов (Дмитрий + Светлана) на базе edge-tts |
| **Векторный поиск** | Qdrant + ru-en-RoSBERTa, чанки по 512 токенов |
| **История чата** | Сохраняется в SQLite, переживает обновление страницы |
| **Кэш генераций** | Саммари/mindmap/карточки не пересчитываются при повторном запросе |

---

## Архитектура

```
                        ┌─────────────┐
                        │   Frontend  │
                        └──────┬──────┘
                               │ HTTP / SSE
                        ┌──────▼──────┐
                        │   Gateway   │  :8000
                        │  JWT · SQLite│
                        └──┬───────┬──┘
                           │       │
              ┌────────────▼─┐  ┌──▼────────────┐
              │  RAG Service │  │Content Service│
              │    :8001     │  │    :8002      │
              │ Qdrant·embeds│  │LLM·TTS·cache  │
              └──────┬───────┘  └───────────────┘
                     │
              ┌──────▼──────┐
              │   Qdrant    │  :6333
              └─────────────┘
```

**4 сервиса в docker-compose:**

- `gateway` — единственная точка входа, JWT-авторизация, SQLite (users / notebooks / sources), оркестрация запросов
- `rag_service` — парсинг документов, локальные эмбеддинги, Qdrant
- `content_service` — генерация контента через LLM, TTS-подкасты
- `qdrant` — векторная БД, одна коллекция на блокнот

---

## Режимы работы

| | Open (облако) | Closed (офлайн) |
|---|---|---|
| **LLM** | Хакатонный API Центр-Инвест | Ollama (локально) |
| **STT** | Хакатонный API | Whisper (локально) |
| **Embeddings** | — | ru-en-RoSBERTa (всегда локально) |
| **Данные** | — | не выходят за периметр |

Эмбеддинги **всегда** считаются локально через `sentence-transformers` — векторы документов не передаются на сторонние серверы.

---

## Быстрый старт

```bash
git clone https://github.com/valotr0n/hack2026
cd hack2026
cp .env.example .env
docker compose up --build
```

После запуска:
- Gateway API: http://localhost:8000
- Qdrant UI: http://localhost:6333/dashboard

---

## API (основные эндпоинты)

```
POST   /auth/register          — регистрация
POST   /auth/login             — получить JWT

POST   /notebooks              — создать блокнот
GET    /notebooks              — список блокнотов
DELETE /notebooks/{id}         — удалить блокнот

POST   /notebooks/{id}/sources — загрузить документ
DELETE /notebooks/{id}/sources/{source_id}

POST   /notebooks/{id}/chat    — чат (SSE стриминг)
GET    /notebooks/{id}/chat/history
DELETE /notebooks/{id}/chat/history

POST   /notebooks/{id}/summary
POST   /notebooks/{id}/mindmap
POST   /notebooks/{id}/flashcards
POST   /notebooks/{id}/podcast
GET    /audio/{filename}
```

> Всего 50+ эндпоинтов · 3 микросервиса · 15 AI-фич

---

## Стек

- **Backend:** Python 3.13, FastAPI, aiosqlite
- **Векторная БД:** Qdrant
- **Эмбеддинги:** `ai-forever/ru-en-RoSBERTa` (sentence-transformers)
- **LLM:** `gpt-oss-20b` / Ollama
- **TTS:** edge-tts (ru-RU-DmitryNeural, ru-RU-SvetlanaNeural), pydub, ffmpeg
- **Инфраструктура:** Docker Compose, JWT HS256

---

## Команда

**input math** · Хакатон Банк Центр-Инвест 2026
