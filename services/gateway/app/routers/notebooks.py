from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import httpx
from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ..auth import require_auth
from ..config import settings
from ..database import (
    create_notebook,
    create_source,
    delete_notebook,
    delete_source,
    get_notebook,
    get_source,
    list_notebooks,
    list_sources,
    save_notebook_content,
    update_notebook_title,
)

router = APIRouter(prefix="/notebooks", tags=["notebooks"])

# Максимальный объём текста для передачи в LLM (~20k символов ≈ 5k токенов)
_MAX_TEXT_LENGTH = 20_000

import json as _json

def _parse_notebook(nb: dict) -> dict:
    """Десериализует JSON-колонки из SQLite в Python-объекты."""
    for field in ("mindmap", "flashcards", "podcast_script"):
        raw = nb.get(field)
        if isinstance(raw, str):
            try:
                nb[field] = _json.loads(raw)
            except Exception:
                nb[field] = None
    return nb


# ── helpers ───────────────────────────────────────────────────────────────────

def _rag(path: str) -> str:
    return f"{settings.rag_service_url.rstrip('/')}/{path.lstrip('/')}"


def _content(path: str) -> str:
    return f"{settings.content_service_url.rstrip('/')}/{path.lstrip('/')}"


async def _owned_notebook(notebook_id: str, user_id: str) -> dict[str, Any]:
    notebook = await get_notebook(settings.db_path, notebook_id)
    if not notebook:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Блокнот не найден")
    if notebook["user_id"] != user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Доступ запрещён")
    return notebook


async def _notebook_text(client: httpx.AsyncClient, notebook_id: str) -> str:
    try:
        resp = await client.get(_rag(f"/notebook/{notebook_id}/content"))
        resp.raise_for_status()
    except httpx.RequestError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc))
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=exc.response.text)

    text: str = resp.json().get("text", "")
    if not text.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Блокнот пуст — загрузите хотя бы один источник",
        )
    # Мягкое обрезание: не ломаем слово посередине
    if len(text) > _MAX_TEXT_LENGTH:
        text = text[:_MAX_TEXT_LENGTH].rsplit(" ", 1)[0]
    return text


# ── schemas ───────────────────────────────────────────────────────────────────

class CreateNotebookRequest(BaseModel):
    title: str = Field(min_length=1, max_length=200)


class UpdateNotebookRequest(BaseModel):
    title: str = Field(min_length=1, max_length=200)


class SourceResponse(BaseModel):
    id: str
    filename: str
    chunks_count: int
    created_at: str


class NotebookResponse(BaseModel):
    id: str
    title: str
    created_at: str
    sources: list[SourceResponse] = []
    summary: str | None = None
    mindmap: dict | None = None
    flashcards: list | None = None
    podcast_url: str | None = None
    podcast_script: list | None = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    query: str
    history: list[ChatMessage] = []


class SummaryRequest(BaseModel):
    style: str = "official"  # official | popular


class FlashcardsRequest(BaseModel):
    count: int = Field(default=10, ge=1, le=50)


class PodcastRequest(BaseModel):
    tone: str = "popular"  # scientific | popular


# ── Notebook CRUD ─────────────────────────────────────────────────────────────

@router.post(
    "",
    response_model=NotebookResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Создать блокнот",
    description="Создаёт новый пустой блокнот. После создания загружайте в него документы через `/notebooks/{id}/sources`.",
)
async def create(
    req: CreateNotebookRequest,
    user_id: str = Depends(require_auth),
) -> NotebookResponse:
    notebook = await create_notebook(settings.db_path, user_id, req.title)
    return NotebookResponse(**notebook)


@router.get(
    "",
    response_model=list[NotebookResponse],
    summary="Список блокнотов",
    description="Возвращает все блокноты текущего пользователя вместе со списком источников в каждом.",
)
async def list_all(user_id: str = Depends(require_auth)) -> list[NotebookResponse]:
    notebooks = await list_notebooks(settings.db_path, user_id)
    result = []
    for nb in notebooks:
        sources = await list_sources(settings.db_path, nb["id"])
        result.append(NotebookResponse(**nb, sources=sources))
    return result


@router.get(
    "/{notebook_id}",
    response_model=NotebookResponse,
    summary="Получить блокнот",
    description="Возвращает блокнот по ID со списком загруженных источников.",
)
async def get_one(
    notebook_id: str,
    user_id: str = Depends(require_auth),
) -> NotebookResponse:
    notebook = _parse_notebook(await _owned_notebook(notebook_id, user_id))
    sources = await list_sources(settings.db_path, notebook_id)
    return NotebookResponse(**notebook, sources=sources)


@router.patch(
    "/{notebook_id}",
    response_model=NotebookResponse,
    summary="Переименовать блокнот",
    description="Обновляет название блокнота.",
)
async def update(
    notebook_id: str,
    req: UpdateNotebookRequest,
    user_id: str = Depends(require_auth),
) -> NotebookResponse:
    notebook = await _owned_notebook(notebook_id, user_id)
    await update_notebook_title(settings.db_path, notebook_id, req.title)
    notebook["title"] = req.title
    sources = await list_sources(settings.db_path, notebook_id)
    return NotebookResponse(**notebook, sources=sources)


@router.delete(
    "/{notebook_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Удалить блокнот",
    description="Удаляет блокнот, все его источники и векторные данные из базы.",
)
async def delete(
    notebook_id: str,
    request: Request,
    user_id: str = Depends(require_auth),
) -> None:
    await _owned_notebook(notebook_id, user_id)
    client: httpx.AsyncClient = request.app.state.http_client
    # Удаляем коллекцию Qdrant (best-effort)
    try:
        await client.delete(_rag(f"/collection/{notebook_id}"))
    except Exception:
        pass
    await delete_notebook(settings.db_path, notebook_id)


# ── Sources ───────────────────────────────────────────────────────────────────

@router.post(
    "/{notebook_id}/sources",
    status_code=status.HTTP_201_CREATED,
    summary="Загрузить документ",
    description="Загружает файл (PDF, DOCX, TXT) в блокнот. Документ автоматически разбивается на чанки и индексируется для поиска. Возвращает `id` источника.",
)
async def upload_source(
    notebook_id: str,
    request: Request,
    file: UploadFile = File(...),
    user_id: str = Depends(require_auth),
) -> dict[str, Any]:
    await _owned_notebook(notebook_id, user_id)
    client: httpx.AsyncClient = request.app.state.http_client

    file_content = await file.read()
    try:
        rag_resp = await client.post(
            _rag("/upload"),
            files={"file": (file.filename, io.BytesIO(file_content), file.content_type or "application/octet-stream")},
            data={"notebook_id": notebook_id},
        )
        rag_resp.raise_for_status()
    except httpx.RequestError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc))
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=exc.response.text)

    rag_data = rag_resp.json()
    return await create_source(
        settings.db_path,
        notebook_id,
        file.filename or "document",
        rag_data.get("chunks", 0),
    )


@router.post(
    "/{notebook_id}/sources/transcribe",
    status_code=status.HTTP_201_CREATED,
    summary="Загрузить аудио/видео и транскрибировать",
    description="""
Принимает аудио или видеофайл, транскрибирует его через Whisper STT и сохраняет транскрипцию как источник в блокноте.

**Поддерживаемые форматы:**
- Аудио: mp3, wav, ogg, m4a, flac, aac, opus
- Видео: mp4, avi, mov, mkv, webm, flv

Возвращает созданный источник с `id`, `filename` (имя_файла_transcription.txt) и `chunks_count`.
    """,
)
async def transcribe_source(
    notebook_id: str,
    request: Request,
    file: UploadFile = File(...),
    user_id: str = Depends(require_auth),
) -> dict[str, Any]:
    await _owned_notebook(notebook_id, user_id)
    client: httpx.AsyncClient = request.app.state.http_client

    file_content = await file.read()
    try:
        transcribe_resp = await client.post(
            _content("/transcribe"),
            files={"file": (file.filename, io.BytesIO(file_content), file.content_type or "application/octet-stream")},
            timeout=300.0,
        )
        transcribe_resp.raise_for_status()
    except httpx.RequestError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc))
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=exc.response.text)

    text = transcribe_resp.json().get("text", "")
    stem = Path(file.filename or "transcription").stem
    txt_filename = f"{stem}_transcription.txt"

    try:
        rag_resp = await client.post(
            _rag("/upload"),
            files={"file": (txt_filename, io.BytesIO(text.encode("utf-8")), "text/plain")},
            data={"notebook_id": notebook_id},
            timeout=120.0,
        )
        rag_resp.raise_for_status()
    except httpx.RequestError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc))
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=exc.response.text)

    rag_data = rag_resp.json()
    return await create_source(
        settings.db_path,
        notebook_id,
        txt_filename,
        rag_data.get("chunks", 0),
    )


@router.delete(
    "/{notebook_id}/sources/{source_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Удалить источник",
    description="Удаляет конкретный документ из блокнота вместе с его векторными данными.",
)
async def remove_source(
    notebook_id: str,
    source_id: str,
    request: Request,
    user_id: str = Depends(require_auth),
) -> None:
    await _owned_notebook(notebook_id, user_id)
    source = await get_source(settings.db_path, source_id)
    if not source or source["notebook_id"] != notebook_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Источник не найден")

    client: httpx.AsyncClient = request.app.state.http_client
    try:
        await client.delete(_rag(f"/collection/{notebook_id}/sources/{source_id}"))
    except Exception:
        pass
    await delete_source(settings.db_path, source_id)


# ── Chat (SSE stream) ─────────────────────────────────────────────────────────

@router.post(
    "/{notebook_id}/chat",
    summary="Чат с документами",
    description="""
Задаёт вопрос по содержимому блокнота. Возвращает **SSE-стрим** (Server-Sent Events).

**Формат каждого события:**
```
data: {"delta": "фрагмент текста", "sources": ["текст чанка 1", ...]}
```

**Завершение стрима:**
```
data: [DONE]
```

**Пример на JS:**
```js
const es = new EventSource('/notebooks/{id}/chat'); // или через fetch с ReadableStream
```

Поле `history` — массив предыдущих сообщений для контекста диалога. При первом вопросе передавайте пустой массив.
    """,
)
async def chat(
    notebook_id: str,
    req: ChatRequest,
    request: Request,
    user_id: str = Depends(require_auth),
) -> StreamingResponse:
    await _owned_notebook(notebook_id, user_id)
    client: httpx.AsyncClient = request.app.state.http_client

    try:
        rag_request = client.build_request(
            "POST",
            _rag("/chat"),
            json={
                "doc_id": notebook_id,
                "query": req.query,
                "history": [m.model_dump() for m in req.history],
            },
        )
        rag_resp = await client.send(rag_request, stream=True)
    except httpx.RequestError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc))

    async def _stream():
        try:
            async for chunk in rag_resp.aiter_bytes():
                yield chunk
        finally:
            await rag_resp.aclose()

    return StreamingResponse(
        _stream(),
        status_code=rag_resp.status_code,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Content generation ────────────────────────────────────────────────────────

@router.post(
    "/{notebook_id}/summary",
    summary="Создать саммари",
    description="""
Генерирует краткое изложение всех документов в блокноте.

**Параметр `style`:**
- `official` — деловой стиль (по умолчанию)
- `popular` — простым языком

**Ответ:**
```json
{"summary": "Текст краткого изложения..."}
```
    """,
)
async def summary(
    notebook_id: str,
    req: SummaryRequest,
    request: Request,
    user_id: str = Depends(require_auth),
) -> dict[str, Any]:
    await _owned_notebook(notebook_id, user_id)
    client: httpx.AsyncClient = request.app.state.http_client
    text = await _notebook_text(client, notebook_id)

    try:
        resp = await client.post(_content("/summary"), json={"text": text, "style": req.style})
        resp.raise_for_status()
        data = resp.json()
        await save_notebook_content(settings.db_path, notebook_id, "summary", data.get("summary", ""))
        return data
    except httpx.RequestError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc))


@router.post(
    "/{notebook_id}/mindmap",
    summary="Создать mindmap",
    description="""
Генерирует структуру mindmap по содержимому блокнота.

**Ответ** — дерево в JSON:
```json
{
  "title": "Главная тема",
  "children": [
    {
      "title": "Подтема 1",
      "children": [
        {"title": "Пункт 1.1", "children": []},
        {"title": "Пункт 1.2", "children": []}
      ]
    }
  ]
}
```
    """,
)
async def mindmap(
    notebook_id: str,
    request: Request,
    user_id: str = Depends(require_auth),
) -> dict[str, Any]:
    await _owned_notebook(notebook_id, user_id)
    client: httpx.AsyncClient = request.app.state.http_client
    text = await _notebook_text(client, notebook_id)

    try:
        resp = await client.post(_content("/mindmap"), json={"text": text})
        resp.raise_for_status()
        data = resp.json()
        await save_notebook_content(settings.db_path, notebook_id, "mindmap", _json.dumps(data, ensure_ascii=False))
        return data
    except httpx.RequestError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc))


@router.post(
    "/{notebook_id}/flashcards",
    summary="Создать карточки",
    description="""
Генерирует флэш-карточки для запоминания материала из блокнота.

**Параметр `count`** — количество карточек (1–50, по умолчанию 10).

**Ответ:**
```json
{
  "flashcards": [
    {"question": "Что такое RAG?", "answer": "Retrieval-Augmented Generation — подход..."},
    ...
  ]
}
```
    """,
)
async def flashcards(
    notebook_id: str,
    req: FlashcardsRequest,
    request: Request,
    user_id: str = Depends(require_auth),
) -> dict[str, Any]:
    await _owned_notebook(notebook_id, user_id)
    client: httpx.AsyncClient = request.app.state.http_client
    text = await _notebook_text(client, notebook_id)

    try:
        resp = await client.post(_content("/flashcards"), json={"text": text, "count": req.count})
        resp.raise_for_status()
        data = resp.json()
        await save_notebook_content(settings.db_path, notebook_id, "flashcards", _json.dumps(data.get("flashcards", []), ensure_ascii=False))
        return data
    except httpx.RequestError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc))


@router.post(
    "/{notebook_id}/podcast",
    summary="Создать подкаст",
    description="""
Генерирует аудиоподкаст — диалог двух ведущих (Алекс и Мария) по материалам блокнота.

**Параметр `tone`:**
- `popular` — живой разговорный стиль (по умолчанию)
- `scientific` — академический стиль

**Ответ:**
```json
{"audio_url": "/audio/abc123.mp3"}
```

Для воспроизведения запросите файл: `GET /audio/{filename}`.
Генерация занимает **30–120 секунд** в зависимости от объёма текста.
    """,
)
async def podcast(
    notebook_id: str,
    req: PodcastRequest,
    request: Request,
    user_id: str = Depends(require_auth),
) -> dict[str, Any]:
    await _owned_notebook(notebook_id, user_id)
    client: httpx.AsyncClient = request.app.state.http_client
    text = await _notebook_text(client, notebook_id)

    try:
        resp = await client.post(_content("/podcast"), json={"text": text, "tone": req.tone})
        resp.raise_for_status()
        data = resp.json()
        await save_notebook_content(settings.db_path, notebook_id, "podcast_url", data.get("audio_url", ""))
        await save_notebook_content(settings.db_path, notebook_id, "podcast_script", _json.dumps(data.get("script", []), ensure_ascii=False))
        return data
    except httpx.RequestError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc))
