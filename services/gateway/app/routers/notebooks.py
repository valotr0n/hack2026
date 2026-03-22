from __future__ import annotations

import io
from pathlib import Path
import re
from typing import Any

import httpx
from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, Query, Request, UploadFile, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, model_validator

from ..auth import require_auth
from ..config import settings
from ..database import (
    clear_chat_history,
    clear_notebook_cache,
    create_notebook,
    create_source,
    delete_notebook,
    delete_source,
    get_chat_history,
    get_notebook,
    get_source,
    list_notebooks,
    list_sources,
    save_chat_message,
    save_notebook_content,
    update_notebook_contour,
    update_notebook_title,
    update_source_autotag,
)

router = APIRouter(prefix="/notebooks", tags=["notebooks"])

# Максимальный объём текста для передачи в LLM (~20k символов ≈ 5k токенов)
_MAX_TEXT_LENGTH = 20_000
# Расширенный лимит для задач извлечения (timeline, contract, knowledge-graph, questions)
# — нужно видеть все источники блокнота (~60k символов ≈ 15k токенов)
_MAX_TEXT_LENGTH_EXTENDED = 60_000

import json as _json

_UUID_RE = re.compile(
    r"[0-9a-fA-F]{8}-"
    r"[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{12}"
)

import logging as _logging

_logger = _logging.getLogger("gateway.notebooks")


async def _run_autotag(client: httpx.AsyncClient, source_id: str, preview: str, contour: str) -> None:
    """Фоновая задача: запускает autotag и сохраняет результат в БД."""
    try:
        tag_resp = await client.post(
            _content("/autotag"),
            json={"text": preview},
            headers={"x-contour": contour},
        )
        tag_resp.raise_for_status()
        tag_data = tag_resp.json()
        await update_source_autotag(
            settings.db_path,
            source_id,
            tag_data.get("doc_type", "прочее"),
            tag_data.get("tags", []),
        )
    except Exception as exc:
        _logger.warning("Background autotag failed for source_id=%s: %s", source_id, exc)


def _parse_notebook(nb: dict) -> dict:
    """Десериализует JSON-колонки из SQLite в Python-объекты."""
    for field in ("mindmap", "flashcards", "podcast_script", "contract", "knowledge_graph", "timeline", "questions", "presentation"):
        raw = nb.get(field)
        if isinstance(raw, str):
            try:
                nb[field] = _json.loads(raw)
            except Exception:
                nb[field] = None
    return nb


def _parse_source(src: dict) -> dict:
    """Десериализует JSON-колонку tags из SQLite."""
    raw = src.get("tags")
    if isinstance(raw, str):
        try:
            src["tags"] = _json.loads(raw)
        except Exception:
            src["tags"] = []
    elif raw is None:
        src["tags"] = []
    return src


def _normalize_pair_payload(
    value: Any,
    list_key: str,
    key_pairs: tuple[tuple[str, str], ...],
) -> Any:
    ids: Any = None

    if isinstance(value, (list, tuple)):
        ids = list(value)
    elif isinstance(value, dict):
        if list_key in value:
            ids = value[list_key]
        elif "ids" in value:
            ids = value["ids"]
        else:
            for left_key, right_key in key_pairs:
                if left_key in value and right_key in value:
                    ids = [value[left_key], value[right_key]]
                    break

    if ids is None:
        return value

    if isinstance(ids, tuple):
        ids = list(ids)

    return {list_key: ids}


def _normalize_notebook_id(value: str) -> str:
    match = _UUID_RE.search(value)
    return match.group(0) if match else value


# ── helpers ───────────────────────────────────────────────────────────────────


def _rag(path: str) -> str:
    return f"{settings.rag_service_url.rstrip('/')}/{path.lstrip('/')}"


def _content(path: str) -> str:
    return f"{settings.content_service_url.rstrip('/')}/{path.lstrip('/')}"


def _contour_headers(notebook: dict) -> dict[str, str]:
    """Передаёт контур блокнота в content_service."""
    return {"x-contour": notebook.get("contour", "open")}


async def _owned_notebook(notebook_id: str, user_id: str) -> dict[str, Any]:
    notebook = await get_notebook(settings.db_path, notebook_id)
    if not notebook:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Блокнот не найден")
    if notebook["user_id"] != user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Доступ запрещён")
    return notebook


async def _notebook_text(
    client: httpx.AsyncClient,
    notebook_id: str,
    max_length: int | None = _MAX_TEXT_LENGTH,
) -> str:
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
    if max_length is not None and len(text) > max_length:
        text = text[:max_length].rsplit(" ", 1)[0]
    return text


async def _notebook_rag_text(
    client: httpx.AsyncClient,
    notebook_id: str,
    query: str,
    top_k: int = 40,
) -> str:
    """Семантически релевантный текст вместо полного — для LLM-фич."""
    try:
        resp = await client.post(
            _rag(f"/notebook/{notebook_id}/search"),
            json={"query": query, "top_k": top_k},
        )
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
    return text


async def _notebook_response(notebook_id: str, user_id: str) -> "NotebookResponse":
    notebook = _parse_notebook(await _owned_notebook(notebook_id, user_id))
    sources = [_parse_source(s) for s in await list_sources(settings.db_path, notebook_id)]
    return NotebookResponse(**notebook, sources=sources)


# ── schemas ───────────────────────────────────────────────────────────────────

class CreateNotebookRequest(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    contour: str = "open"  # open | closed


class ContourRequest(BaseModel):
    contour: str  # open | closed


class UpdateNotebookRequest(BaseModel):
    title: str = Field(min_length=1, max_length=200)


class SourceResponse(BaseModel):
    id: str
    filename: str
    chunks_count: int
    created_at: str
    status: str = "ready"   # processing | ready | error
    error: str | None = None
    doc_type: str | None = None
    tags: list[str] = []


class NotebookListItem(BaseModel):
    id: str
    title: str
    created_at: str


class NotebookResponse(BaseModel):
    id: str
    title: str
    created_at: str
    contour: str = "open"
    sources: list[SourceResponse] = []
    summary: str | None = None
    mindmap: dict | None = None
    flashcards: list | None = None
    podcast_url: str | None = None
    podcast_script: list | None = None
    contract: dict | None = None
    knowledge_graph: dict | None = None
    timeline: dict | None = None
    questions: dict | None = None
    presentation: dict | None = None


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


class PodcastSpeaker(BaseModel):
    name: str
    voice: str


class PodcastRequest(BaseModel):
    tone: str = "popular"  # scientific | popular
    speakers: list[PodcastSpeaker] = []


class QuizCheckRequest(BaseModel):
    question: str
    correct_answer: str
    user_answer: str


class MultiSearchRequest(BaseModel):
    notebook_ids: list[str] = Field(min_length=1, max_length=10)
    query: str


class QuestionsRequest(BaseModel):
    context: str = "general"  # general | credit | legal | financial


class CompareRequest(BaseModel):
    notebook_ids: list[str] = Field(min_length=2, max_length=2)

    @model_validator(mode="before")
    @classmethod
    def _normalize_payload(cls, value: Any) -> Any:
        return _normalize_pair_payload(
            value,
            "notebook_ids",
            (
                ("notebook_id_a", "notebook_id_b"),
                ("notebookIdA", "notebookIdB"),
                ("left_id", "right_id"),
                ("leftId", "rightId"),
                ("first_id", "second_id"),
                ("firstId", "secondId"),
                ("id1", "id2"),
                ("a", "b"),
            ),
        )


class PresentationRequest(BaseModel):
    title: str = ""
    style: str = "business"  # business | academic | popular


class UrlSourceRequest(BaseModel):
    url: str


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
    notebook = await create_notebook(settings.db_path, user_id, req.title, req.contour)
    return NotebookResponse(**notebook)


@router.get(
    "",
    response_model=list[NotebookListItem],
    summary="Список блокнотов",
    description="Возвращает все блокноты текущего пользователя (только id и название).",
)
async def list_all(user_id: str = Depends(require_auth)) -> list[NotebookListItem]:
    notebooks = await list_notebooks(settings.db_path, user_id)
    return [NotebookListItem(**nb) for nb in notebooks]


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
    return await _notebook_response(notebook_id, user_id)


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
    await _owned_notebook(notebook_id, user_id)
    await update_notebook_title(settings.db_path, notebook_id, req.title)
    return await _notebook_response(notebook_id, user_id)


@router.patch(
    "/{notebook_id}/contour",
    response_model=NotebookResponse,
    summary="Переключить контур блокнота",
    description="""
Переключает блокнот между открытым и закрытым контуром обработки данных.

- `open` — данные обрабатываются через внешний API (быстро, качественно)
- `closed` — все запросы идут через локальные модели (ollama + whisper), данные не покидают сервер

Настройка применяется ко всем последующим генерациям в этом блокноте.
    """,
)
async def set_contour(
    notebook_id: str,
    req: ContourRequest,
    user_id: str = Depends(require_auth),
) -> NotebookResponse:
    await _owned_notebook(notebook_id, user_id)
    await update_notebook_contour(settings.db_path, notebook_id, req.contour)
    return await _notebook_response(notebook_id, user_id)


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
    notebook = await _owned_notebook(notebook_id, user_id)
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
    description="Загружает файл (PDF, DOCX, TXT, CSV, XLSX) в блокнот. Документ автоматически разбивается на чанки и индексируется для поиска. Таблицы конвертируются в текст. Возвращает `id` источника.",
)
async def upload_source(
    notebook_id: str,
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    user_id: str = Depends(require_auth),
) -> dict[str, Any]:
    notebook = await _owned_notebook(notebook_id, user_id)
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
    source = await create_source(
        settings.db_path,
        notebook_id,
        file.filename or "document",
        rag_data.get("chunks", 0),
        status="ready",
        source_id=rag_data.get("source_id"),
    )
    await clear_notebook_cache(settings.db_path, notebook_id)
    preview = rag_data.get("preview", "")
    if preview.strip():
        background_tasks.add_task(_run_autotag, client, source["id"], preview, notebook.get("contour", "open"))
    return source


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
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    user_id: str = Depends(require_auth),
) -> dict[str, Any]:
    notebook = await _owned_notebook(notebook_id, user_id)
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
    source = await create_source(
        settings.db_path,
        notebook_id,
        txt_filename,
        rag_data.get("chunks", 0),
        status="ready",
    )
    await clear_notebook_cache(settings.db_path, notebook_id)
    preview = text[:500]
    if preview.strip():
        background_tasks.add_task(_run_autotag, client, source["id"], preview, notebook.get("contour", "open"))
    return source


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
    notebook = await _owned_notebook(notebook_id, user_id)
    source = await get_source(settings.db_path, source_id)
    if not source or source["notebook_id"] != notebook_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Источник не найден")

    client: httpx.AsyncClient = request.app.state.http_client
    try:
        await client.delete(_rag(f"/collection/{notebook_id}/sources/{source_id}"))
    except Exception:
        pass
    await delete_source(settings.db_path, source_id)
    await clear_notebook_cache(settings.db_path, notebook_id)


# ── URL source ────────────────────────────────────────────────────────────────

def _is_youtube(url: str) -> bool:
    return "youtube.com/watch" in url or "youtu.be/" in url


def _youtube_video_id(url: str) -> str:
    import re
    m = re.search(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})", url)
    if not m:
        raise ValueError("Не удалось извлечь ID видео из URL")
    return m.group(1)


def _fetch_youtube_text(video_id: str) -> tuple[str, str]:
    """Возвращает (текст транскрипции, название для filename)."""
    from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=["ru", "en"])
    except NoTranscriptFound:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
    except TranscriptsDisabled:
        raise ValueError("Субтитры для этого видео отключены")
    text = " ".join(entry["text"] for entry in transcript)
    return text, f"youtube_{video_id}.txt"


def _fetch_web_text(url: str) -> tuple[str, str]:
    """Возвращает (текст страницы, название для filename)."""
    import trafilatura
    from urllib.parse import urlparse
    html = trafilatura.fetch_url(url)
    if not html:
        raise ValueError("Не удалось загрузить страницу")
    text = trafilatura.extract(html, include_comments=False, include_tables=True)
    if not text:
        raise ValueError("Не удалось извлечь текст со страницы")
    domain = urlparse(url).netloc.replace("www.", "")
    return text, f"{domain}.txt"


@router.post(
    "/{notebook_id}/sources/url",
    status_code=status.HTTP_201_CREATED,
    summary="Добавить источник по URL",
    description="""
Загружает контент по URL и добавляет его как источник в блокнот.

**Поддерживаемые типы:**
- Веб-страницы — извлекается основной текст (без рекламы и навигации)
- YouTube — извлекаются субтитры/транскрипция видео
- PDF по прямой ссылке — скачивается и парсится как обычный PDF

**Пример запроса:**
```json
{"url": "https://youtube.com/watch?v=dQw4w9WgXcQ"}
```
    """,
)
async def upload_source_url(
    notebook_id: str,
    req: UrlSourceRequest,
    request: Request,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(require_auth),
) -> dict[str, Any]:
    import asyncio
    notebook = await _owned_notebook(notebook_id, user_id)
    client: httpx.AsyncClient = request.app.state.http_client
    url = req.url.strip()

    # Определяем тип URL и извлекаем текст
    if _is_youtube(url):
        try:
            video_id = _youtube_video_id(url)
            text, filename = await asyncio.to_thread(_fetch_youtube_text, video_id)
        except Exception as exc:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    elif url.lower().endswith(".pdf"):
        # PDF по прямой ссылке — скачиваем и отправляем в rag_service как файл
        try:
            pdf_resp = await client.get(url, follow_redirects=True, timeout=60.0)
            pdf_resp.raise_for_status()
        except Exception as exc:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=f"Не удалось скачать PDF: {exc}")
        from urllib.parse import urlparse
        filename = urlparse(url).path.split("/")[-1] or "document.pdf"
        try:
            rag_resp = await client.post(
                _rag("/upload"),
                files={"file": (filename, io.BytesIO(pdf_resp.content), "application/pdf")},
                data={"notebook_id": notebook_id},
            )
            rag_resp.raise_for_status()
        except httpx.RequestError as exc:
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc))
        except httpx.HTTPStatusError as exc:
            raise HTTPException(status_code=exc.response.status_code, detail=exc.response.text)
        rag_data = rag_resp.json()
        source = await create_source(settings.db_path, notebook_id, filename, rag_data.get("chunks", 0), status="ready")
        await clear_notebook_cache(settings.db_path, notebook_id)
        preview = rag_data.get("preview", "")
        if preview.strip():
            background_tasks.add_task(_run_autotag, client, source["id"], preview, notebook.get("contour", "open"))
        return source
    else:
        try:
            text, filename = await asyncio.to_thread(_fetch_web_text, url)
        except Exception as exc:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))

    # Для YouTube и веб-страниц — отправляем текст как TXT в rag_service
    try:
        rag_resp = await client.post(
            _rag("/upload"),
            files={"file": (filename, io.BytesIO(text.encode("utf-8")), "text/plain")},
            data={"notebook_id": notebook_id},
        )
        rag_resp.raise_for_status()
    except httpx.RequestError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc))
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=exc.response.text)

    rag_data = rag_resp.json()
    source = await create_source(settings.db_path, notebook_id, filename, rag_data.get("chunks", 0), status="ready")
    await clear_notebook_cache(settings.db_path, notebook_id)
    preview = text[:500]
    if preview.strip():
        background_tasks.add_task(_run_autotag, client, source["id"], preview, notebook.get("contour", "open"))
    return source


# ── Chat (SSE stream) ─────────────────────────────────────────────────────────

@router.post(
    "/{notebook_id}/chat",
    summary="Чат с документами",
    description="""
Задаёт вопрос по содержимому блокнота. Возвращает **SSE-стрим** (Server-Sent Events).

**Формат первого события** (содержит источники):
```
data: {"delta": "фрагмент текста", "sources": ["текст чанка 1", ...]}
```

**Формат последующих событий** (только текст):
```
data: {"delta": "следующий фрагмент"}
```

**Завершение стрима:**
```
data: [DONE]
```

Поле `sources` присутствует **только в первом событии** стрима. В остальных событиях его нет.

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
    import json as _json_mod
    notebook = await _owned_notebook(notebook_id, user_id)
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
            headers=_contour_headers(notebook),
        )
        rag_resp = await client.send(rag_request, stream=True)
    except httpx.RequestError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc))

    await save_chat_message(settings.db_path, notebook_id, "user", req.query)

    async def _stream():
        accumulated: list[str] = []
        sources: list[str] = []
        try:
            async for chunk in rag_resp.aiter_bytes():
                yield chunk
                for line in chunk.decode("utf-8", errors="ignore").splitlines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        parsed = _json_mod.loads(data)
                        if delta := parsed.get("delta"):
                            accumulated.append(delta)
                        if not sources and parsed.get("sources"):
                            sources = parsed["sources"]
                    except Exception:
                        pass
        finally:
            await rag_resp.aclose()
            if accumulated:
                await save_chat_message(
                    settings.db_path, notebook_id, "assistant", "".join(accumulated), sources
                )

    return StreamingResponse(
        _stream(),
        status_code=rag_resp.status_code,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get(
    "/{notebook_id}/chat/history",
    summary="История чата",
    description="Возвращает все сообщения чата для блокнота в хронологическом порядке.",
)
async def chat_history(
    notebook_id: str,
    user_id: str = Depends(require_auth),
) -> list[dict]:
    await _owned_notebook(notebook_id, user_id)
    return await get_chat_history(settings.db_path, notebook_id)


@router.delete(
    "/{notebook_id}/chat/history",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Очистить историю чата",
    description="Удаляет все сообщения чата для указанного блокнота. Операция необратима.",
)
async def delete_chat_history(
    notebook_id: str,
    user_id: str = Depends(require_auth),
) -> None:
    await _owned_notebook(notebook_id, user_id)
    await clear_chat_history(settings.db_path, notebook_id)


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

> Результат кэшируется. Повторный запрос возвращает кэш мгновенно. Для перегенерации: `?force=true`
    """,
)
async def summary(
    notebook_id: str,
    req: SummaryRequest,
    request: Request,
    user_id: str = Depends(require_auth),
) -> dict[str, Any]:
    notebook = await _owned_notebook(notebook_id, user_id)
    client: httpx.AsyncClient = request.app.state.http_client
    text = await _notebook_rag_text(client, notebook_id,
        query="основные идеи, ключевые темы, главные выводы и результаты документа",
        top_k=40)

    try:
        resp = await client.post(
            _content("/summary"),
            json={"text": text, "style": req.style},
            headers=_contour_headers(notebook),
            timeout=300.0,
        )
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

**Ответ** — иерархическое дерево в JSON (максимум 3 уровня):
```json
{
  "title": "Главная тема",
  "children": [
    {
      "title": "Раздел 1",
      "children": [
        {"title": "Подпункт 1.1", "children": []},
        {"title": "Подпункт 1.2", "children": []}
      ]
    },
    {
      "title": "Раздел 2",
      "children": []
    }
  ]
}
```

> Результат кэшируется. Повторный запрос возвращает кэш мгновенно. Для перегенерации: `?force=true`
    """,
)
async def mindmap(
    notebook_id: str,
    request: Request,
    user_id: str = Depends(require_auth),
) -> dict[str, Any]:
    notebook = await _owned_notebook(notebook_id, user_id)
    client: httpx.AsyncClient = request.app.state.http_client
    text = await _notebook_rag_text(client, notebook_id,
        query="структура документа, разделы, темы, подтемы и ключевые концепции",
        top_k=30)

    try:
        resp = await client.post(_content("/mindmap"), json={"text": text}, headers=_contour_headers(notebook), timeout=300.0)
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

> Результат кэшируется. Повторный запрос возвращает кэш мгновенно. Для перегенерации: `?force=true`
    """,
)
async def flashcards(
    notebook_id: str,
    req: FlashcardsRequest,
    request: Request,
    user_id: str = Depends(require_auth),
) -> dict[str, Any]:
    notebook = await _owned_notebook(notebook_id, user_id)
    client: httpx.AsyncClient = request.app.state.http_client
    text = await _notebook_rag_text(client, notebook_id,
        query="важные факты, определения, ключевые понятия и термины для запоминания",
        top_k=50)

    try:
        resp = await client.post(_content("/flashcards"), json={"text": text, "count": req.count}, headers=_contour_headers(notebook))
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

> Результат кэшируется. Повторный запрос возвращает кэш мгновенно. Для перегенерации: `?force=true`
    """,
)
async def podcast(
    notebook_id: str,
    req: PodcastRequest,
    request: Request,
    user_id: str = Depends(require_auth),
) -> dict[str, Any]:
    notebook = await _owned_notebook(notebook_id, user_id)
    client: httpx.AsyncClient = request.app.state.http_client
    text = await _notebook_rag_text(client, notebook_id,
        query="интересные факты, главные темы, ключевые идеи для обсуждения в подкасте",
        top_k=30)

    try:
        podcast_payload: dict = {"text": text, "tone": req.tone}
        if req.speakers:
            podcast_payload["speakers"] = [s.model_dump() for s in req.speakers]
        resp = await client.post(_content("/podcast"), json=podcast_payload, headers=_contour_headers(notebook))
        resp.raise_for_status()
        data = resp.json()
        # Переписываем URL чтобы аудио шло через gateway, а не напрямую на content_service:8002
        if "audio_url" in data:
            data["audio_url"] = f"/api/content{data['audio_url']}"
        await save_notebook_content(settings.db_path, notebook_id, "podcast_url", data.get("audio_url", ""))
        await save_notebook_content(settings.db_path, notebook_id, "podcast_script", _json.dumps(data.get("script", []), ensure_ascii=False))
        return data
    except httpx.RequestError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc))


@router.get(
    "/voices",
    summary="Список доступных голосов TTS для подкаста",
)
async def list_voices(
    request: Request,
    user_id: str = Depends(require_auth),
) -> list[dict]:
    client: httpx.AsyncClient = request.app.state.http_client
    try:
        resp = await client.get(_content("/voices"))
        resp.raise_for_status()
        return resp.json()
    except httpx.RequestError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc))


@router.get(
    "/voices/{voice_id}/sample",
    summary="Аудиосэмпл голоса",
)
async def voice_sample(
    voice_id: str,
    request: Request,
    user_id: str = Depends(require_auth),
):
    client: httpx.AsyncClient = request.app.state.http_client
    try:
        resp = await client.get(_content(f"/voices/{voice_id}/sample"))
        resp.raise_for_status()
        return StreamingResponse(
            resp.aiter_bytes(),
            media_type="audio/mpeg",
            headers={"Cache-Control": "public, max-age=86400"},
        )
    except httpx.RequestError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc))


@router.post(
    "/{notebook_id}/contract",
    summary="Анализ договора",
    description="""
Извлекает из документов блокнота структурированный анализ договора:
стороны, предмет, ключевые условия, обязательства, риски, сроки, штрафы.

Идеально для юридических документов, кредитных договоров, соглашений.

**Ответ:**
```json
{
  "parties": ["ООО Ромашка", "Банк"],
  "subject": "Кредитный договор на сумму 5 млн руб.",
  "key_conditions": ["Ставка 14% годовых", "Срок 36 месяцев"],
  "obligations": [{"party": "Заёмщик", "text": "Погашать ежемесячно"}],
  "risks": ["Штраф 0.1% в день за просрочку"],
  "deadlines": ["Дата первого платежа — 01.05.2026"],
  "penalties": ["Неустойка 1% от суммы долга"]
}
```

> Результат кэшируется. Повторный запрос возвращает кэш мгновенно. Для перегенерации: `?force=true`
    """,
)
async def contract(
    notebook_id: str,
    request: Request,
    user_id: str = Depends(require_auth),
) -> dict[str, Any]:
    notebook = await _owned_notebook(notebook_id, user_id)
    client: httpx.AsyncClient = request.app.state.http_client
    text = await _notebook_text(client, notebook_id, max_length=_MAX_TEXT_LENGTH_EXTENDED)

    try:
        resp = await client.post(_content("/contract"), json={"text": text}, headers=_contour_headers(notebook), timeout=300.0)
        resp.raise_for_status()
        data = resp.json()
        await save_notebook_content(settings.db_path, notebook_id, "contract", _json.dumps(data, ensure_ascii=False))
        return data
    except httpx.RequestError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc))


@router.post(
    "/{notebook_id}/knowledge-graph",
    summary="Граф знаний",
    description="""
Строит граф знаний из документов блокнота: извлекает сущности (персоны, организации, концепции, события)
и связи между ними. Результат можно визуализировать через D3.js, Cytoscape, vis.js.

**Ответ:**
```json
{
  "nodes": [
    {"id": "bank", "label": "Банк Центр-Инвест", "type": "org"},
    {"id": "credit_rate", "label": "Ставка 14%", "type": "term"}
  ],
  "edges": [
    {"source": "bank", "target": "credit_rate", "label": "устанавливает"}
  ]
}
```

> Результат кэшируется. Повторный запрос возвращает кэш мгновенно. Для перегенерации: `?force=true`
    """,
)
async def knowledge_graph(
    notebook_id: str,
    request: Request,
    user_id: str = Depends(require_auth),
) -> dict[str, Any]:
    notebook = await _owned_notebook(notebook_id, user_id)
    client: httpx.AsyncClient = request.app.state.http_client
    text = await _notebook_rag_text(client, notebook_id,
        query="сущности, организации, персоны, концепции и связи между ними",
        top_k=50)

    try:
        resp = await client.post(_content("/knowledge-graph"), json={"text": text}, headers=_contour_headers(notebook), timeout=300.0)
        resp.raise_for_status()
        data = resp.json()
        await save_notebook_content(settings.db_path, notebook_id, "knowledge_graph", _json.dumps(data, ensure_ascii=False))
        return data
    except httpx.RequestError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc))


@router.post(
    "/{notebook_id}/flashcards/check",
    summary="Проверить ответ (автотест)",
    description="""
Проверяет ответ пользователя на вопрос флэш-карточки. LLM оценивает правильность,
выставляет оценку 0–1 и объясняет ошибки. Геймификация обучения.

**Пример запроса:**
```json
{
  "question": "Что такое RAG?",
  "correct_answer": "Retrieval-Augmented Generation — метод...",
  "user_answer": "Это когда нейросеть ищет информацию в базе данных"
}
```

**Ответ:**
```json
{
  "is_correct": true,
  "score": 0.8,
  "feedback": "Верно в целом! Ты уловил суть, но не упомянул..."
}
```
    """,
)
async def check_flashcard(
    notebook_id: str,
    req: QuizCheckRequest,
    request: Request,
    user_id: str = Depends(require_auth),
) -> dict[str, Any]:
    notebook = await _owned_notebook(notebook_id, user_id)
    client: httpx.AsyncClient = request.app.state.http_client

    try:
        resp = await client.post(
            _content("/flashcards/check"),
            json={
                "question": req.question,
                "correct_answer": req.correct_answer,
                "user_answer": req.user_answer,
            },
            headers=_contour_headers(notebook),
        )
        resp.raise_for_status()
        return resp.json()
    except httpx.RequestError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc))


@router.post(
    "/search",
    summary="Поиск по нескольким блокнотам",
    description="""
Задаёт вопрос сразу по нескольким блокнотам одновременно. LLM анализирует
объединённый контекст всех документов и формирует единый ответ.

**Пример запроса:**
```json
{
  "notebook_ids": ["id1", "id2", "id3"],
  "query": "Какая ставка рефинансирования упоминается в договорах?"
}
```

**Ответ:**
```json
{
  "answer": "В договоре с ООО Ромашка указана ставка 14%...",
  "notebooks_searched": ["Договор №1", "Кредитный договор", "Допсоглашение"]
}
```
    """,
)
async def multi_search(
    req: MultiSearchRequest,
    request: Request,
    user_id: str = Depends(require_auth),
) -> dict[str, Any]:
    client: httpx.AsyncClient = request.app.state.http_client

    texts: list[str] = []
    titles: list[str] = []
    contours: list[str] = []

    for nb_id in req.notebook_ids:
        nb = await _owned_notebook(nb_id, user_id)
        titles.append(nb["title"])
        contours.append(nb.get("contour", "open"))
        try:
            text = await _notebook_text(client, nb_id)
            texts.append(f"[Блокнот: {nb['title']}]\n{text}")
        except HTTPException:
            texts.append(f"[Блокнот: {nb['title']}]\n(документы не загружены)")

    # Наиболее закрытый контур из всех блокнотов
    contour = "closed" if "closed" in contours else "open"
    combined = "\n\n---\n\n".join(texts)
    if len(combined) > _MAX_TEXT_LENGTH:
        combined = combined[:_MAX_TEXT_LENGTH].rsplit(" ", 1)[0]

    try:
        resp = await client.post(
            _content("/answer"),
            json={"text": combined, "question": req.query},
            headers={"x-contour": contour},
        )
        resp.raise_for_status()
        data = resp.json()
        return {"answer": data.get("answer", ""), "notebooks_searched": titles}
    except httpx.RequestError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc))


@router.post(
    "/{notebook_id}/timeline",
    summary="Временная шкала",
    description="""
Извлекает все события с датами из документов блокнота и строит хронологическую шкалу.

Незаменимо для кредитных историй, договоров, судебных дел, проектной документации.

**Типы событий:** `payment` · `deadline` · `event` · `agreement` · `violation` · `other`

**Ответ:**
```json
{
  "events": [
    {"date": "01.03.2026", "title": "Подписание договора", "description": "Кредит на 5 млн руб.", "type": "agreement"},
    {"date": "01.04.2026", "title": "Первый платёж",      "description": "150 000 руб.",          "type": "payment"},
    {"date": "30.06.2026", "title": "Дедлайн отчётности", "description": "Предоставить баланс",    "type": "deadline"}
  ]
}
```

> Результат кэшируется. Повторный запрос возвращает кэш мгновенно. Для перегенерации: `?force=true`
    """,
)
async def timeline(
    notebook_id: str,
    request: Request,
    user_id: str = Depends(require_auth),
) -> dict[str, Any]:
    notebook = await _owned_notebook(notebook_id, user_id)
    client: httpx.AsyncClient = request.app.state.http_client
    text = await _notebook_rag_text(client, notebook_id,
        query="даты, события, хронология, временная последовательность",
        top_k=50)

    try:
        resp = await client.post(_content("/timeline"), json={"text": text}, headers=_contour_headers(notebook), timeout=300.0)
        resp.raise_for_status()
        data = resp.json()
        await save_notebook_content(settings.db_path, notebook_id, "timeline", _json.dumps(data, ensure_ascii=False))
        return data
    except httpx.RequestError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc))


_QUESTIONS_RAG_QUERY = {
    "credit":    "кредитоспособность, залог, поручительство, доходы, обязательства, риски кредита",
    "legal":     "обязательства сторон, условия расторжения, штрафы, санкции, неясные формулировки",
    "financial": "финансовые показатели, выручка, прибыль, долг, активы, риски",
    "general":   "пробелы в информации, неясные моменты, риски, требуемые уточнения",
}


@router.post(
    "/{notebook_id}/questions",
    summary="Вопросы к документу",
    description="""
Анализирует документы блокнота и генерирует список вопросов, на которые они **не дают ответа**.
Помогает аналитику понять, какую информацию нужно запросить дополнительно.

**Параметр `context`** — роль, с позиции которой анализируется документ:
- `general` — универсальный анализ (по умолчанию)
- `credit` — кредитный аналитик банка
- `legal` — юрист
- `financial` — финансовый аналитик

**Категории вопросов:**
- `missing_info` — информация отсутствует
- `clarification` — формулировка неясна
- `risk` — требует уточнения из-за риска
- `required_doc` — нужен дополнительный документ

**Приоритеты:** `high` (блокирующие) · `medium` · `low`

**Ответ:**
```json
{
  "questions": [
    {"question": "Кто является поручителем?", "category": "missing_info", "priority": "high"},
    {"question": "Что значит 'существенное нарушение' в п. 4.2?", "category": "clarification", "priority": "medium"}
  ],
  "summary": "Документ не содержит информации о залоге и поручительстве..."
}
```

> Результат кэшируется. Повторный запрос возвращает кэш мгновенно. Для перегенерации: `?force=true`
    """,
)
async def generate_questions(
    notebook_id: str,
    req: QuestionsRequest,
    request: Request,
    user_id: str = Depends(require_auth),
) -> dict[str, Any]:
    notebook = await _owned_notebook(notebook_id, user_id)
    client: httpx.AsyncClient = request.app.state.http_client
    rag_query = _QUESTIONS_RAG_QUERY.get(req.context, _QUESTIONS_RAG_QUERY["general"])
    text = await _notebook_rag_text(client, notebook_id, query=rag_query, top_k=50)

    try:
        resp = await client.post(_content("/questions"), json={"text": text, "context": req.context}, headers=_contour_headers(notebook), timeout=300.0)
        resp.raise_for_status()
        data = resp.json()
        await save_notebook_content(settings.db_path, notebook_id, "questions", _json.dumps(data, ensure_ascii=False))
        return data
    except httpx.RequestError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc))


@router.post(
    "/compare",
    summary="Сравнить два блокнота",
    description="""
Сравнивает документы из двух блокнотов и возвращает список расхождений.
Незаменимо для сравнения версий договора, двух предложений от разных банков,
оригинала и дополнительного соглашения.

**Тело запроса:** `{"notebook_ids": ["id_первого", "id_второго"]}`

**Ответ:**
```json
{
  "changes": [
    {
      "section": "п. 3.1 Процентная ставка",
      "type": "modified",
      "description": "Ставка изменена с 12% до 14% годовых",
      "quote_a": "процентная ставка составляет 12% годовых",
      "quote_b": "процентная ставка составляет 14% годовых",
      "severity": "critical"
    }
  ],
  "summary": "Новая версия ужесточает штрафные условия...",
  "risk_level": "high",
  "label_a": "Название блокнота 1",
  "label_b": "Название блокнота 2"
}
```

**Типы изменений:** `added` · `removed` · `modified` · `conflict`
**Severity:** `critical` · `warning` · `info`
    """,
)
async def compare_notebooks(
    req: CompareRequest,
    request: Request,
    user_id: str = Depends(require_auth),
) -> dict[str, Any]:
    nb_a = await _owned_notebook(req.notebook_ids[0], user_id)
    nb_b = await _owned_notebook(req.notebook_ids[1], user_id)
    # Используем наиболее закрытый контур из двух блокнотов
    contour = "closed" if "closed" in (nb_a.get("contour", "open"), nb_b.get("contour", "open")) else "open"
    client: httpx.AsyncClient = request.app.state.http_client

    text_a = await _notebook_text(client, req.notebook_ids[0])
    text_b = await _notebook_text(client, req.notebook_ids[1])

    try:
        resp = await client.post(
            _content("/compare"),
            json={
                "text_a": text_a,
                "text_b": text_b,
                "label_a": nb_a["title"],
                "label_b": nb_b["title"],
            },
            headers={"x-contour": contour},
        )
        resp.raise_for_status()
        data = resp.json()
        data["label_a"] = nb_a["title"]
        data["label_b"] = nb_b["title"]
        return data
    except httpx.RequestError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc))


class SourceCompareRequest(BaseModel):
    source_ids: list[str] = Field(min_length=2, max_length=2)

    @model_validator(mode="before")
    @classmethod
    def _normalize_payload(cls, value: Any) -> Any:
        return _normalize_pair_payload(
            value,
            "source_ids",
            (
                ("source_id_a", "source_id_b"),
                ("sourceIdA", "sourceIdB"),
                ("left_id", "right_id"),
                ("leftId", "rightId"),
                ("first_id", "second_id"),
                ("firstId", "secondId"),
                ("id1", "id2"),
                ("a", "b"),
            ),
        )


@router.post(
    "/{notebook_id}/sources/compare",
    summary="Сравнить два файла внутри блокнота",
    description="""
Сравнивает два источника из одного блокнота. Удобно для случая «оригинал vs изменённая версия»
без создания отдельных блокнотов.

**Тело запроса:** `{"source_ids": ["id_первого", "id_второго"]}`

Возвращает список изменений с цитатами из обоих документов.
    """,
)
async def compare_sources(
    notebook_id: str,
    req: SourceCompareRequest,
    request: Request,
    user_id: str = Depends(require_auth),
) -> dict[str, Any]:
    notebook_id = _normalize_notebook_id(notebook_id)
    notebook = await _owned_notebook(notebook_id, user_id)
    client: httpx.AsyncClient = request.app.state.http_client

    src_a = await get_source(settings.db_path, req.source_ids[0])
    src_b = await get_source(settings.db_path, req.source_ids[1])
    for src in (src_a, src_b):
        if not src or src["notebook_id"] != notebook_id:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Источник не найден")

    async def _source_text(source_id: str) -> str:
        try:
            source = src_a if source_id == req.source_ids[0] else src_b
            resp = await client.get(
                _rag(f"/notebook/{notebook_id}/sources/{source_id}/content"),
                params={"filename": source["filename"]},
            )
            resp.raise_for_status()
            return resp.json().get("text", "")
        except httpx.RequestError as exc:
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc))
        except httpx.HTTPStatusError as exc:
            raise HTTPException(status_code=exc.response.status_code, detail=exc.response.text)

    import asyncio as _asyncio
    text_a, text_b = await _asyncio.gather(_source_text(req.source_ids[0]), _source_text(req.source_ids[1]))

    if not text_a.strip() or not text_b.strip():
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Один из источников пуст")

    try:
        resp = await client.post(
            _content("/compare"),
            json={
                "text_a": text_a[:_MAX_TEXT_LENGTH],
                "text_b": text_b[:_MAX_TEXT_LENGTH],
                "label_a": src_a["filename"],
                "label_b": src_b["filename"],
            },
            headers=_contour_headers(notebook),
            timeout=300.0,
        )
        resp.raise_for_status()
        data = resp.json()
        data["label_a"] = src_a["filename"]
        data["label_b"] = src_b["filename"]
        return data
    except httpx.RequestError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc))


@router.post(
    "/{notebook_id}/presentation/preview",
    summary="Структура презентации (превью)",
    description="""
Генерирует структуру слайдов в JSON — для отображения превью в браузере перед скачиванием.

**Параметр `style`:** `business` · `academic` · `popular`

**Ответ:**
```json
{
  "slides": [
    {"type": "title", "title": "Анализ кредитного портфеля", "subtitle": "Q1 2026"},
    {"type": "content", "title": "Ключевые показатели", "bullets": ["NPL 2.3%", "ROE 18%"]},
    {"type": "summary", "title": "Итоги", "bullets": ["Портфель вырос на 12%"]}
  ]
}
```
    """,
)
async def presentation_preview(
    notebook_id: str,
    req: PresentationRequest,
    request: Request,
    user_id: str = Depends(require_auth),
) -> dict[str, Any]:
    notebook = await _owned_notebook(notebook_id, user_id)
    client: httpx.AsyncClient = request.app.state.http_client
    text = await _notebook_text(client, notebook_id)

    try:
        resp = await client.post(
            _content("/presentation/preview"),
            json={"text": text, "title": req.title or notebook["title"], "style": req.style},
            headers=_contour_headers(notebook),
        )
        resp.raise_for_status()
        data = resp.json()
        await save_notebook_content(settings.db_path, notebook_id, "presentation", _json.dumps(data, ensure_ascii=False))
        return data
    except httpx.RequestError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc))


@router.post(
    "/{notebook_id}/presentation/download",
    summary="Скачать презентацию (PPTX)",
    description="""
Генерирует и возвращает готовый файл `.pptx` для скачивания.
Презентация оформлена в корпоративном стиле (тёмно-синий + золотой).

**Параметр `style`:** `business` · `academic` · `popular`

Возвращает бинарный файл `presentation.pptx`.
    """,
)
async def presentation_download(
    notebook_id: str,
    req: PresentationRequest,
    request: Request,
    user_id: str = Depends(require_auth),
) -> StreamingResponse:
    notebook = await _owned_notebook(notebook_id, user_id)
    client: httpx.AsyncClient = request.app.state.http_client
    text = await _notebook_text(client, notebook_id)

    try:
        resp = await client.post(
            _content("/presentation/download"),
            json={"text": text, "title": req.title or notebook["title"], "style": req.style},
            headers=_contour_headers(notebook),
            timeout=120.0,
        )
        resp.raise_for_status()
    except httpx.RequestError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc))

    return StreamingResponse(
        resp.aiter_bytes(),
        media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        headers={"Content-Disposition": f'attachment; filename="presentation.pptx"'},
    )


@router.post(
    "/{notebook_id}/autotag",
    summary="Автотегирование документов блокнота",
    description="""
Определяет тип и теги документов в блокноте на основе их содержимого.

**Ответ:**
```json
{
  "doc_type": "договор",
  "tags": ["кредитный договор", "ипотека", "Сбербанк"]
}
```

**Типы документов:** договор · отчёт · инструкция · письмо · книга · судебный документ · научная статья · новость · прочее
    """,
)
async def autotag_notebook(
    notebook_id: str,
    request: Request,
    user_id: str = Depends(require_auth),
) -> dict[str, Any]:
    notebook = await _owned_notebook(notebook_id, user_id)
    client: httpx.AsyncClient = request.app.state.http_client

    try:
        rag_resp = await client.get(_rag(f"/notebook/{notebook_id}/content"))
        rag_resp.raise_for_status()
        text = rag_resp.json().get("text", "")[:500]
    except httpx.RequestError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc))
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=exc.response.text)

    if not text.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Блокнот пуст — загрузите хотя бы один источник",
        )

    try:
        resp = await client.post(
            _content("/autotag"),
            json={"text": text},
            headers=_contour_headers(notebook),
        )
        resp.raise_for_status()
        return resp.json()
    except httpx.RequestError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc))
