from __future__ import annotations

import io
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
    update_notebook_title,
)

router = APIRouter(prefix="/notebooks", tags=["notebooks"])

# Максимальный объём текста для передачи в LLM (~20k символов ≈ 5k токенов)
_MAX_TEXT_LENGTH = 20_000


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

@router.post("", response_model=NotebookResponse, status_code=status.HTTP_201_CREATED)
async def create(
    req: CreateNotebookRequest,
    user_id: str = Depends(require_auth),
) -> NotebookResponse:
    notebook = await create_notebook(settings.db_path, user_id, req.title)
    return NotebookResponse(**notebook)


@router.get("", response_model=list[NotebookResponse])
async def list_all(user_id: str = Depends(require_auth)) -> list[NotebookResponse]:
    notebooks = await list_notebooks(settings.db_path, user_id)
    result = []
    for nb in notebooks:
        sources = await list_sources(settings.db_path, nb["id"])
        result.append(NotebookResponse(**nb, sources=sources))
    return result


@router.get("/{notebook_id}", response_model=NotebookResponse)
async def get_one(
    notebook_id: str,
    user_id: str = Depends(require_auth),
) -> NotebookResponse:
    notebook = await _owned_notebook(notebook_id, user_id)
    sources = await list_sources(settings.db_path, notebook_id)
    return NotebookResponse(**notebook, sources=sources)


@router.patch("/{notebook_id}", response_model=NotebookResponse)
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


@router.delete("/{notebook_id}", status_code=status.HTTP_204_NO_CONTENT)
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

@router.post("/{notebook_id}/sources", status_code=status.HTTP_201_CREATED)
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


@router.delete("/{notebook_id}/sources/{source_id}", status_code=status.HTTP_204_NO_CONTENT)
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

@router.post("/{notebook_id}/chat")
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

@router.post("/{notebook_id}/summary")
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
        return resp.json()
    except httpx.RequestError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc))


@router.post("/{notebook_id}/mindmap")
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
        return resp.json()
    except httpx.RequestError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc))


@router.post("/{notebook_id}/flashcards")
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
        return resp.json()
    except httpx.RequestError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc))


@router.post("/{notebook_id}/podcast")
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
        return resp.json()
    except httpx.RequestError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc))
