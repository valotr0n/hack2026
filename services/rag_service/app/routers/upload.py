import logging
from time import perf_counter
from typing import Optional
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile, status

from ..rag import (
    build_upload_preview,
    create_document_collection,
    extract_text_from_upload,
    fetch_embeddings,
    get_collection_embedding_backend,
    split_text_into_chunks,
)
from ..schemas import UploadResponse

logger = logging.getLogger(__name__)
router = APIRouter(tags=["rag"])


@router.post("/upload", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    notebook_id: Optional[str] = Form(None),
) -> UploadResponse:
    started_at = perf_counter()
    source_id = str(uuid4())
    filename = file.filename or "document"
    existing_backend = await get_collection_embedding_backend(notebook_id) if notebook_id else None
    if existing_backend == "open":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                "Блокнот был проиндексирован внешним эмбеддером. "
                "Очистите источники и загрузите их заново после перехода на локальный режим."
            ),
        )

    embedding_model = request.app.state.embedding_model

    extract_started_at = perf_counter()
    full_text = await extract_text_from_upload(file)
    extract_elapsed = perf_counter() - extract_started_at

    chunk_started_at = perf_counter()
    chunks = split_text_into_chunks(full_text)
    chunk_elapsed = perf_counter() - chunk_started_at

    embed_started_at = perf_counter()
    embeddings = await fetch_embeddings(
        embedding_model,
        chunks,
        prompt_name="search_document",
    )
    embed_elapsed = perf_counter() - embed_started_at

    index_started_at = perf_counter()
    collection_id = await create_document_collection(
        embedding_dimension=len(embeddings[0]),
        filename=filename,
        chunks=chunks,
        embeddings=embeddings,
        notebook_id=notebook_id,
        source_id=source_id,
    )
    index_elapsed = perf_counter() - index_started_at

    total_elapsed = perf_counter() - started_at
    logger.info(
        "Upload processed: filename=%s notebook_id=%s backend=local chars=%d chunks=%d timings extract=%.2fs chunk=%.2fs embed=%.2fs index=%.2fs total=%.2fs",
        filename,
        notebook_id or "-",
        len(full_text),
        len(chunks),
        extract_elapsed,
        chunk_elapsed,
        embed_elapsed,
        index_elapsed,
        total_elapsed,
    )

    return UploadResponse(
        doc_id=collection_id,
        chunks=len(chunks),
        source_id=source_id,
        preview=build_upload_preview(full_text),
    )
