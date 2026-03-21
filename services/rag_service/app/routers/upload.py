from typing import Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, UploadFile, status
from sentence_transformers import SentenceTransformer

from ..dependencies import get_embedding_model
from ..rag import (
    create_document_collection,
    extract_text_from_upload,
    fetch_embeddings,
    get_embedding_dimension,
    split_text_into_chunks,
)
from ..schemas import UploadResponse

router = APIRouter(tags=["rag"])


@router.post("/upload", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    notebook_id: Optional[str] = Form(None),
    embedding_model: SentenceTransformer = Depends(get_embedding_model),
) -> UploadResponse:
    source_id = str(uuid4())
    full_text = await extract_text_from_upload(file)
    chunks = split_text_into_chunks(full_text)
    embeddings = await fetch_embeddings(
        embedding_model,
        chunks,
        prompt_name="search_document",
    )
    collection_id = await create_document_collection(
        embedding_dimension=get_embedding_dimension(embedding_model),
        filename=file.filename or "document",
        chunks=chunks,
        embeddings=embeddings,
        notebook_id=notebook_id,
        source_id=source_id,
    )
    return UploadResponse(doc_id=collection_id, chunks=len(chunks), source_id=source_id)
