from fastapi import APIRouter, Depends, File, UploadFile, status
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
    embedding_model: SentenceTransformer = Depends(get_embedding_model),
) -> UploadResponse:
    full_text = await extract_text_from_upload(file)
    chunks = split_text_into_chunks(full_text)
    embeddings = await fetch_embeddings(
        embedding_model,
        chunks,
        prompt_name="search_document",
    )
    doc_id = await create_document_collection(
        embedding_dimension=get_embedding_dimension(embedding_model),
        filename=file.filename or "document",
        chunks=chunks,
        embeddings=embeddings,
    )
    return UploadResponse(doc_id=doc_id, chunks=len(chunks))