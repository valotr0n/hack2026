from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from ..dependencies import get_embedding_model
from ..rag import delete_collection, delete_source_chunks, fetch_embeddings, get_notebook_content, get_source_content, search_notebook_text

router = APIRouter(tags=["notebooks"])


class NotebookContentResponse(BaseModel):
    text: str
    sources: list[str]


@router.get("/notebook/{notebook_id}/content", response_model=NotebookContentResponse)
async def get_content(notebook_id: str) -> NotebookContentResponse:
    content = await get_notebook_content(notebook_id)
    if not content["text"].strip():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Коллекция пуста или не найдена",
        )
    return NotebookContentResponse(**content)


@router.get("/notebook/{notebook_id}/sources/{source_id}/content")
async def get_source_text(
    notebook_id: str,
    source_id: str,
    filename: str | None = Query(None),
) -> dict:
    content = await get_source_content(notebook_id, source_id, filename)
    if not content["text"].strip():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Источник не найден или пуст",
        )
    return content


class NotebookSearchRequest(BaseModel):
    query: str
    top_k: int = 40


class NotebookSearchResponse(BaseModel):
    text: str
    chunks_found: int


@router.post("/notebook/{notebook_id}/search", response_model=NotebookSearchResponse)
async def semantic_search(
    notebook_id: str,
    req: NotebookSearchRequest,
    embedding_model: SentenceTransformer = Depends(get_embedding_model),
) -> NotebookSearchResponse:
    """Семантический поиск по ноутбуку — для фич вместо полного текста."""
    query_embedding = (
        await fetch_embeddings(embedding_model, [req.query], prompt_name="search_query")
    )[0]
    text = await search_notebook_text(notebook_id, query_embedding, top_k=req.top_k)
    if not text.strip():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Коллекция пуста или не найдена",
        )
    return NotebookSearchResponse(text=text, chunks_found=text.count("\n\n") + 1)


@router.delete("/collection/{collection_id}/sources/{source_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_source_chunks(collection_id: str, source_id: str) -> None:
    await delete_source_chunks(collection_id, source_id)


@router.delete("/collection/{collection_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_collection(collection_id: str) -> None:
    await delete_collection(collection_id)
