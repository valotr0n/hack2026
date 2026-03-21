from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from ..rag import delete_collection, delete_source_chunks, get_notebook_content

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


@router.delete("/collection/{collection_id}/sources/{source_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_source_chunks(collection_id: str, source_id: str) -> None:
    await delete_source_chunks(collection_id, source_id)


@router.delete("/collection/{collection_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_collection(collection_id: str) -> None:
    await delete_collection(collection_id)
