from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI

from ..dependencies import get_llm_client
from ..rag import (
    build_chat_messages,
    build_system_prompt,
    create_llm_stream,
    fetch_embeddings,
    get_collection_embedding_backend,
    search_document_chunks,
    stream_chat_response,
)
from ..schemas import ChatRequest

router = APIRouter(tags=["rag"])


@router.post("/chat")
async def chat(
    payload: ChatRequest,
    request: Request,
    llm_client: AsyncOpenAI = Depends(get_llm_client),
) -> StreamingResponse:
    if not payload.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query must not be empty.",
        )

    collection_backend = await get_collection_embedding_backend(payload.doc_id)
    if collection_backend == "open":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                "Блокнот был проиндексирован внешним эмбеддером. "
                "Очистите источники и загрузите их заново после перехода на локальный режим."
            ),
        )

    embedding_model = request.app.state.embedding_model
    query_embedding = (
        await fetch_embeddings(
            embedding_model,
            [payload.query],
            prompt_name="search_query",
        )
    )[0]
    chunks = await search_document_chunks(
        doc_id=payload.doc_id,
        query_embedding=query_embedding,
    )
    system_prompt, sources = build_system_prompt(chunks)
    messages = build_chat_messages(system_prompt, payload.history, payload.query)
    stream = await create_llm_stream(llm_client, messages)

    return StreamingResponse(
        stream_chat_response(stream, sources),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
