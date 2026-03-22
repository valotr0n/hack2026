import logging
import time

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI
from sentence_transformers import SentenceTransformer

from ..dependencies import get_embedding_model, get_llm_client
from ..rag import (
    build_chat_messages,
    build_system_prompt,
    create_llm_stream,
    fetch_embeddings,
    search_document_chunks,
    stream_chat_response,
)
from ..schemas import ChatRequest

logger = logging.getLogger("rag_service.chat")
router = APIRouter(tags=["rag"])


@router.post("/chat")
async def chat(
    payload: ChatRequest,
    embedding_model: SentenceTransformer = Depends(get_embedding_model),
    llm_client: AsyncOpenAI = Depends(get_llm_client),
) -> StreamingResponse:
    if not payload.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query must not be empty.",
        )

    t0 = time.perf_counter()
    logger.info("Chat started doc_id=%s query_chars=%d history_len=%d", payload.doc_id, len(payload.query), len(payload.history))

    query_embedding = (
        await fetch_embeddings(
            embedding_model,
            [payload.query],
            prompt_name="search_query",
        )
    )[0]
    logger.info("Query embedded doc_id=%s %.2fs", payload.doc_id, time.perf_counter() - t0)

    chunks = await search_document_chunks(
        doc_id=payload.doc_id,
        query_embedding=query_embedding,
    )
    logger.info("Chunks retrieved doc_id=%s chunks=%d %.2fs", payload.doc_id, len(chunks), time.perf_counter() - t0)

    system_prompt, sources = build_system_prompt(chunks)
    messages = build_chat_messages(system_prompt, payload.history, payload.query)
    logger.info("LLM stream starting doc_id=%s context_chars=%d", payload.doc_id, len(system_prompt))
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