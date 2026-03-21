from __future__ import annotations

import asyncio
import inspect
import logging
from contextlib import asynccontextmanager, suppress

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI
from sentence_transformers import SentenceTransformer

from .config import settings
from .routers.chat import router as chat_router
from .routers.notebook_content import router as notebook_content_router
from .routers.upload import router as upload_router

logger = logging.getLogger(__name__)


async def _close_resource(resource: object) -> None:
    close = getattr(resource, "close", None)
    if not callable(close):
        return

    result = close()
    if inspect.isawaitable(result):
        await result


@asynccontextmanager
async def lifespan(app: FastAPI):
    embedding_model = SentenceTransformer(settings.embedder_model)
    llm_http_client = httpx.AsyncClient(verify=False, timeout=120.0)
    llm_client = AsyncOpenAI(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        http_client=llm_http_client,
    )

    app.state.embedding_model = embedding_model
    app.state.llm_client = llm_client
    vision_preload_task: asyncio.Task[None] | None = None
    if settings.vision_enabled and settings.vision_preload:
        from .vision import preload_vision_model

        vision_preload_task = asyncio.create_task(
            preload_vision_model(settings.vision_model_id),
            name="rag-service-vision-preload",
        )
        app.state.vision_preload_task = vision_preload_task
        logger.info("Scheduled background preload for vision model: %s", settings.vision_model_id)

    try:
        yield
    finally:
        if vision_preload_task is not None and not vision_preload_task.done():
            vision_preload_task.cancel()
            with suppress(asyncio.CancelledError):
                await vision_preload_task
        await _close_resource(llm_client)
        await _close_resource(llm_http_client)


app = FastAPI(title="RAG Service", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload_router)
app.include_router(chat_router)
app.include_router(notebook_content_router)


@app.get("/health")
async def healthcheck() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "rag_service",
        "embedder_model": settings.embedder_model,
    }
