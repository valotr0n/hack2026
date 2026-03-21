from __future__ import annotations

import asyncio
import inspect
import logging
import os
from contextlib import asynccontextmanager, suppress
from time import perf_counter

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


def _configure_cpu_runtime() -> None:
    import torch

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

    torch.set_num_threads(settings.torch_num_threads)
    try:
        torch.set_num_interop_threads(settings.torch_num_interop_threads)
    except RuntimeError as exc:
        logger.warning("Unable to set torch interop threads: %s", exc)

    logger.info(
        "Configured CPU runtime: cpus=%s torch_threads=%d interop_threads=%d embedding_batch_size=%d",
        os.cpu_count() or 1,
        settings.torch_num_threads,
        settings.torch_num_interop_threads,
        settings.embedding_batch_size,
    )


def _warmup_embedding_model(embedding_model: SentenceTransformer) -> None:
    started_at = perf_counter()
    embedding_model.encode(
        ["warmup"],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=1,
    )
    logger.info("Embedding model warmup finished in %.2fs", perf_counter() - started_at)


async def _warmup_open_embedder(client: AsyncOpenAI) -> int:
    started_at = perf_counter()
    try:
        response = await client.embeddings.create(
            model=settings.open_embedder_model,
            input=["warmup"],
        )
        embedding = response.data[0].embedding
        dimension = len(embedding)
        logger.info(
            "Open embedder warmup finished in %.2fs with dimension=%d model=%s",
            perf_counter() - started_at,
            dimension,
            settings.open_embedder_model,
        )
        return dimension
    except Exception as exc:
        logger.warning("Open embedder warmup failed: %s", exc)
        return 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    _configure_cpu_runtime()
    embedding_model = SentenceTransformer(settings.embedder_model)
    local_embedding_dimension = embedding_model.get_sentence_embedding_dimension()
    open_embedder_http_client = httpx.AsyncClient(
        verify=False,
        timeout=settings.open_embedding_timeout_seconds,
        limits=httpx.Limits(
            max_connections=settings.open_embedding_max_connections,
            max_keepalive_connections=settings.open_embedding_max_keepalive_connections,
        ),
        trust_env=False,
    )
    open_embedder_client = AsyncOpenAI(
        base_url=settings.open_embedder_base_url,
        api_key=settings.open_embedder_api_key,
        http_client=open_embedder_http_client,
    )
    llm_http_client = httpx.AsyncClient(verify=False, timeout=120.0)
    llm_client = AsyncOpenAI(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        http_client=llm_http_client,
    )

    app.state.embedding_model = embedding_model
    app.state.local_embedding_dimension = local_embedding_dimension
    app.state.open_embedder_client = open_embedder_client
    app.state.open_embedding_dimension = None
    app.state.llm_client = llm_client
    preload_tasks: list[asyncio.Task[None]] = []
    if settings.embedding_preload:
        preload_tasks.append(
            asyncio.create_task(
                asyncio.to_thread(_warmup_embedding_model, embedding_model),
                name="rag-service-embedding-preload",
            )
        )
        logger.info("Scheduled preload for embedding model: %s", settings.embedder_model)
        preload_tasks.append(
            asyncio.create_task(
                _warmup_open_embedder(open_embedder_client),
                name="rag-service-open-embedding-preload",
            )
        )
        logger.info("Scheduled preload for open embedder: %s", settings.open_embedder_model)

    if settings.vision_enabled and settings.vision_preload:
        from .vision import preload_vision_model

        preload_tasks.append(
            asyncio.create_task(
                preload_vision_model(
                    settings.vision_model_id,
                    settings.vision_max_new_tokens,
                    settings.vision_max_image_side,
                ),
                name="rag-service-vision-preload",
            )
        )
        logger.info("Scheduled preload for vision model: %s", settings.vision_model_id)

    app.state.preload_tasks = preload_tasks
    if preload_tasks and settings.blocking_model_preload:
        logger.info("Blocking startup preload enabled; waiting for %d task(s)", len(preload_tasks))
        results = await asyncio.gather(*preload_tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, Exception):
                raise result
            if isinstance(result, int) and result > 0:
                app.state.open_embedding_dimension = result
        logger.info("Blocking startup preload finished")
    elif preload_tasks:
        for preload_task in preload_tasks:
            def _store_open_embedding_dimension(task: asyncio.Task[object]) -> None:
                if task.cancelled():
                    return
                try:
                    result = task.result()
                except Exception:
                    return
                if isinstance(result, int) and result > 0:
                    app.state.open_embedding_dimension = result

            preload_task.add_done_callback(_store_open_embedding_dimension)

    try:
        yield
    finally:
        for preload_task in preload_tasks:
            if preload_task.done():
                continue
            preload_task.cancel()
            with suppress(asyncio.CancelledError):
                await preload_task
        await _close_resource(open_embedder_client)
        await _close_resource(open_embedder_http_client)
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
        "open_embedder_model": settings.open_embedder_model,
    }
