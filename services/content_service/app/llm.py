from __future__ import annotations

import asyncio
import contextvars
import logging
import time

import httpx
from openai import AsyncOpenAI

from .config import settings

logger = logging.getLogger("content_service.llm")

# Последовательные LLM-вызовы — API обрабатывает запросы последовательно, параллельность бессмысленна
_llm_semaphore = asyncio.Semaphore(1)

# Переменная контура для текущего запроса (устанавливается middleware)
contour_var: contextvars.ContextVar[str] = contextvars.ContextVar("contour", default="open")

# Открытый контур — хакатонный API
_open_client = AsyncOpenAI(
    base_url=settings.llm_base_url,
    api_key=settings.llm_api_key,
    max_retries=0,  # Без ретраев: 3 попытки × 120s = 360s на один чанк
    http_client=httpx.AsyncClient(verify=False, timeout=300.0),
)

# Закрытый контур — локальный ollama
_closed_client = AsyncOpenAI(
    base_url=settings.ollama_base_url,
    api_key="ollama",
    max_retries=0,
    http_client=httpx.AsyncClient(timeout=300.0),
)


async def chat(system: str, user: str, temperature: float = 0.7) -> str:
    contour = contour_var.get()
    if contour == "closed":
        client = _closed_client
        model = settings.ollama_model
    else:
        client = _open_client
        model = settings.llm_model

    input_chars = len(system) + len(user)
    started_at = time.perf_counter()

    async with _llm_semaphore:
        logger.info("LLM call started model=%s contour=%s temperature=%.1f input_chars=%d", model, contour, temperature, input_chars)
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
        )

    elapsed = time.perf_counter() - started_at
    result = response.choices[0].message.content
    logger.info("LLM call done model=%s contour=%s output_chars=%d %.2fs", model, contour, len(result), elapsed)
    return result
