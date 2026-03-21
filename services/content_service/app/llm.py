from __future__ import annotations

import contextvars

import httpx
from openai import AsyncOpenAI

from .config import settings

# Переменная контура для текущего запроса (устанавливается middleware)
contour_var: contextvars.ContextVar[str] = contextvars.ContextVar("contour", default="open")

# Открытый контур — хакатонный API
_open_client = AsyncOpenAI(
    base_url=settings.llm_base_url,
    api_key=settings.llm_api_key,
    http_client=httpx.AsyncClient(verify=False, timeout=120.0),
)

# Закрытый контур — локальный ollama
_closed_client = AsyncOpenAI(
    base_url=settings.ollama_base_url,
    api_key="ollama",
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

    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
    )
    return response.choices[0].message.content
