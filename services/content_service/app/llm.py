from __future__ import annotations

import contextvars

import httpx
from openai import AsyncOpenAI

from .config import settings

# Переменная контура для текущего запроса (устанавливается middleware)
contour_var: contextvars.ContextVar[str] = contextvars.ContextVar("contour", default="open")


def _build_http_client(timeout_seconds: float, max_connections: int, max_keepalive_connections: int, verify: bool = True) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        verify=verify,
        timeout=httpx.Timeout(timeout_seconds),
        limits=httpx.Limits(
            max_connections=max_connections,
            max_keepalive_connections=max_keepalive_connections,
        ),
    )


# Открытый контур — хакатонный API
_open_client = AsyncOpenAI(
    base_url=settings.llm_base_url,
    api_key=settings.llm_api_key,
    http_client=_build_http_client(
        settings.llm_open_timeout_seconds,
        settings.llm_open_max_connections,
        settings.llm_open_max_keepalive_connections,
        verify=False,
    ),
)

# Закрытый контур — локальный ollama
_closed_client = AsyncOpenAI(
    base_url=settings.ollama_base_url,
    api_key="ollama",
    http_client=_build_http_client(
        settings.llm_closed_timeout_seconds,
        settings.llm_closed_max_connections,
        settings.llm_closed_max_keepalive_connections,
    ),
)


async def chat(system: str, user: str, temperature: float = 0.7, max_tokens: int | None = None) -> str:
    contour = contour_var.get()
    if contour == "closed":
        client = _closed_client
        model = settings.ollama_model
    else:
        client = _open_client
        model = settings.llm_model

    request_kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
    }
    if max_tokens is not None:
        request_kwargs["max_tokens"] = max_tokens

    response = await client.chat.completions.create(
        **request_kwargs,
    )
    if not response.choices:
        raise RuntimeError(f"LLM returned no choices for model={model} contour={contour}")

    message = response.choices[0].message
    content = message.content

    if isinstance(content, str):
        normalized = content.strip()
        if normalized:
            return normalized
        raise RuntimeError(
            f"LLM returned empty text content for model={model} contour={contour}"
        )

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                if item.strip():
                    parts.append(item.strip())
                continue
            text = getattr(item, "text", None)
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
                continue
            if isinstance(item, dict):
                maybe_text = item.get("text")
                if isinstance(maybe_text, str) and maybe_text.strip():
                    parts.append(maybe_text.strip())
        joined = "\n".join(part for part in parts if part)
        if joined:
            return joined

    finish_reason = getattr(response.choices[0], "finish_reason", None)
    raise RuntimeError(
        f"LLM returned non-text content for model={model} contour={contour} finish_reason={finish_reason}"
    )
