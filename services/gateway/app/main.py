from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .auth import require_auth
from .config import settings
from .database import init_db
from .routers.auth import router as auth_router
from .routers.notebooks import router as notebooks_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("gateway")

HOP_BY_HOP_HEADERS = {
    "connection",
    "content-encoding",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
}

UPSTREAMS = {
    "rag": settings.rag_service_url.rstrip("/"),
    "content": settings.content_service_url.rstrip("/"),
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db(settings.db_path)
    app.state.http_client = httpx.AsyncClient(timeout=settings.request_timeout_seconds)
    try:
        yield
    finally:
        await app.state.http_client.aclose()


app = FastAPI(
    title="AI Platform Gateway",
    version="1.0.0",
    description="""
## AI-платформа переработки знаний

Аналог **NotebookLM** — загружайте документы, задавайте вопросы и генерируйте контент.

### Как начать работу

1. **Зарегистрируйтесь** — `POST /auth/register`
2. **Войдите** — `POST /auth/login` → получите `access_token`
3. **Передавайте токен** в заголовке каждого запроса:
   ```
   Authorization: Bearer <access_token>
   ```
4. **Создайте блокнот** — `POST /notebooks`
5. **Загрузите документы** — `POST /notebooks/{id}/sources` (PDF, DOCX, TXT)
6. **Используйте функции** — чат, саммари, mindmap, карточки, подкаст

### Базовый URL
```
http://89.169.174.78:8000
```
    """,
    lifespan=lifespan,
)

cors_allow_origins = settings.cors_origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if cors_allow_origins == ["*"] else cors_allow_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(notebooks_router)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    started_at = time.perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        duration_ms = (time.perf_counter() - started_at) * 1000
        logger.info(
            "%s %s %s %.2fms",
            request.method,
            request.url.path,
            status_code,
            duration_ms,
        )


@app.get("/voices", tags=["podcast"], summary="Список доступных голосов TTS")
async def list_voices(request: Request, user_id: str = Depends(require_auth)) -> list[dict]:
    client: httpx.AsyncClient = request.app.state.http_client
    resp = await client.get(f"{UPSTREAMS['content']}/voices")
    resp.raise_for_status()
    return resp.json()


@app.get("/voices/{voice_id}/sample", tags=["podcast"], summary="Аудиосэмпл голоса")
async def voice_sample(voice_id: str, request: Request, user_id: str = Depends(require_auth)):
    client: httpx.AsyncClient = request.app.state.http_client
    resp = await client.get(f"{UPSTREAMS['content']}/voices/{voice_id}/sample")
    resp.raise_for_status()
    return StreamingResponse(
        resp.aiter_bytes(),
        media_type="audio/mpeg",
        headers={"Cache-Control": "public, max-age=86400"},
    )


@app.get("/")
async def root() -> dict[str, str]:
    return {
        "service": "gateway",
        "status": "ok",
        "docs": "/docs",
    }


# ── Low-level proxy (для прямого доступа к сервисам при отладке) ──────────────

def _filter_request_headers(request: Request) -> dict[str, str]:
    return {
        k: v for k, v in request.headers.items()
        if k.lower() not in HOP_BY_HOP_HEADERS
    }


def _filter_response_headers(headers: httpx.Headers) -> dict[str, str]:
    return {
        k: v for k, v in headers.items()
        if k.lower() not in HOP_BY_HOP_HEADERS
    }


async def _proxy_request(request: Request, upstream_base_url: str, remainder_path: str) -> Response:
    client: httpx.AsyncClient = request.app.state.http_client
    upstream_path = remainder_path.lstrip("/")
    url = f"{upstream_base_url}/{upstream_path}" if upstream_path else upstream_base_url
    if request.url.query:
        url = f"{url}?{request.url.query}"

    try:
        upstream_request = client.build_request(
            request.method,
            url,
            content=await request.body(),
            headers=_filter_request_headers(request),
        )
        upstream_response = await client.send(upstream_request, stream=True)
    except httpx.RequestError as exc:
        logger.warning("upstream request failed url=%s error=%s", url, exc)
        return JSONResponse(
            status_code=502,
            content={"detail": "Upstream service unavailable", "upstream": upstream_base_url},
        )

    content_type = upstream_response.headers.get("content-type", "")
    response_headers = _filter_response_headers(upstream_response.headers)

    if "text/event-stream" in content_type:
        async def event_stream():
            try:
                async for chunk in upstream_response.aiter_bytes():
                    yield chunk
            finally:
                await upstream_response.aclose()

        return StreamingResponse(
            event_stream(),
            status_code=upstream_response.status_code,
            headers=response_headers,
            media_type="text/event-stream",
        )

    content = await upstream_response.aread()
    await upstream_response.aclose()
    return Response(
        content=content,
        status_code=upstream_response.status_code,
        headers=response_headers,
        media_type=upstream_response.headers.get("content-type"),
    )


@app.api_route("/api/rag", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"])
@app.api_route("/api/rag/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"])
async def proxy_rag(request: Request, path: str = "") -> Response:
    return await _proxy_request(request, UPSTREAMS["rag"], path)


@app.api_route("/api/content", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"])
@app.api_route("/api/content/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"])
async def proxy_content(request: Request, path: str = "") -> Response:
    return await _proxy_request(request, UPSTREAMS["content"], path)


async def _fetch_health(client: httpx.AsyncClient, service_name: str, base_url: str) -> dict[str, Any]:
    url = f"{base_url}/health"
    try:
        response = await client.get(url)
        try:
            payload: Any = response.json()
        except ValueError:
            payload = response.text
        return {"service": service_name, "ok": response.status_code == 200, "status_code": response.status_code, "data": payload}
    except httpx.RequestError as exc:
        return {"service": service_name, "ok": False, "status_code": None, "error": str(exc)}


@app.get("/health")
async def health() -> JSONResponse:
    client: httpx.AsyncClient = app.state.http_client
    rag_health, content_health = await asyncio.gather(
        _fetch_health(client, "rag_service", UPSTREAMS["rag"]),
        _fetch_health(client, "content_service", UPSTREAMS["content"]),
    )
    checks = [rag_health, content_health]
    healthy = all(item["ok"] for item in checks)
    return JSONResponse(
        status_code=200 if healthy else 503,
        content={"status": "ok" if healthy else "degraded", "services": checks},
    )
