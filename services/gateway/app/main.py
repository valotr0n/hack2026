from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

from .auth import create_access_token, require_auth, verify_credentials
from .config import settings

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
    app.state.http_client = httpx.AsyncClient(timeout=settings.request_timeout_seconds)
    try:
        yield
    finally:
        await app.state.http_client.aclose()


app = FastAPI(title="AI Platform Gateway", lifespan=lifespan)

cors_allow_origins = settings.cors_origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if cors_allow_origins == ["*"] else cors_allow_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


@app.get("/")
async def root() -> dict[str, str]:
    return {
        "service": "gateway",
        "status": "ok",
        "routes": "/api/rag/*, /api/content/*, /health, /auth/login",
    }


@app.post("/auth/login", response_model=TokenResponse)
async def login(req: LoginRequest) -> TokenResponse:
    if not verify_credentials(req.username, req.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Неверный логин или пароль",
        )
    return TokenResponse(access_token=create_access_token(req.username))


@app.get("/auth/me")
async def me(username: str = Depends(require_auth)) -> dict[str, str]:
    return {"username": username}


def _filter_request_headers(request: Request) -> dict[str, str]:
    headers = {}
    for key, value in request.headers.items():
        if key.lower() in HOP_BY_HOP_HEADERS:
            continue
        headers[key] = value
    return headers


def _filter_response_headers(headers: httpx.Headers) -> dict[str, str]:
    filtered: dict[str, str] = {}
    for key, value in headers.items():
        if key.lower() in HOP_BY_HOP_HEADERS:
            continue
        filtered[key] = value
    return filtered


async def _proxy_request(
    request: Request,
    upstream_base_url: str,
    remainder_path: str,
) -> Response:
    client: httpx.AsyncClient = request.app.state.http_client
    upstream_path = remainder_path.lstrip("/")
    url = f"{upstream_base_url}/{upstream_path}" if upstream_path else upstream_base_url
    query = request.url.query
    if query:
        url = f"{url}?{query}"

    body = await request.body()
    headers = _filter_request_headers(request)

    try:
        upstream_request = client.build_request(
            request.method, url, content=body, headers=headers
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
async def proxy_rag(
    request: Request,
    path: str = "",
    _: str = Depends(require_auth),
) -> Response:
    return await _proxy_request(request, UPSTREAMS["rag"], path)


@app.api_route(
    "/api/content",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
)
@app.api_route(
    "/api/content/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
)
async def proxy_content(
    request: Request,
    path: str = "",
    _: str = Depends(require_auth),
) -> Response:
    return await _proxy_request(request, UPSTREAMS["content"], path)


async def _fetch_health(client: httpx.AsyncClient, service_name: str, base_url: str) -> dict[str, Any]:
    url = f"{base_url}/health"
    try:
        response = await client.get(url)
        payload: Any
        try:
            payload = response.json()
        except ValueError:
            payload = response.text
        return {
            "service": service_name,
            "ok": response.status_code == 200,
            "status_code": response.status_code,
            "data": payload,
        }
    except httpx.RequestError as exc:
        return {
            "service": service_name,
            "ok": False,
            "status_code": None,
            "error": str(exc),
        }


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
        content={
            "status": "ok" if healthy else "degraded",
            "services": checks,
        },
    )
