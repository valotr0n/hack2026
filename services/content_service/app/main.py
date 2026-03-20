from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI(title="Content Service", version="0.1.0")


@app.get("/health")
async def healthcheck() -> dict[str, str]:
    return {"status": "ok", "service": "content_service"}


@app.api_route(
    "/{path:path}",
    methods=["DELETE", "GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT"],
)
async def stub_endpoint(path: str, request: Request) -> JSONResponse:
    normalized_path = f"/{path}" if path else "/"
    return JSONResponse(
        status_code=501,
        content={
            "service": "content_service",
            "status": "stub",
            "message": "Content service scaffold is running. Replace this handler with real endpoints.",
            "method": request.method,
            "path": normalized_path,
        },
    )
