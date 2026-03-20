from fastapi import FastAPI
from .routers import summary, mindmap, flashcards, podcast

app = FastAPI(title="Content Service", version="0.1.0")

app.include_router(summary.router)
app.include_router(mindmap.router)
app.include_router(flashcards.router)
app.include_router(podcast.router)


@app.get("/health")
async def healthcheck() -> dict[str, str]:
    return {"status": "ok", "service": "content_service"}
