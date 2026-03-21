from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routers import summary, mindmap, flashcards, podcast, transcribe, contract, knowledge_graph, quiz, answer

app = FastAPI(title="Content Service", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(summary.router)
app.include_router(mindmap.router)
app.include_router(flashcards.router)
app.include_router(podcast.router)
app.include_router(transcribe.router)
app.include_router(contract.router)
app.include_router(knowledge_graph.router)
app.include_router(quiz.router)
app.include_router(answer.router)


@app.get("/health")
async def healthcheck() -> dict[str, str]:
    return {"status": "ok", "service": "content_service"}
