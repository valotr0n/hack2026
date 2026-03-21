from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from .routers import summary, mindmap, flashcards, podcast, transcribe, contract, knowledge_graph, quiz, answer, timeline, questions, compare, presentation, autotag
from .llm import contour_var

app = FastAPI(title="Content Service", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def contour_middleware(request: Request, call_next):
    contour = request.headers.get("x-contour", "open")
    token = contour_var.set(contour if contour in ("open", "closed") else "open")
    try:
        return await call_next(request)
    finally:
        contour_var.reset(token)

app.include_router(summary.router)
app.include_router(mindmap.router)
app.include_router(flashcards.router)
app.include_router(podcast.router)
app.include_router(transcribe.router)
app.include_router(contract.router)
app.include_router(knowledge_graph.router)
app.include_router(quiz.router)
app.include_router(answer.router)
app.include_router(timeline.router)
app.include_router(questions.router)
app.include_router(compare.router)
app.include_router(presentation.router)
app.include_router(autotag.router)


@app.get("/health")
async def healthcheck() -> dict[str, str]:
    return {"status": "ok", "service": "content_service"}
