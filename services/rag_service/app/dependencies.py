from fastapi import Request
from openai import AsyncOpenAI
from sentence_transformers import SentenceTransformer


def get_embedding_model(request: Request) -> SentenceTransformer:
    return request.app.state.embedding_model


def get_llm_client(request: Request) -> AsyncOpenAI:
    contour = request.headers.get("x-contour", "open")
    if contour == "closed":
        return request.app.state.closed_llm_client
    return request.app.state.open_llm_client