from fastapi import Request
from openai import AsyncOpenAI
from sentence_transformers import SentenceTransformer


def get_embedding_model(request: Request) -> SentenceTransformer:
    return request.app.state.embedding_model


def get_llm_client(request: Request) -> AsyncOpenAI:
    return request.app.state.llm_client