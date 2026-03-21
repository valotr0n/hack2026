from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel
from ..llm import chat

router = APIRouter()


class AnswerRequest(BaseModel):
    text: str
    question: str


class AnswerResponse(BaseModel):
    answer: str


@router.post(
    "/answer",
    response_model=AnswerResponse,
    summary="Ответить на вопрос по тексту",
    description="Отвечает на произвольный вопрос строго на основе переданного текста. Используется для мульти-блокнотного поиска.",
)
async def answer_question(req: AnswerRequest) -> AnswerResponse:
    system = (
        "Ты — точный аналитик. Отвечай только на основе предоставленного текста. "
        "Если информации недостаточно — прямо укажи это. "
        "Цитируй конкретные места из текста для подтверждения ответа."
    )
    user = f"Текст:\n{req.text}\n\nВопрос: {req.question}"
    answer = await chat(system=system, user=user)
    return AnswerResponse(answer=answer)
