from __future__ import annotations

import json
from fastapi import APIRouter
from pydantic import BaseModel
from ..llm import chat

router = APIRouter()


class QuizCheckRequest(BaseModel):
    question: str
    correct_answer: str
    user_answer: str


class QuizCheckResponse(BaseModel):
    is_correct: bool
    score: float  # 0.0 – 1.0
    feedback: str


@router.post(
    "/flashcards/check",
    response_model=QuizCheckResponse,
    summary="Проверить ответ на карточку",
    description="""
Оценивает ответ пользователя на вопрос флэш-карточки. LLM выступает в роли преподавателя:
проверяет правильность, выставляет оценку и объясняет ошибки.

**Ответ:**
```json
{
  "is_correct": true,
  "score": 0.85,
  "feedback": "Верно! Ты правильно указал основную идею, но упустил..."
}
```
`score` — от 0.0 (полностью неверно) до 1.0 (идеально).
    """,
)
async def check_flashcard(req: QuizCheckRequest) -> QuizCheckResponse:
    system = (
        "Ты — строгий, но справедливый преподаватель. "
        "Оцени ответ студента. Будь конкретен в объяснении ошибок. "
        "Отвечай строго в формате JSON без лишнего текста."
    )
    user = (
        f"Вопрос: {req.question}\n"
        f"Правильный ответ: {req.correct_answer}\n"
        f"Ответ студента: {req.user_answer}\n\n"
        "Верни JSON:\n"
        '{"is_correct": true/false, "score": 0.0-1.0, '
        '"feedback": "краткое объяснение с указанием ошибок или похвалой"}'
    )

    raw = await chat(system=system, user=user)

    try:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        data = json.loads(raw[start:end])
        return QuizCheckResponse(
            is_correct=bool(data.get("is_correct", False)),
            score=float(data.get("score", 0.0)),
            feedback=str(data.get("feedback", "")),
        )
    except Exception:
        return QuizCheckResponse(
            is_correct=False,
            score=0.0,
            feedback="Не удалось оценить ответ. Попробуйте ещё раз.",
        )
