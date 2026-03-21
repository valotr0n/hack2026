import json
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from ..llm import chat

router = APIRouter()


class FlashcardsRequest(BaseModel):
    text: str
    count: int = 10


class Flashcard(BaseModel):
    question: str
    answer: str


class FlashcardsResponse(BaseModel):
    flashcards: list[Flashcard]


@router.post("/flashcards", response_model=FlashcardsResponse)
async def generate_flashcards(req: FlashcardsRequest) -> FlashcardsResponse:
    system = (
        "Ты — ассистент по созданию учебных материалов. "
        "Используй ТОЛЬКО информацию из предоставленного текста. "
        "Отвечай строго в формате JSON, без лишнего текста."
    )

    user = (
        f"Создай {req.count} карточек для самопроверки на основе текста ниже.\n"
        "Верни JSON в следующем формате:\n"
        '{"flashcards": [{"question": "вопрос", "answer": "ответ"}]}\n\n'
        f"Текст:\n{req.text}"
    )

    raw = await chat(system=system, user=user)

    try:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        data = json.loads(raw[start:end])
        return FlashcardsResponse(flashcards=data["flashcards"])
    except Exception:
        raise HTTPException(status_code=500, detail="Не удалось разобрать ответ модели")
