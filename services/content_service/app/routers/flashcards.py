import logging
from fastapi import APIRouter
from pydantic import BaseModel
from ..json_utils import candidate_sentences, parse_json_payload
from ..llm import chat

router = APIRouter()
logger = logging.getLogger(__name__)


class FlashcardsRequest(BaseModel):
    text: str
    count: int = 10


class Flashcard(BaseModel):
    question: str
    answer: str


class FlashcardsResponse(BaseModel):
    flashcards: list[Flashcard]


def _fallback_flashcards(text: str, count: int) -> FlashcardsResponse:
    sentences = candidate_sentences(text, min_length=24, max_items=max(3, count))
    flashcards = [
        Flashcard(
            question=f"Что говорится в материале о фрагменте {index + 1}?",
            answer=sentence[:240],
        )
        for index, sentence in enumerate(sentences[:count])
    ]
    if not flashcards:
        flashcards = [Flashcard(question="О чём этот документ?", answer="Недостаточно данных для генерации карточек.")]
    return FlashcardsResponse(flashcards=flashcards)


@router.post("/flashcards", response_model=FlashcardsResponse)
async def generate_flashcards(req: FlashcardsRequest) -> FlashcardsResponse:
    system = (
        "Ты — ассистент по созданию учебных материалов. "
        "Используй ТОЛЬКО информацию из предоставленного текста. "
        "Если ответ на вопрос нельзя найти в тексте — не создавай эту карточку. "
        "Ответы должны быть точными цитатами или перефразировками из текста, без домысливания. "
        "Отвечай строго в формате JSON, без лишнего текста."
    )

    user = (
        f"Создай {req.count} карточек для самопроверки на основе текста ниже.\n"
        "Правила:\n"
        "- Вопрос должен проверять конкретный факт из текста\n"
        "- Ответ должен явно присутствовать в тексте\n"
        "- Не придумывай вопросы на общие темы — только по содержимому\n\n"
        "Верни JSON в следующем формате:\n"
        '{"flashcards": [{"question": "вопрос", "answer": "ответ из текста"}]}\n\n'
        f"Текст:\n{req.text}"
    )

    try:
        raw = await chat(system=system, user=user, temperature=0.2)
    except Exception as exc:
        logger.warning("Flashcards generation failed before parsing: %s", exc)
        return _fallback_flashcards(req.text, req.count)

    data = parse_json_payload(raw)
    if not isinstance(data, dict):
        logger.warning("Flashcards parse failed, using fallback. raw_sample=%r", raw[:500])
        return _fallback_flashcards(req.text, req.count)

    try:
        return FlashcardsResponse(flashcards=data["flashcards"])
    except Exception as exc:
        logger.warning("Flashcards normalization failed, using fallback: %s; data=%r", exc, data)
        return _fallback_flashcards(req.text, req.count)
