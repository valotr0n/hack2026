from __future__ import annotations

import asyncio
import json
import logging
import math

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from ..llm import chat

logger = logging.getLogger("content_service.flashcards")

router = APIRouter()

_CHUNK_SIZE = 15_000


class FlashcardsRequest(BaseModel):
    text: str
    count: int = 10


class Flashcard(BaseModel):
    question: str
    answer: str


class FlashcardsResponse(BaseModel):
    flashcards: list[Flashcard]


_EXTRACT_SYSTEM = (
    "Ты — ассистент по созданию учебных материалов. "
    "Используй ТОЛЬКО информацию из предоставленного текста. "
    "Если ответ на вопрос нельзя найти в тексте — не создавай эту карточку. "
    "Ответы должны быть точными цитатами или перефразировками из текста, без домысливания. "
    "Отвечай строго в формате JSON, без лишнего текста."
)

_CARD_FORMAT = '{"flashcards": [{"question": "вопрос", "answer": "ответ из текста"}]}'

_EXTRACT_RULES = (
    "- Вопрос должен проверять конкретный факт из текста\n"
    "- Ответ должен явно присутствовать в тексте\n"
    "- Не придумывай вопросы на общие темы — только по содержимому\n"
    "- Запрещено создавать карточки про: ISBN, УДК, ББК, переводчика, издательство, год издания, авторские права — это не учебный материал\n"
    "- Карточки должны охватывать разные части фрагмента, не только начало\n"
)


def _split_text(text: str, chunk_size: int) -> list[str]:
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end >= len(text):
            chunks.append(text[start:].strip())
            break
        cut = text.rfind("\n", start, end)
        if cut == -1 or cut <= start:
            cut = text.rfind(" ", start, end)
        if cut == -1 or cut <= start:
            cut = end
        chunks.append(text[start:cut].strip())
        start = cut + 1
    return [c for c in chunks if c]


def _parse_flashcards(raw: str) -> list[dict]:
    start = raw.find("{")
    end = raw.rfind("}") + 1
    data = json.loads(raw[start:end])
    return data.get("flashcards", [])


async def _extract_chunk_flashcards(chunk: str, part_label: str, per_chunk: int) -> list[dict]:
    try:
        user = (
            f"{part_label}\n\n"
            f"Создай {per_chunk} карточек для самопроверки на основе этого фрагмента.\n"
            f"Правила:\n{_EXTRACT_RULES}\n"
            f"Верни JSON:\n{_CARD_FORMAT}\n\n"
            f"Фрагмент:\n{chunk}"
        )
        raw = await chat(system=_EXTRACT_SYSTEM, user=user, temperature=0.2)
        return _parse_flashcards(raw)
    except Exception as e:
        logger.warning("Chunk extraction failed (%s), skipping", type(e).__name__)
        return []


async def _hierarchical_flashcards(text: str, count: int) -> list[dict]:
    if len(text) <= _CHUNK_SIZE:
        logger.info("Flashcards single-pass chars=%d count=%d", len(text), count)
        user = (
            f"Создай {count} карточек для самопроверки на основе текста ниже.\n"
            f"Правила:\n{_EXTRACT_RULES}\n"
            f"Верни JSON:\n{_CARD_FORMAT}\n\n"
            f"Текст:\n{text}"
        )
        raw = await chat(system=_EXTRACT_SYSTEM, user=user, temperature=0.2)
        return _parse_flashcards(raw)

    chunks = _split_text(text, _CHUNK_SIZE)
    n = len(chunks)
    # Запрашиваем немного больше с каждого чанка чтобы после дедупликации осталось достаточно
    per_chunk = max(3, math.ceil(count * 1.5 / n))
    logger.info("Flashcards map-reduce started chars=%d chunks=%d count=%d per_chunk=%d", len(text), n, count, per_chunk)

    tasks = [
        _extract_chunk_flashcards(chunk, f"Фрагмент {i + 1} из {n}:", per_chunk)
        for i, chunk in enumerate(chunks)
    ]
    partial: list[list[dict]] = await asyncio.gather(*tasks)

    all_cards = [card for cards in partial for card in cards]
    logger.info("Flashcards map done total_cards=%d", len(all_cards))

    if not all_cards:
        return []

    # Программная дедупликация по вопросу — без LLM-вызова
    seen: set[str] = set()
    deduped: list[dict] = []
    for c in all_cards:
        key = c.get("question", "").strip().lower()
        if key not in seen:
            seen.add(key)
            deduped.append(c)
    logger.info("Flashcards dedup done cards=%d -> returning %d", len(deduped), min(count, len(deduped)))
    return deduped[:count]


@router.post("/flashcards", response_model=FlashcardsResponse)
async def generate_flashcards(req: FlashcardsRequest) -> FlashcardsResponse:
    cards_data = await _hierarchical_flashcards(req.text, req.count)
    try:
        return FlashcardsResponse(flashcards=[Flashcard(**c) for c in cards_data])
    except Exception:
        raise HTTPException(status_code=500, detail="Не удалось разобрать ответ модели")
