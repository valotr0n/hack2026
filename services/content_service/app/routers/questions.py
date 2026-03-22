from __future__ import annotations

import asyncio
import json
import logging
import math

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from ..llm import chat

logger = logging.getLogger("content_service.questions")

router = APIRouter()

_CHUNK_SIZE = 20_000


class QuestionsRequest(BaseModel):
    text: str
    context: str = "general"  # general | credit | legal | financial


class Question(BaseModel):
    question: str
    category: str   # missing_info | clarification | risk | required_doc
    priority: str   # high | medium | low


class QuestionsResponse(BaseModel):
    questions: list[Question]
    summary: str  # краткий вывод о пробелах в документе


_CONTEXT_PROMPTS = {
    "credit": "кредитного аналитика банка, проверяющего заявку на кредит",
    "legal": "юриста, проверяющего договор перед подписанием",
    "financial": "финансового аналитика, изучающего отчётность компании",
    "general": "эксперта-аналитика, изучающего документ",
}

_MERGE_SYSTEM = (
    "Ты — старший аналитик, объединяющий вопросы из нескольких фрагментов документа. "
    "Удали дублирующиеся и похожие вопросы (оставь лучшую формулировку). "
    "Расставь приоритеты по всему документу. "
    "Напиши итоговый summary о главных пробелах документа. "
    "Отвечай строго в формате JSON без лишнего текста."
)

_QUESTIONS_FORMAT = (
    '{"questions": [{"question": "текст вопроса", '
    '"category": "missing_info|clarification|risk|required_doc", '
    '"priority": "high|medium|low"}], '
    '"summary": "краткий вывод о главных пробелах документа (2-3 предложения)"}'
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


def _parse_questions(raw: str) -> dict:
    start = raw.find("{")
    end = raw.rfind("}") + 1
    return json.loads(raw[start:end])


async def _extract_chunk_questions(chunk: str, part_label: str, role: str, per_chunk: int) -> dict:
    system = (
        f"Ты — опытный {role}. "
        "Твоя задача — найти пробелы, неясности и риски в этом фрагменте документа. "
        "Отвечай строго в формате JSON без лишнего текста."
    )
    user = (
        f"{part_label}\n\n"
        "Проанализируй этот фрагмент и составь список вопросов, на которые фрагмент "
        "НЕ даёт чёткого ответа или которые необходимо уточнить.\n\n"
        f"Верни JSON:\n{_QUESTIONS_FORMAT}\n\n"
        "Правила:\n"
        "- Вопросы high priority — без ответа нельзя принять решение\n"
        "- Вопросы medium — важны, но не блокирующие\n"
        "- Вопросы low — желательно уточнить\n"
        f"- Максимум {per_chunk} вопросов по этому фрагменту\n"
        "- Вопросы должны быть конкретными, не абстрактными\n\n"
        f"Фрагмент:\n{chunk}"
    )
    raw = await chat(system=system, user=user, temperature=0.3)
    try:
        return _parse_questions(raw)
    except Exception:
        return {"questions": [], "summary": ""}


async def _hierarchical_questions(text: str, context: str) -> dict:
    role = _CONTEXT_PROMPTS.get(context, _CONTEXT_PROMPTS["general"])

    if len(text) <= _CHUNK_SIZE:
        logger.info("Questions single-pass chars=%d context=%s", len(text), context)
        system = (
            f"Ты — опытный {role}. "
            "Твоя задача — найти пробелы, неясности и риски в документе. "
            "Отвечай строго в формате JSON без лишнего текста."
        )
        user = (
            "Проанализируй документ и составь список вопросов, на которые документ "
            "НЕ даёт чёткого ответа или которые необходимо уточнить перед принятием решения.\n\n"
            f"Верни JSON:\n{_QUESTIONS_FORMAT}\n\n"
            "Правила:\n"
            "- Вопросы high priority — без ответа нельзя принять решение\n"
            "- Вопросы medium — важны, но не блокирующие\n"
            "- Вопросы low — желательно уточнить\n"
            "- Минимум 5 вопросов, максимум 15\n"
            "- Вопросы должны быть конкретными, не абстрактными\n\n"
            f"Документ:\n{text}"
        )
        raw = await chat(system=system, user=user, temperature=0.3)
        return _parse_questions(raw)

    chunks = _split_text(text, _CHUNK_SIZE)
    n = len(chunks)
    per_chunk = max(5, math.ceil(15 / n))
    logger.info("Questions map-reduce started chars=%d chunks=%d context=%s", len(text), n, context)

    tasks = [
        _extract_chunk_questions(chunk, f"Фрагмент {i + 1} из {n}:", role, per_chunk)
        for i, chunk in enumerate(chunks)
    ]
    partial: list[dict] = await asyncio.gather(*tasks)

    all_questions = [q for p in partial for q in p.get("questions", [])]
    logger.info("Questions map done total_questions=%d", len(all_questions))

    if not all_questions:
        return {"questions": [], "summary": "Документ не содержит информации для анализа."}

    combined = json.dumps({"questions": all_questions}, ensure_ascii=False, indent=2)
    merge_user = (
        f"Ты анализируешь документ с позиции {role}.\n"
        "Ниже собраны вопросы из всех фрагментов одного документа. "
        "Удали дублирующиеся и похожие вопросы (оставь лучшую формулировку). "
        "Итоговый список: минимум 5, максимум 15 вопросов. "
        "Напиши итоговый summary о главных пробелах всего документа. "
        f"Верни JSON в том же формате:\n{_QUESTIONS_FORMAT}\n\n"
        f"Все вопросы:\n{combined}"
    )
    raw = await chat(system=_MERGE_SYSTEM, user=merge_user, temperature=0.3)
    try:
        result = _parse_questions(raw)
        logger.info("Questions merge done questions=%d", len(result.get("questions", [])))
        return result
    except Exception:
        logger.warning("Questions merge parse failed, returning unmerged")
        partial_summaries = " ".join(p.get("summary", "") for p in partial if p.get("summary"))
        return {"questions": all_questions[:15], "summary": partial_summaries}


@router.post(
    "/questions",
    response_model=QuestionsResponse,
    summary="Вопросы к документу",
    description="""
Анализирует документы блокнота и генерирует список вопросов, на которые они **не дают ответа**.
Помогает аналитику понять, какую информацию нужно запросить дополнительно.
Для больших документов автоматически применяется **map-reduce**.

**Параметр `context`** — роль, с позиции которой анализируется документ:
- `general` — универсальный анализ (по умолчанию)
- `credit` — кредитный аналитик банка
- `legal` — юрист
- `financial` — финансовый аналитик

**Категории вопросов:**
- `missing_info` — информация отсутствует
- `clarification` — формулировка неясна
- `risk` — требует уточнения из-за риска
- `required_doc` — нужен дополнительный документ

**Приоритеты:** `high` (блокирующие) · `medium` · `low`

**Ответ:**
```json
{
  "questions": [
    {"question": "Кто является поручителем?", "category": "missing_info", "priority": "high"},
    {"question": "Что значит 'существенное нарушение' в п. 4.2?", "category": "clarification", "priority": "medium"}
  ],
  "summary": "Документ не содержит информации о залоге и поручительстве..."
}
```
    """,
)
async def generate_questions(req: QuestionsRequest) -> QuestionsResponse:
    data = await _hierarchical_questions(req.text, req.context)
    try:
        return QuestionsResponse(
            questions=[Question(**q) for q in data.get("questions", [])],
            summary=data.get("summary", ""),
        )
    except Exception:
        raise HTTPException(status_code=500, detail="Не удалось разобрать ответ модели")
