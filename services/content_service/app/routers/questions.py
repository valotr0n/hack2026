from __future__ import annotations

import logging
import re
from fastapi import APIRouter
from pydantic import BaseModel
from ..config import settings
from ..json_utils import parse_json_payload
from ..llm import chat

router = APIRouter()
logger = logging.getLogger(__name__)


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


def _fallback_questions(req: QuestionsRequest) -> QuestionsResponse:
    text = req.text.lower()
    questions: list[Question] = []

    checks = [
        (
            not re.search(r"\b\d{1,2}[./]\d{1,2}[./]\d{2,4}\b|срок|дата|до |в течение|не позднее", text),
            "Какие конкретные сроки, даты и дедлайны предусмотрены документом?",
            "missing_info",
            "high",
        ),
        (
            not any(token in text for token in ("ооо", "ип", "ао", "пао", "банк", "сторона", "заемщик", "кредитор")),
            "Кто является сторонами документа и кто несёт ключевые обязательства?",
            "missing_info",
            "high",
        ),
        (
            not any(token in text for token in ("штраф", "пен", "неустой", "ответствен")),
            "Какая ответственность и какие санкции предусмотрены за нарушение условий?",
            "risk",
            "medium",
        ),
        (
            not any(token in text for token in ("сумм", "руб", "процент", "ставк", "платеж", "цена")),
            "Какие суммы, ставки, цены или платежи должны быть подтверждены дополнительно?",
            "missing_info",
            "high",
        ),
        (
            not any(token in text for token in ("приложен", "документ", "акт", "отчет", "справк", "выписк")),
            "Какие дополнительные документы или приложения нужно запросить для проверки условий?",
            "required_doc",
            "medium",
        ),
    ]

    for should_add, question, category, priority in checks:
        if should_add:
            questions.append(Question(question=question, category=category, priority=priority))

    if not questions:
        questions = [
            Question(
                question="Какие формулировки документа остаются двусмысленными и требуют письменного уточнения?",
                category="clarification",
                priority="medium",
            ),
            Question(
                question="Какие риски или обязательства сторон нужно подтвердить дополнительными материалами?",
                category="risk",
                priority="medium",
            ),
        ]

    return QuestionsResponse(
        questions=questions[:8],
        summary="Автоматический fallback выявил области, по которым в документе недостаточно явной информации для уверенного решения.",
    )


@router.post(
    "/questions",
    response_model=QuestionsResponse,
    summary="Вопросы к документу",
    description="""
Анализирует документ и генерирует список вопросов, на которые документ **не даёт ответа**.
Помогает понять, какую информацию нужно запросить дополнительно.

**Параметр `context`:**
- `general` — универсальный анализ (по умолчанию)
- `credit` — с позиции кредитного аналитика банка
- `legal` — с позиции юриста
- `financial` — с позиции финансового аналитика

**Категории вопросов:**
- `missing_info` — информация отсутствует в документе
- `clarification` — формулировка неясна или двусмысленна
- `risk` — потенциальный риск, требующий уточнения
- `required_doc` — нужен дополнительный документ

**Ответ:**
```json
{
  "questions": [
    {
      "question": "Кто является поручителем по кредиту?",
      "category": "missing_info",
      "priority": "high"
    },
    {
      "question": "Что подразумевается под 'существенным нарушением' в п. 4.2?",
      "category": "clarification",
      "priority": "medium"
    }
  ],
  "summary": "Документ не содержит информации о залоге и поручительстве..."
}
```
    """,
)
async def generate_questions(req: QuestionsRequest) -> QuestionsResponse:
    role = _CONTEXT_PROMPTS.get(req.context, _CONTEXT_PROMPTS["general"])

    system = (
        f"Ты — опытный {role}. "
        "Твоя задача — найти пробелы, неясности и риски в документе. "
        "Отвечай строго в формате JSON без лишнего текста."
    )
    user = (
        "Проанализируй документ и составь список вопросов, на которые документ НЕ даёт чёткого ответа "
        "или которые необходимо уточнить перед принятием решения.\n\n"
        "Верни JSON:\n"
        '{"questions": [{"question": "текст вопроса", '
        '"category": "missing_info|clarification|risk|required_doc", '
        '"priority": "high|medium|low"}], '
        '"summary": "краткий вывод о главных пробелах документа (2-3 предложения)"}\n\n'
        "Правила:\n"
        "- Вопросы high priority — без ответа на них нельзя принять решение\n"
        "- Вопросы medium — важны, но не блокирующие\n"
        "- Вопросы low — желательно уточнить\n"
        "- Минимум 5 вопросов, максимум 15\n"
        "- Вопросы должны быть конкретными, не абстрактными\n\n"
        f"Документ:\n{req.text}"
    )

    try:
        raw = await chat(
            system=system,
            user=user,
            temperature=0.3,
            max_tokens=settings.questions_max_tokens,
        )
    except Exception as exc:
        logger.warning("Questions generation failed before parsing: %s", exc)
        return _fallback_questions(req)

    data = parse_json_payload(raw)
    if not isinstance(data, dict):
        logger.warning("Questions parse failed, using fallback. raw_sample=%r", raw[:500])
        return _fallback_questions(req)

    try:
        return QuestionsResponse(
            questions=[Question(**q) for q in data.get("questions", [])],
            summary=data.get("summary", ""),
        )
    except Exception as exc:
        logger.warning("Questions normalization failed, using fallback: %s; data=%r", exc, data)
        return _fallback_questions(req)
