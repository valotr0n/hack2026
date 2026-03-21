from __future__ import annotations

import json
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from ..llm import chat

router = APIRouter()


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

    raw = await chat(system=system, user=user)

    try:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        data = json.loads(raw[start:end])
        return QuestionsResponse(
            questions=[Question(**q) for q in data.get("questions", [])],
            summary=data.get("summary", ""),
        )
    except Exception:
        raise HTTPException(status_code=500, detail="Не удалось разобрать ответ модели")
