from __future__ import annotations

import json
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from ..llm import chat

router = APIRouter()


class TimelineRequest(BaseModel):
    text: str


class TimelineEvent(BaseModel):
    date: str        # "14 марта 2026", "Q2 2025", "с момента подписания" — как в тексте
    title: str       # краткое название события
    description: str # детали
    type: str        # payment | deadline | event | agreement | violation | other


class TimelineResponse(BaseModel):
    events: list[TimelineEvent]


@router.post(
    "/timeline",
    response_model=TimelineResponse,
    summary="Временная шкала",
    description="""
Извлекает все события с датами из текста и возвращает хронологическую шкалу.

**Типы событий:** `payment`, `deadline`, `event`, `agreement`, `violation`, `other`

**Ответ:**
```json
{
  "events": [
    {
      "date": "01.03.2026",
      "title": "Подписание договора",
      "description": "Кредитный договор №123 на сумму 5 млн руб.",
      "type": "agreement"
    },
    {
      "date": "01.04.2026",
      "title": "Первый платёж",
      "description": "Ежемесячный платёж 150 000 руб.",
      "type": "payment"
    }
  ]
}
```
    """,
)
async def generate_timeline(req: TimelineRequest) -> TimelineResponse:
    system = (
        "Ты — аналитик, специализирующийся на извлечении хронологии из документов. "
        "Находи ВСЕ события с датами, сроками или временными привязками. "
        "Отвечай строго в формате JSON без лишнего текста."
    )
    user = (
        "Извлеки все события с датами из текста и верни JSON:\n"
        '{"events": [{"date": "дата как в тексте или вычисленная", '
        '"title": "краткое название", '
        '"description": "детали события", '
        '"type": "payment|deadline|event|agreement|violation|other"}]}\n'
        "Правила:\n"
        "- Сортируй события хронологически\n"
        "- Если дата относительная ('через 30 дней') — укажи её как есть\n"
        "- Включай платежи, дедлайны, штрафные даты, даты вступления в силу\n"
        "- Минимум 3 события если они есть в тексте\n\n"
        f"Текст:\n{req.text}"
    )

    raw = await chat(system=system, user=user)

    try:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        data = json.loads(raw[start:end])
        return TimelineResponse(
            events=[TimelineEvent(**e) for e in data.get("events", [])]
        )
    except Exception:
        raise HTTPException(status_code=500, detail="Не удалось разобрать ответ модели")
