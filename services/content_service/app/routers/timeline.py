from __future__ import annotations

import json
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from ..config import settings
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
        "Извлекай ТОЛЬКО те события и даты, которые явно указаны в тексте. "
        "Запрещено вычислять или предполагать даты, которых нет в тексте. "
        "Если дата относительная ('через 30 дней') — указывай дословно как в тексте. "
        "Отвечай строго в формате JSON без лишнего текста."
    )
    user = (
        "Извлеки все события с датами из текста и верни JSON:\n"
        '{"events": [{"date": "дата дословно из текста", '
        '"title": "краткое название события", '
        '"description": "детали — только из текста", '
        '"type": "payment|deadline|event|agreement|violation|other"}]}\n'
        "Правила:\n"
        "- Включай только события с явными датами или сроками из текста\n"
        "- Не придумывай события которых нет в тексте\n"
        "- Сортируй хронологически\n"
        "- Если дат нет вообще — верни пустой массив events\n\n"
        f"Текст:\n{req.text}"
    )

    raw = await chat(
        system=system,
        user=user,
        temperature=0.1,
        max_tokens=settings.timeline_max_tokens,
    )

    try:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        data = json.loads(raw[start:end])
        return TimelineResponse(
            events=[TimelineEvent(**e) for e in data.get("events", [])]
        )
    except Exception:
        raise HTTPException(status_code=500, detail="Не удалось разобрать ответ модели")
