from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from ..llm import chat

logger = logging.getLogger("content_service.timeline")

router = APIRouter()

_CHUNK_SIZE = 20_000


class TimelineRequest(BaseModel):
    text: str


class TimelineEvent(BaseModel):
    date: str        # "14 марта 2026", "Q2 2025", "с момента подписания" — как в тексте
    title: str       # краткое название события
    description: str # детали
    type: str        # payment | deadline | event | agreement | violation | other


class TimelineResponse(BaseModel):
    events: list[TimelineEvent]


_EXTRACT_SYSTEM = (
    "Ты — аналитик, специализирующийся на извлечении хронологии из документов. "
    "Извлекай ТОЛЬКО те события и даты, которые явно указаны в тексте. "
    "Запрещено вычислять или предполагать даты, которых нет в тексте. "
    "Если дата относительная ('через 30 дней') — указывай дословно как в тексте. "
    "Отвечай строго в формате JSON без лишнего текста."
)

_MERGE_SYSTEM = (
    "Ты — аналитик, объединяющий хронологические данные из нескольких фрагментов одного документа. "
    "Удали точные дубликаты (одно и то же событие упомянуто в разных фрагментах). "
    "Отсортируй события хронологически. "
    "Отвечай строго в формате JSON без лишнего текста."
)

_EVENT_FORMAT = (
    '{"events": [{"date": "дата дословно из текста", '
    '"title": "краткое название события", '
    '"description": "детали — только из текста", '
    '"type": "payment|deadline|event|agreement|violation|other"}]}'
)

_EXTRACT_RULES = (
    "- Только события с явными датами или сроками из текста\n"
    "- Не придумывай события которых нет в тексте\n"
    "- Если дат нет — верни пустой массив\n"
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


def _parse_events(raw: str) -> list[dict]:
    start = raw.find("{")
    end = raw.rfind("}") + 1
    data = json.loads(raw[start:end])
    return data.get("events", [])


async def _extract_chunk_events(chunk: str, part_label: str) -> list[dict]:
    user = (
        f"{part_label}\n\n"
        f"Извлеки все события с датами из этого фрагмента и верни JSON:\n{_EVENT_FORMAT}\n\n"
        f"Правила:\n{_EXTRACT_RULES}\n"
        f"Фрагмент:\n{chunk}"
    )
    raw = await chat(system=_EXTRACT_SYSTEM, user=user, temperature=0.1)
    try:
        return _parse_events(raw)
    except Exception:
        return []


async def _hierarchical_timeline(text: str) -> list[dict]:
    if len(text) <= _CHUNK_SIZE:
        logger.info("Timeline single-pass chars=%d", len(text))
        user = (
            f"Извлеки все события с датами из текста и верни JSON:\n{_EVENT_FORMAT}\n\n"
            f"Правила:\n{_EXTRACT_RULES}"
            "- Сортируй хронологически\n\n"
            f"Текст:\n{text}"
        )
        raw = await chat(system=_EXTRACT_SYSTEM, user=user, temperature=0.1)
        return _parse_events(raw)

    chunks = _split_text(text, _CHUNK_SIZE)
    n = len(chunks)
    logger.info("Timeline map-reduce started chars=%d chunks=%d", len(text), n)

    tasks = [
        _extract_chunk_events(chunk, f"Фрагмент {i + 1} из {n}:")
        for i, chunk in enumerate(chunks)
    ]
    partial: list[list[dict]] = await asyncio.gather(*tasks)

    all_events = [e for events in partial for e in events]
    logger.info("Timeline map done total_events=%d", len(all_events))

    if not all_events:
        return []

    combined = json.dumps({"events": all_events}, ensure_ascii=False, indent=2)
    merge_user = (
        "Ниже собраны события из всех фрагментов одного документа. "
        "Удали точные дубликаты. Отсортируй хронологически. "
        f"Верни JSON в том же формате:\n{_EVENT_FORMAT}\n\n"
        f"Все события:\n{combined}"
    )
    raw = await chat(system=_MERGE_SYSTEM, user=merge_user, temperature=0.1)
    try:
        result = _parse_events(raw)
        logger.info("Timeline merge done events=%d", len(result))
        return result
    except Exception:
        logger.warning("Timeline merge parse failed, returning unmerged")
        return all_events


@router.post(
    "/timeline",
    response_model=TimelineResponse,
    summary="Временная шкала",
    description="""
Извлекает все события с датами из текста и возвращает хронологическую шкалу.
Для больших документов автоматически применяется **map-reduce**: каждый фрагмент
обрабатывается параллельно, затем результаты объединяются и дедублицируются.

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
    events_data = await _hierarchical_timeline(req.text)
    try:
        return TimelineResponse(events=[TimelineEvent(**e) for e in events_data])
    except Exception:
        raise HTTPException(status_code=500, detail="Не удалось разобрать ответ модели")
