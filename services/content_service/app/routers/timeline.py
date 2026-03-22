from __future__ import annotations

import asyncio
import json
import logging
import re

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from ..llm import chat

logger = logging.getLogger("content_service.timeline")

router = APIRouter()

_CHUNK_SIZE = 15_000


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

_EVENT_FORMAT = (
    '{"events": [{"date": "дата дословно из текста", '
    '"title": "краткое название события", '
    '"description": "детали — только из текста", '
    '"type": "payment|deadline|event|agreement|violation|other"}]}'
)

_EXTRACT_RULES = (
    "- Только реальные события из содержания документа с явными датами или сроками\n"
    "- Игнорировать: метаданные файлов (timestamps типа '21/04/14 17:54'), "
    "копирайты и авторские права (© ...), даты издания/перевода/публикации книги, "
    "технические метки документа (indd, pdf и т.п.)\n"
    "- Не придумывай события которых нет в тексте\n"
    "- Если смысловых событий нет — верни пустой массив\n"
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
    try:
        user = (
            f"{part_label}\n\n"
            f"Извлеки все события с датами из этого фрагмента и верни JSON:\n{_EVENT_FORMAT}\n\n"
            f"Правила:\n{_EXTRACT_RULES}\n"
            f"Фрагмент:\n{chunk}"
        )
        raw = await chat(system=_EXTRACT_SYSTEM, user=user, temperature=0.1)
        return _parse_events(raw)
    except Exception as e:
        logger.warning("Chunk extraction failed (%s), skipping", type(e).__name__)
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

    # Программная дедупликация по (date, title) — без LLM-вызова
    seen: set[tuple[str, str]] = set()
    deduped: list[dict] = []
    for e in all_events:
        key = (e.get("date", "").strip().lower(), e.get("title", "").strip().lower())
        if key not in seen:
            seen.add(key)
            deduped.append(e)

    # Сортировка по дате: извлекаем DD, MM, YYYY из строки даты
    def _sort_key(e: dict) -> tuple[int, int, int]:
        date_str = e.get("date", "")
        nums = re.findall(r"\d+", date_str)
        # Пробуем определить год (4 цифры), месяц, день
        year = next((int(n) for n in nums if len(n) == 4), 9999)
        # Формат DD.MM.YYYY — второй двузначный = месяц, первый = день
        two_digit = [int(n) for n in nums if len(n) <= 2 and 1 <= int(n) <= 99]
        month = two_digit[1] if len(two_digit) >= 2 else 0
        day = two_digit[0] if len(two_digit) >= 1 else 0
        return (year, month, day)

    deduped.sort(key=_sort_key)
    logger.info("Timeline dedup done events=%d", len(deduped))
    return deduped


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
