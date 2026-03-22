from __future__ import annotations

import logging
import re
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from ..config import settings
from ..json_utils import parse_json_payload, safe_sample, strip_code_fences
from ..llm import chat

router = APIRouter()
logger = logging.getLogger(__name__)


class TimelineRequest(BaseModel):
    text: str


class TimelineEvent(BaseModel):
    date: str        # "14 марта 2026", "Q2 2025", "с момента подписания" — как в тексте
    title: str       # краткое название события
    description: str # детали
    type: str        # payment | deadline | event | agreement | violation | other


class TimelineResponse(BaseModel):
    events: list[TimelineEvent]


_MONTHS = (
    "января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря"
)
_DATE_RE = re.compile(
    rf"\b(?:\d{{1,2}}[./]\d{{1,2}}[./]\d{{2,4}}|"
    rf"\d{{1,2}}\s+(?:{_MONTHS})\s+\d{{4}}|"
    rf"Q[1-4]\s+\d{{4}}|"
    rf"(?:через|за|спустя|после|до|в течение|на следующий день|в тот же день|ровно через)[^.!?\n]{{0,48}})\b",
    flags=re.IGNORECASE,
)


def _parse_timeline_payload(raw: object | None) -> list[dict] | None:
    data = parse_json_payload(raw)
    if isinstance(data, dict):
        events = data.get("events")
        if isinstance(events, list):
            return events
    if isinstance(data, list):
        return data
    return None


def _guess_event_type(text: str) -> str:
    normalized = text.lower()
    if any(token in normalized for token in ("платеж", "оплат", "взнос", "погашен", "перечисл")):
        return "payment"
    if any(token in normalized for token in ("срок", "дедлайн", "не позднее", "до ", "в течение")):
        return "deadline"
    if any(token in normalized for token in ("договор", "соглашен", "подписан", "контракт")):
        return "agreement"
    if any(token in normalized for token in ("наруш", "просроч", "штраф", "неустой", "дефолт")):
        return "violation"
    if any(token in normalized for token in ("встрет", "обнаруж", "получ", "переех", "увид", "произош")):
        return "event"
    return "other"


def _build_title(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text).strip(" .")
    if not normalized:
        return "Событие"
    words = normalized.split()
    short = " ".join(words[:8]).strip(" ,.")
    return short[:96] or "Событие"


def _normalize_event(event: dict) -> TimelineEvent | None:
    if not isinstance(event, dict):
        return None

    date = str(event.get("date") or "").strip()
    title = str(event.get("title") or "").strip()
    description = str(event.get("description") or "").strip()
    event_type = str(event.get("type") or "").strip().lower()

    if not date:
        source_text = " ".join(part for part in (title, description) if part)
        match = _DATE_RE.search(source_text)
        date = match.group(0).strip(" .,:;") if match else ""
    if not description:
        description = title
    if not title:
        title = _build_title(description)
    if event_type not in {"payment", "deadline", "event", "agreement", "violation", "other"}:
        event_type = _guess_event_type(f"{title} {description}")

    if not date or not description:
        return None

    return TimelineEvent(
        date=date[:80],
        title=title[:120],
        description=description[:400],
        type=event_type,
    )


def _fallback_timeline_events(text: str) -> list[TimelineEvent]:
    sentences = [
        chunk.strip()
        for chunk in re.split(r"(?<=[.!?])\s+|\n+", text)
        if chunk.strip()
    ]
    events: list[TimelineEvent] = []
    seen: set[tuple[str, str]] = set()

    for sentence in sentences:
        match = _DATE_RE.search(sentence)
        if not match:
            continue
        date = match.group(0).strip(" .,:;")
        description = re.sub(r"\s+", " ", sentence).strip()
        event = TimelineEvent(
            date=date[:80],
            title=_build_title(re.sub(_DATE_RE, "", description, count=1).strip(" ,.-") or description),
            description=description[:400],
            type=_guess_event_type(description),
        )
        key = (event.date.lower(), event.description.lower())
        if key in seen:
            continue
        seen.add(key)
        events.append(event)
        if len(events) >= 20:
            break

    return events


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

    try:
        raw = await chat(
            system=system,
            user=user,
            temperature=0.1,
            max_tokens=settings.timeline_max_tokens,
        )
    except Exception as exc:
        logger.warning("Timeline generation failed before parsing: %s", exc)
        return TimelineResponse(events=_fallback_timeline_events(req.text))

    parsed_events = _parse_timeline_payload(raw)
    if parsed_events is None:
        logger.warning("Timeline parse failed, using fallback. raw_sample=%r", safe_sample(raw))
        return TimelineResponse(events=_fallback_timeline_events(req.text))

    normalized_events = [event for item in parsed_events if (event := _normalize_event(item)) is not None]
    if normalized_events:
        return TimelineResponse(events=normalized_events)

    logger.warning("Timeline normalization produced no valid events, using fallback.")
    return TimelineResponse(events=_fallback_timeline_events(req.text))
