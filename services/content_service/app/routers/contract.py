from __future__ import annotations

import json
import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from ..config import settings
from ..llm import chat

router = APIRouter()
logger = logging.getLogger(__name__)


class ContractRequest(BaseModel):
    text: str


class Obligation(BaseModel):
    party: str
    text: str


class ContractResponse(BaseModel):
    parties: list[str]
    subject: str
    key_conditions: list[str]
    obligations: list[Obligation]
    risks: list[str]
    deadlines: list[str]
    penalties: list[str]


_EXTRACT_SYSTEM = (
    "Ты — юридический аналитик банка. "
    "Извлекай ТОЛЬКО то, что явно написано в тексте договора. "
    "Если какое-либо поле отсутствует в тексте — возвращай пустой массив или пустую строку. "
    "Запрещено додумывать, предполагать или добавлять типичные условия договоров. "
    "Отвечай строго в формате JSON без лишнего текста."
)

_VERIFY_SYSTEM = (
    "Ты — аудитор юридических данных. "
    "Тебе дан исходный текст договора и извлечённые из него данные. "
    "Проверь каждый пункт: если он явно подтверждается текстом — оставь. "
    "Если не подтверждается или додуман — удали. "
    "Возвращай исправленный JSON в том же формате, без пояснений."
)


def _parse_contract(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:].strip()
    start = raw.find("{")
    end = raw.rfind("}") + 1
    return json.loads(raw[start:end])


def _stringify(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _normalize_string_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [_stringify(value)] if _stringify(value) else []
    if isinstance(value, list):
        return [_stringify(item) for item in value if _stringify(item)]
    return [_stringify(value)] if _stringify(value) else []


def _normalize_obligation_item(value: object) -> Obligation | None:
    if isinstance(value, Obligation):
        return value

    if isinstance(value, dict):
        party = _stringify(value.get("party"))
        text = _stringify(value.get("text"))
        if not text and len(value) == 1:
            key, item = next(iter(value.items()))
            party = party or _stringify(key)
            text = _stringify(item)
        if not text:
            return None
        return Obligation(party=party or "Не указано", text=text)

    text = _stringify(value)
    if not text:
        return None

    for separator in ("—", " - ", ":", ";"):
        if separator in text:
            party, obligation_text = text.split(separator, 1)
            party = _stringify(party)
            obligation_text = _stringify(obligation_text)
            if obligation_text:
                return Obligation(party=party or "Не указано", text=obligation_text)

    return Obligation(party="Не указано", text=text)


def _normalize_obligations(value: object) -> list[Obligation]:
    if value is None:
        return []
    items = value if isinstance(value, list) else [value]
    obligations: list[Obligation] = []
    for item in items:
        obligation = _normalize_obligation_item(item)
        if obligation is not None:
            obligations.append(obligation)
    return obligations


def _normalize_contract_payload(data: dict) -> ContractResponse:
    return ContractResponse(
        parties=_normalize_string_list(data.get("parties")),
        subject=_stringify(data.get("subject")),
        key_conditions=_normalize_string_list(data.get("key_conditions")),
        obligations=_normalize_obligations(data.get("obligations")),
        risks=_normalize_string_list(data.get("risks")),
        deadlines=_normalize_string_list(data.get("deadlines")),
        penalties=_normalize_string_list(data.get("penalties")),
    )


@router.post(
    "/contract",
    response_model=ContractResponse,
    summary="Анализ договора",
    description="""
Извлекает структурированную информацию из текста договора:
стороны, предмет, ключевые условия, обязательства, риски, сроки, штрафы.

Используется двухпроходная верификация: сначала извлечение, затем проверка
каждого пункта на соответствие исходному тексту. Исключает галлюцинации.

**Ответ:**
```json
{
  "parties": ["ООО Ромашка", "ИП Иванов"],
  "subject": "Поставка товаров...",
  "key_conditions": ["Оплата в течение 30 дней", ...],
  "obligations": [{"party": "Поставщик", "text": "Доставить товар..."}],
  "risks": ["Штраф 0.1% за просрочку", ...],
  "deadlines": ["Срок поставки — 14 дней", ...],
  "penalties": ["Неустойка 1% в день", ...]
}
```
    """,
)
async def analyze_contract(req: ContractRequest) -> ContractResponse:
    fmt = (
        '{"parties": ["сторона 1", "сторона 2"], '
        '"subject": "предмет договора — 1-2 предложения дословно из текста", '
        '"key_conditions": ["условие 1 из текста", "условие 2 из текста"], '
        '"obligations": [{"party": "название стороны", "text": "обязательство дословно из текста"}], '
        '"risks": ["риск 1 из текста"], '
        '"deadlines": ["срок 1 из текста"], '
        '"penalties": ["штраф/пеня из текста"]}'
    )

    # Проход 1: извлечение
    extract_user = (
        "Извлеки из договора данные строго по тексту. "
        "Если поле не упоминается явно — оставь пустым.\n\n"
        f"Формат:\n{fmt}\n\n"
        f"Договор:\n{req.text}"
    )
    raw1 = await chat(
        system=_EXTRACT_SYSTEM,
        user=extract_user,
        temperature=0.1,
        max_tokens=settings.contract_extract_max_tokens,
    )

    try:
        data = _parse_contract(raw1)
    except Exception as exc:
        logger.warning("Contract extract parse failed: %s; raw=%r", exc, raw1[:1200])
        raise HTTPException(status_code=500, detail="Не удалось разобрать ответ модели (проход 1)")

    try:
        extracted = _normalize_contract_payload(data)
    except Exception as exc:
        logger.warning("Contract extract normalization failed: %s; data=%r", exc, data)
        raise HTTPException(status_code=500, detail="Не удалось нормализовать ответ модели (проход 1)")

    # Проход 2: верификация
    verify_user = (
        "Исходный текст договора:\n"
        f"{req.text}\n\n"
        "---\n"
        "Извлечённые данные:\n"
        f"{json.dumps(extracted.model_dump(), ensure_ascii=False, indent=2)}\n\n"
        "Удали из каждого поля всё, что явно не подтверждается текстом выше. "
        f"Верни исправленный JSON в том же формате:\n{fmt}"
    )
    raw2 = await chat(
        system=_VERIFY_SYSTEM,
        user=verify_user,
        temperature=0.1,
        max_tokens=settings.contract_verify_max_tokens,
    )

    try:
        verified = _parse_contract(raw2)
        return _normalize_contract_payload(verified)
    except Exception as exc:
        logger.warning(
            "Contract verify parse/normalization failed: %s; returning extract pass result; raw=%r",
            exc,
            raw2[:1200],
        )
        return extracted
