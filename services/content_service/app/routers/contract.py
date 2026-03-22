from __future__ import annotations

import json
import logging
import time

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from ..llm import chat

logger = logging.getLogger("content_service.contract")

router = APIRouter()


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
    start = raw.find("{")
    end = raw.rfind("}") + 1
    return json.loads(raw[start:end])


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
    logger.info("Contract analysis started chars=%d", len(req.text))
    started_at = time.perf_counter()
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
    logger.info("Contract pass 1/2: extracting...")
    extract_user = (
        "Извлеки из договора данные строго по тексту. "
        "Если поле не упоминается явно — оставь пустым.\n\n"
        f"Формат:\n{fmt}\n\n"
        f"Договор:\n{req.text}"
    )
    raw1 = await chat(system=_EXTRACT_SYSTEM, user=extract_user, temperature=0.1)

    try:
        data = _parse_contract(raw1)
    except Exception:
        raise HTTPException(status_code=500, detail="Не удалось разобрать ответ модели (проход 1)")

    # Проход 2: верификация
    logger.info("Contract pass 2/2: verifying...")
    verify_user = (
        "Исходный текст договора:\n"
        f"{req.text}\n\n"
        "---\n"
        "Извлечённые данные:\n"
        f"{json.dumps(data, ensure_ascii=False, indent=2)}\n\n"
        "Удали из каждого поля всё, что явно не подтверждается текстом выше. "
        f"Верни исправленный JSON в том же формате:\n{fmt}"
    )
    raw2 = await chat(system=_VERIFY_SYSTEM, user=verify_user, temperature=0.1)

    try:
        verified = _parse_contract(raw2)
        logger.info("Contract analysis done %.2fs", time.perf_counter() - started_at)
        return ContractResponse(
            parties=verified.get("parties", []),
            subject=verified.get("subject", ""),
            key_conditions=verified.get("key_conditions", []),
            obligations=[Obligation(**o) for o in verified.get("obligations", [])],
            risks=verified.get("risks", []),
            deadlines=verified.get("deadlines", []),
            penalties=verified.get("penalties", []),
        )
    except Exception:
        raise HTTPException(status_code=500, detail="Не удалось разобрать ответ модели (проход 2)")
