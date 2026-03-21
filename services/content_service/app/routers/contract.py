from __future__ import annotations

import json
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from ..llm import chat

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


@router.post(
    "/contract",
    response_model=ContractResponse,
    summary="Анализ договора",
    description="""
Извлекает структурированную информацию из текста договора:
стороны, предмет, ключевые условия, обязательства, риски, сроки, штрафы.

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
    system = (
        "Ты — юридический аналитик банка. Тщательно извлекай структурированную информацию из договоров. "
        "Отвечай строго в формате JSON без лишнего текста."
    )
    user = (
        "Проанализируй договор и верни JSON строго в формате:\n"
        '{"parties": ["сторона 1", "сторона 2"], '
        '"subject": "предмет договора в 1-2 предложениях", '
        '"key_conditions": ["условие 1", "условие 2"], '
        '"obligations": [{"party": "название стороны", "text": "обязательство"}], '
        '"risks": ["риск 1", "риск 2"], '
        '"deadlines": ["срок 1", "срок 2"], '
        '"penalties": ["штраф/пеня 1"]}\n\n'
        f"Договор:\n{req.text}"
    )

    raw = await chat(system=system, user=user)

    try:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        data = json.loads(raw[start:end])
        return ContractResponse(
            parties=data.get("parties", []),
            subject=data.get("subject", ""),
            key_conditions=data.get("key_conditions", []),
            obligations=[Obligation(**o) for o in data.get("obligations", [])],
            risks=data.get("risks", []),
            deadlines=data.get("deadlines", []),
            penalties=data.get("penalties", []),
        )
    except Exception:
        raise HTTPException(status_code=500, detail="Не удалось разобрать ответ модели")
