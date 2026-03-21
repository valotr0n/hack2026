from __future__ import annotations

import json
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from ..llm import chat

router = APIRouter()


class CompareRequest(BaseModel):
    text_a: str
    text_b: str
    label_a: str = "Документ A"
    label_b: str = "Документ Б"


class CompareChange(BaseModel):
    section: str    # к какому разделу/пункту относится
    type: str       # added | removed | modified | conflict
    description: str
    severity: str   # critical | warning | info


class CompareResponse(BaseModel):
    changes: list[CompareChange]
    summary: str           # 2-3 предложения о главных расхождениях
    risk_level: str        # high | medium | low — насколько критичны различия


@router.post(
    "/compare",
    response_model=CompareResponse,
    summary="Сравнение двух документов",
    description="""
Сравнивает два документа и возвращает список расхождений: что добавили, убрали, изменили,
и есть ли противоречия между документами.

**Типы изменений:** `added` · `removed` · `modified` · `conflict`
**Severity:** `critical` · `warning` · `info`

**Ответ:**
```json
{
  "changes": [
    {
      "section": "п. 3.1 Процентная ставка",
      "type": "modified",
      "description": "Ставка изменена с 12% до 14% годовых",
      "severity": "critical"
    },
    {
      "section": "п. 7. Досрочное погашение",
      "type": "removed",
      "description": "Условие о досрочном погашении без штрафа удалено",
      "severity": "warning"
    }
  ],
  "summary": "Новая версия договора содержит повышение ставки и ужесточение штрафных условий...",
  "risk_level": "high"
}
```
    """,
)
async def compare_documents(req: CompareRequest) -> CompareResponse:
    system = (
        "Ты — опытный юридический аналитик банка. "
        "Сравниваешь два документа и находишь все расхождения. "
        "Особое внимание — на изменения в финансовых условиях, сроках и штрафах. "
        "Отвечай строго в формате JSON без лишнего текста."
    )
    user = (
        f"Сравни два документа и найди все расхождения.\n\n"
        f"=== {req.label_a} ===\n{req.text_a}\n\n"
        f"=== {req.label_b} ===\n{req.text_b}\n\n"
        "Верни JSON:\n"
        '{"changes": [{"section": "раздел/пункт", '
        '"type": "added|removed|modified|conflict", '
        '"description": "что изменилось — конкретно", '
        '"severity": "critical|warning|info"}], '
        '"summary": "2-3 предложения о главных расхождениях", '
        '"risk_level": "high|medium|low"}\n\n'
        "Правила:\n"
        "- `critical` — изменения в деньгах, сроках, штрафах, правах сторон\n"
        "- `warning` — изменения в процедурах, условиях расторжения\n"
        "- `info` — технические и редакционные правки\n"
        "- Если документы идентичны — верни пустой список изменений\n"
        "- Описание должно содержать конкретные значения (было → стало)"
    )

    raw = await chat(system=system, user=user)

    try:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        data = json.loads(raw[start:end])
        return CompareResponse(
            changes=[CompareChange(**c) for c in data.get("changes", [])],
            summary=data.get("summary", ""),
            risk_level=data.get("risk_level", "low"),
        )
    except Exception:
        raise HTTPException(status_code=500, detail="Не удалось разобрать ответ модели")
