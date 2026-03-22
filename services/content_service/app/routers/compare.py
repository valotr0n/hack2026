from __future__ import annotations

import difflib
import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from ..config import settings
from ..json_utils import parse_json_payload, safe_sample
from ..llm import chat

router = APIRouter()
logger = logging.getLogger(__name__)


class CompareRequest(BaseModel):
    text_a: str
    text_b: str
    label_a: str = "Документ A"
    label_b: str = "Документ Б"


class CompareChange(BaseModel):
    section: str      # раздел/пункт договора
    type: str         # added | removed | modified | conflict
    description: str  # что изменилось
    quote_a: str = "" # цитата из оригинала (было)
    quote_b: str = "" # цитата из новой версии (стало)
    severity: str     # critical | warning | info


class CompareResponse(BaseModel):
    changes: list[CompareChange]
    summary: str
    risk_level: str   # high | medium | low


def _severity_for_text(text: str) -> str:
    normalized = text.lower()
    if any(token in normalized for token in ("сумм", "руб", "%", "процент", "ставк", "срок", "дата", "штраф", "пен", "неустой", "платеж")):
        return "critical"
    if any(token in normalized for token in ("услов", "расторж", "поряд", "процедур", "обязан", "прав")):
        return "warning"
    return "info"


def _risk_level(changes: list[CompareChange]) -> str:
    severities = {change.severity for change in changes}
    if "critical" in severities:
        return "high"
    if "warning" in severities:
        return "medium"
    return "low"


def _fallback_compare(req: CompareRequest) -> CompareResponse:
    lines_a = [line.strip() for line in req.text_a.splitlines() if line.strip()]
    lines_b = [line.strip() for line in req.text_b.splitlines() if line.strip()]

    if lines_a == lines_b:
        return CompareResponse(
            changes=[],
            summary="Существенных различий между документами не обнаружено.",
            risk_level="low",
        )

    matcher = difflib.SequenceMatcher(a=lines_a, b=lines_b)
    changes: list[CompareChange] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        quote_a = " ".join(lines_a[i1:i2])[:500]
        quote_b = " ".join(lines_b[j1:j2])[:500]
        combined = f"{quote_a} {quote_b}".strip()
        severity = _severity_for_text(combined)
        change_type = {
            "insert": "added",
            "delete": "removed",
            "replace": "modified",
        }.get(tag, "conflict")
        changes.append(
            CompareChange(
                section="Без точного раздела",
                type=change_type,
                description=(
                    "Добавлен новый фрагмент."
                    if change_type == "added"
                    else "Удалён фрагмент."
                    if change_type == "removed"
                    else "Изменён текст документа."
                ),
                quote_a=quote_a,
                quote_b=quote_b,
                severity=severity,
            )
        )
        if len(changes) >= 12:
            break

    if not changes:
        changes.append(
            CompareChange(
                section="Без точного раздела",
                type="modified",
                description="Документы отличаются, но различия не удалось структурировать автоматически.",
                quote_a=req.text_a[:300],
                quote_b=req.text_b[:300],
                severity="warning",
            )
        )

    return CompareResponse(
        changes=changes,
        summary=f"Найдено {len(changes)} различий между {req.label_a} и {req.label_b}.",
        risk_level=_risk_level(changes),
    )


@router.post(
    "/compare",
    response_model=CompareResponse,
    summary="Сравнение двух документов",
    description="""
Сравнивает два документа и возвращает список расхождений с цитатами из оригиналов.

**Типы изменений:** `added` · `removed` · `modified` · `conflict`
**Severity:** `critical` · `warning` · `info`

**Ответ:**
```json
{
  "changes": [
    {
      "section": "п. 3.1 Процентная ставка",
      "type": "modified",
      "description": "Ставка повышена",
      "quote_a": "процентная ставка составляет 12% годовых",
      "quote_b": "процентная ставка составляет 14% годовых",
      "severity": "critical"
    }
  ],
  "summary": "Новая версия договора содержит повышение ставки...",
  "risk_level": "high"
}
```
    """,
)
async def compare_documents(req: CompareRequest) -> CompareResponse:
    system = (
        "Ты — опытный юридический аналитик банка. "
        "Сравниваешь два документа и находишь все расхождения. "
        "Для каждого изменения обязательно приводи точные цитаты из обоих документов — "
        "дословно как написано в тексте, без додумывания. "
        "Особое внимание — на изменения в финансовых условиях, сроках и штрафах. "
        "Отвечай строго в формате JSON без лишнего текста."
    )
    user = (
        f"Сравни два документа и найди все расхождения.\n\n"
        f"=== {req.label_a} ===\n{req.text_a}\n\n"
        f"=== {req.label_b} ===\n{req.text_b}\n\n"
        "Верни JSON:\n"
        '{"changes": [{'
        '"section": "номер пункта или раздел", '
        '"type": "added|removed|modified|conflict", '
        '"description": "суть изменения — конкретно что изменилось", '
        '"quote_a": "дословная цитата из первого документа или пустая строка если добавлено", '
        '"quote_b": "дословная цитата из второго документа или пустая строка если удалено", '
        '"severity": "critical|warning|info"}], '
        '"summary": "2-3 предложения о главных расхождениях", '
        '"risk_level": "high|medium|low"}\n\n'
        "Правила:\n"
        "- `critical` — изменения в деньгах, сроках, штрафах, правах сторон\n"
        "- `warning` — изменения в процедурах, условиях расторжения\n"
        "- `info` — технические и редакционные правки\n"
        "- quote_a и quote_b — точные цитаты из текста, не перефраз\n"
        "- Если документы идентичны — верни пустой список изменений\n"
        "- Описание должно содержать конкретные значения: было → стало"
    )

    try:
        raw = await chat(
            system=system,
            user=user,
            temperature=0.1,
            max_tokens=settings.compare_max_tokens,
        )
    except Exception as exc:
        logger.warning("Compare generation failed before parsing: %s", exc)
        return _fallback_compare(req)

    data = parse_json_payload(raw)
    if not isinstance(data, dict):
        logger.warning("Compare parse failed, using fallback. raw_sample=%r", safe_sample(raw))
        return _fallback_compare(req)

    try:
        return CompareResponse(
            changes=[CompareChange(**c) for c in data.get("changes", [])],
            summary=data.get("summary", ""),
            risk_level=data.get("risk_level", "low"),
        )
    except Exception as exc:
        logger.warning("Compare normalization failed, using fallback: %s; data=%r", exc, data)
        return _fallback_compare(req)
