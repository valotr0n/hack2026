import logging
from fastapi import APIRouter
from pydantic import BaseModel
from ..json_utils import parse_json_payload, safe_sample, top_keywords
from ..llm import chat

router = APIRouter()
logger = logging.getLogger(__name__)

DOC_TYPES = ["договор", "отчёт", "инструкция", "письмо", "книга", "судебный документ", "научная статья", "новость", "прочее"]


class AutotagRequest(BaseModel):
    text: str  # первые ~500 символов документа


class AutotagResponse(BaseModel):
    doc_type: str
    tags: list[str]


def _fallback_autotag(text: str) -> AutotagResponse:
    lowered = text.lower()
    if any(token in lowered for token in ("договор", "соглашение", "заемщик", "кредитор")):
        doc_type = "договор"
    elif any(token in lowered for token in ("отчет", "баланс", "прибыль", "выручка")):
        doc_type = "отчёт"
    elif any(token in lowered for token in ("инструкция", "порядок", "регламент")):
        doc_type = "инструкция"
    elif any(token in lowered for token in ("суд", "иск", "арбитраж")):
        doc_type = "судебный документ"
    elif any(token in lowered for token in ("исследование", "метод", "эксперимент")):
        doc_type = "научная статья"
    else:
        doc_type = "прочее"
    tags = top_keywords(text, limit=5)
    return AutotagResponse(doc_type=doc_type, tags=tags)


@router.post("/autotag", response_model=AutotagResponse)
async def autotag(req: AutotagRequest) -> AutotagResponse:
    system = (
        "Ты — классификатор документов. "
        "Отвечай строго в формате JSON, без лишнего текста."
    )

    user = (
        f"Определи тип документа и проставь теги по его началу.\n\n"
        f"Возможные типы: {', '.join(DOC_TYPES)}.\n\n"
        "Верни JSON:\n"
        '{"doc_type": "тип", "tags": ["тег1", "тег2", "тег3"]}\n\n'
        "Правила:\n"
        "- doc_type — один из предложенных типов\n"
        "- tags — 2-5 коротких тегов на русском, отражающих тему документа\n"
        "- теги конкретные: не 'документ', не 'текст', а например 'кредитный договор', 'ипотека', 'Сбербанк'\n\n"
        f"Начало документа:\n{req.text[:500]}"
    )

    try:
        raw = await chat(system=system, user=user)
    except Exception as exc:
        logger.warning("Autotag generation failed before parsing: %s", exc)
        return _fallback_autotag(req.text)

    data = parse_json_payload(raw)
    if not isinstance(data, dict):
        logger.warning("Autotag parse failed, using fallback. raw_sample=%r", safe_sample(raw))
        return _fallback_autotag(req.text)

    try:
        doc_type = data.get("doc_type", "прочее")
        tags = data.get("tags", [])
        if doc_type not in DOC_TYPES:
            doc_type = "прочее"
        return AutotagResponse(doc_type=doc_type, tags=tags[:5])
    except Exception as exc:
        logger.warning("Autotag normalization failed, using fallback: %s; data=%r", exc, data)
        return _fallback_autotag(req.text)
