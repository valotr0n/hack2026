import json
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from ..llm import chat

router = APIRouter()

DOC_TYPES = ["договор", "отчёт", "инструкция", "письмо", "книга", "судебный документ", "научная статья", "новость", "прочее"]


class AutotagRequest(BaseModel):
    text: str  # первые ~500 символов документа


class AutotagResponse(BaseModel):
    doc_type: str
    tags: list[str]


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

    raw = await chat(system=system, user=user)

    try:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        data = json.loads(raw[start:end])
        doc_type = data.get("doc_type", "прочее")
        tags = data.get("tags", [])
        if doc_type not in DOC_TYPES:
            doc_type = "прочее"
        return AutotagResponse(doc_type=doc_type, tags=tags[:5])
    except Exception:
        raise HTTPException(status_code=500, detail="Не удалось разобрать ответ модели")
