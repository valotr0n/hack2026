import json
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from ..llm import chat

router = APIRouter()


class MindmapRequest(BaseModel):
    text: str


class MindmapResponse(BaseModel):
    nodes: list[dict]
    edges: list[dict]


@router.post("/mindmap", response_model=MindmapResponse)
async def generate_mindmap(req: MindmapRequest) -> MindmapResponse:
    system = (
        "Ты — ассистент по анализу документов. "
        "Используй ТОЛЬКО информацию из предоставленного текста. "
        "Отвечай строго в формате JSON, без лишнего текста, без пояснений до или после JSON."
    )

    user = (
        "Построй mindmap по тексту ниже.\n\n"
        "Правила:\n"
        "1. nodes — уникальные конкретные понятия из текста (не абстрактные слова вроде 'понятие', 'термин', 'содержит').\n"
        "2. Каждый node встречается ровно один раз, id — уникальная строка-число.\n"
        "3. edges — смысловые связи. label ребра — конкретный глагол или фраза, описывающая отношение (например: 'использует', 'определяет', 'зависит от', 'является частью'). Запрещено использовать 'содержит' как единственный label.\n"
        "4. Не создавай рёбра между одинаковыми узлами.\n\n"
        "Формат ответа (только JSON, ничего лишнего):\n"
        '{"nodes": [{"id": "1", "label": "понятие"}], '
        '"edges": [{"from": "1", "to": "2", "label": "конкретная связь"}]}\n\n'
        f"Текст:\n{req.text}"
    )

    raw = await chat(system=system, user=user)

    try:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        data = json.loads(raw[start:end])
        return MindmapResponse(nodes=data["nodes"], edges=data["edges"])
    except Exception:
        raise HTTPException(status_code=500, detail="Не удалось разобрать ответ модели")
