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
        "Отвечай строго в формате JSON, без лишнего текста."
    )

    user = (
        "Выдели ключевые понятия и связи между ними из текста ниже. "
        "Верни JSON в следующем формате:\n"
        '{"nodes": [{"id": "1", "label": "понятие"}], '
        '"edges": [{"from": "1", "to": "2", "label": "связь"}]}\n\n'
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
