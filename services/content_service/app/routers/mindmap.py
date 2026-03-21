import json
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from ..llm import chat

router = APIRouter()


class MindmapNode(BaseModel):
    title: str
    children: list["MindmapNode"] = []


MindmapNode.model_rebuild()


class MindmapRequest(BaseModel):
    text: str


class MindmapResponse(BaseModel):
    title: str
    children: list[MindmapNode] = []


@router.post("/mindmap", response_model=MindmapResponse)
async def generate_mindmap(req: MindmapRequest) -> MindmapResponse:
    system = (
        "Ты — ассистент по анализу документов. "
        "Используй ТОЛЬКО информацию из предоставленного текста. "
        "Отвечай строго в формате JSON, без лишнего текста, без пояснений до или после JSON."
    )

    user = (
        "Построй mindmap по тексту ниже в виде иерархического дерева.\n\n"
        "Правила:\n"
        "1. title корня — главная тема документа (одна фраза).\n"
        "2. children корня — 3-7 ключевых разделов/тем.\n"
        "3. У каждого раздела children — конкретные подпункты (2-5 штук).\n"
        "4. Глубина дерева — не более 3 уровней.\n"
        "5. Названия узлов — короткие конкретные фразы (2-5 слов), без воды.\n"
        "6. Не дублируй понятия на разных ветках.\n\n"
        "Формат ответа (только JSON, ничего лишнего):\n"
        '{"title": "Главная тема", "children": ['
        '{"title": "Раздел 1", "children": ['
        '{"title": "Подпункт 1.1", "children": []}, '
        '{"title": "Подпункт 1.2", "children": []}]}, '
        '{"title": "Раздел 2", "children": []}]}\n\n'
        f"Текст:\n{req.text}"
    )

    raw = await chat(system=system, user=user, temperature=0.3)

    try:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        data = json.loads(raw[start:end])
        return MindmapResponse(title=data["title"], children=data.get("children", []))
    except Exception:
        raise HTTPException(status_code=500, detail="Не удалось разобрать ответ модели")
