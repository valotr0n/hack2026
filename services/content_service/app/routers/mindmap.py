import logging
from fastapi import APIRouter
from pydantic import BaseModel
from ..json_utils import candidate_sentences, parse_json_payload, top_keywords
from ..llm import chat

router = APIRouter()
logger = logging.getLogger(__name__)


class MindmapNode(BaseModel):
    title: str
    children: list["MindmapNode"] = []


MindmapNode.model_rebuild()


class MindmapRequest(BaseModel):
    text: str


class MindmapResponse(BaseModel):
    title: str
    children: list[MindmapNode] = []


def _fallback_mindmap(text: str) -> MindmapResponse:
    keywords = top_keywords(text, limit=4)
    sentences = candidate_sentences(text, min_length=20, max_items=8)
    children: list[MindmapNode] = []
    for index, keyword in enumerate(keywords or ["Ключевая тема", "Детали", "Выводы"]):
        branch_sentences = sentences[index * 2:(index + 1) * 2]
        branch_children = [MindmapNode(title=sentence[:60], children=[]) for sentence in branch_sentences]
        children.append(MindmapNode(title=keyword[:40].capitalize(), children=branch_children))
    title = (keywords[0].capitalize() if keywords else "Структура документа")
    return MindmapResponse(title=title, children=children[:4])


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

    try:
        raw = await chat(system=system, user=user, temperature=0.3)
    except Exception as exc:
        logger.warning("Mindmap generation failed before parsing: %s", exc)
        return _fallback_mindmap(req.text)

    data = parse_json_payload(raw)
    if not isinstance(data, dict):
        logger.warning("Mindmap parse failed, using fallback. raw_sample=%r", raw[:500])
        return _fallback_mindmap(req.text)

    try:
        return MindmapResponse(title=data["title"], children=data.get("children", []))
    except Exception as exc:
        logger.warning("Mindmap normalization failed, using fallback: %s; data=%r", exc, data)
        return _fallback_mindmap(req.text)
