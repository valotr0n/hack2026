from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from ..llm import chat

logger = logging.getLogger("content_service.knowledge_graph")

router = APIRouter()

_CHUNK_SIZE = 20_000


class KnowledgeGraphRequest(BaseModel):
    text: str


class KGNode(BaseModel):
    id: str
    label: str
    type: str  # concept | person | org | event | date | place | term


class KGEdge(BaseModel):
    source: str
    target: str
    label: str


class KnowledgeGraphResponse(BaseModel):
    nodes: list[KGNode]
    edges: list[KGEdge]


_EXTRACT_SYSTEM = (
    "Ты — эксперт по извлечению знаний и построению графов. "
    "Извлекай только смысловые сущности из содержания текста — понятия, явления, процессы, организации, персоны. "
    "Игнорируй библиографические метаданные: ISBN, УДК, ББК, даты публикации, издательства, грифы. "
    "Все подписи (label) и рёбра (label) — строго на русском языке. "
    "Отвечай строго в формате JSON без лишнего текста."
)

_MERGE_SYSTEM = (
    "Ты — эксперт по слиянию графов знаний. "
    "Тебе даны узлы и рёбра из нескольких фрагментов одного документа. "
    "Объедини их: удали дублирующиеся узлы (одна сущность — один узел с уникальным id), "
    "добавь связи между узлами из разных фрагментов если они очевидно связаны. "
    "Все id — уникальные slug латиницей, все label — на русском языке. "
    "Отвечай строго в формате JSON без лишнего текста."
)

_GRAPH_FORMAT = (
    '{"nodes": [{"id": "slug_латиницей", "label": "название на русском", '
    '"type": "concept|person|org|event|place|term"}], '
    '"edges": [{"source": "id_узла", "target": "id_узла", "label": "связь на русском"}]}'
)

_EXTRACT_RULES = (
    "- Минимум 5 узлов из содержания фрагмента (не из титульной страницы)\n"
    "- id — уникальный slug латиницей без пробелов\n"
    "- label ребра — глагол или фраза на русском: 'определяет', 'является частью', 'влияет на'\n"
    "- Каждый узел участвует хотя бы в одной связи\n"
    "- Не включать: авторов, ISBN, УДК, ББК, издательства, даты выхода книги\n"
)


def _split_text(text: str, chunk_size: int) -> list[str]:
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end >= len(text):
            chunks.append(text[start:].strip())
            break
        cut = text.rfind("\n", start, end)
        if cut == -1 or cut <= start:
            cut = text.rfind(" ", start, end)
        if cut == -1 or cut <= start:
            cut = end
        chunks.append(text[start:cut].strip())
        start = cut + 1
    return [c for c in chunks if c]


def _parse_graph(raw: str) -> dict:
    start = raw.find("{")
    end = raw.rfind("}") + 1
    return json.loads(raw[start:end])


async def _extract_chunk_graph(chunk: str, part_label: str) -> dict:
    user = (
        f"{part_label}\n\n"
        f"Извлеки граф знаний из этого фрагмента. Верни JSON:\n{_GRAPH_FORMAT}\n\n"
        f"Правила:\n{_EXTRACT_RULES}\n"
        f"Фрагмент:\n{chunk}"
    )
    raw = await chat(system=_EXTRACT_SYSTEM, user=user, temperature=0.3)
    try:
        return _parse_graph(raw)
    except Exception:
        return {"nodes": [], "edges": []}


async def _hierarchical_knowledge_graph(text: str) -> dict:
    if len(text) <= _CHUNK_SIZE:
        logger.info("KnowledgeGraph single-pass chars=%d", len(text))
        user = (
            f"Извлеки граф знаний из текста. Верни JSON:\n{_GRAPH_FORMAT}\n\n"
            f"Правила:\n{_EXTRACT_RULES}\n"
            f"Текст:\n{text}"
        )
        raw = await chat(system=_EXTRACT_SYSTEM, user=user, temperature=0.3)
        return _parse_graph(raw)

    chunks = _split_text(text, _CHUNK_SIZE)
    n = len(chunks)
    logger.info("KnowledgeGraph map-reduce started chars=%d chunks=%d", len(text), n)

    tasks = [
        _extract_chunk_graph(chunk, f"Фрагмент {i + 1} из {n}:")
        for i, chunk in enumerate(chunks)
    ]
    partial: list[dict] = await asyncio.gather(*tasks)

    all_nodes = [node for g in partial for node in g.get("nodes", [])]
    all_edges = [edge for g in partial for edge in g.get("edges", [])]
    logger.info("KnowledgeGraph map done nodes=%d edges=%d", len(all_nodes), len(all_edges))

    combined = json.dumps(
        {"nodes": all_nodes, "edges": all_edges},
        ensure_ascii=False,
        indent=2,
    )
    merge_user = (
        "Ниже собраны узлы и рёбра из всех фрагментов одного документа. "
        "Объедини их в единый связный граф: удали дублирующиеся сущности, "
        "нормализуй id (один уникальный slug на каждую сущность), "
        "добавь связи между сущностями из разных фрагментов где это очевидно. "
        f"Верни JSON в том же формате:\n{_GRAPH_FORMAT}\n\n"
        f"Собранные данные:\n{combined}"
    )
    raw = await chat(system=_MERGE_SYSTEM, user=merge_user, temperature=0.3)
    try:
        result = _parse_graph(raw)
        logger.info(
            "KnowledgeGraph merge done nodes=%d edges=%d",
            len(result.get("nodes", [])),
            len(result.get("edges", [])),
        )
        return result
    except Exception:
        logger.warning("KnowledgeGraph merge parse failed, returning unmerged")
        return {"nodes": all_nodes, "edges": all_edges}


@router.post(
    "/knowledge-graph",
    response_model=KnowledgeGraphResponse,
    summary="Граф знаний",
    description="""
Извлекает сущности и связи между ними из текста и возвращает интерактивный граф знаний.
Для больших документов автоматически применяется **map-reduce**: каждый фрагмент
обрабатывается параллельно, затем узлы и рёбра объединяются в единый граф.

**Типы узлов:** `concept`, `person`, `org`, `event`, `date`, `place`, `term`

**Ответ:**
```json
{
  "nodes": [
    {"id": "rag", "label": "RAG", "type": "concept"},
    {"id": "openai", "label": "OpenAI", "type": "org"}
  ],
  "edges": [
    {"source": "rag", "target": "openai", "label": "разработан"}
  ]
}
```
    """,
)
async def generate_knowledge_graph(req: KnowledgeGraphRequest) -> KnowledgeGraphResponse:
    data = await _hierarchical_knowledge_graph(req.text)
    try:
        return KnowledgeGraphResponse(
            nodes=[KGNode(**n) for n in data.get("nodes", [])],
            edges=[KGEdge(**e) for e in data.get("edges", [])],
        )
    except Exception:
        raise HTTPException(status_code=500, detail="Не удалось разобрать ответ модели")
