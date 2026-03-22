from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from ..llm import chat

logger = logging.getLogger("content_service.knowledge_graph")

router = APIRouter()

_CHUNK_SIZE = 15_000


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
    try:
        user = (
            f"{part_label}\n\n"
            f"Извлеки граф знаний из этого фрагмента. Верни JSON:\n{_GRAPH_FORMAT}\n\n"
            f"Правила:\n{_EXTRACT_RULES}\n"
            f"Фрагмент:\n{chunk}"
        )
        raw = await chat(system=_EXTRACT_SYSTEM, user=user, temperature=0.3)
        return _parse_graph(raw)
    except Exception as e:
        logger.warning("Chunk extraction failed (%s), skipping", type(e).__name__)
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

    # Программная дедупликация nodes по id, edges по (source, target, label)
    seen_nodes: set[str] = set()
    deduped_nodes: list[dict] = []
    for n in all_nodes:
        nid = n.get("id", "").strip().lower()
        if nid and nid not in seen_nodes:
            seen_nodes.add(nid)
            deduped_nodes.append(n)

    seen_edges: set[tuple[str, str, str]] = set()
    deduped_edges: list[dict] = []
    for e in all_edges:
        key = (e.get("source", ""), e.get("target", ""), e.get("label", "").strip().lower())
        # Оставляем только рёбра между известными узлами
        if key not in seen_edges and e.get("source") in seen_nodes and e.get("target") in seen_nodes:
            seen_edges.add(key)
            deduped_edges.append(e)

    logger.info("KnowledgeGraph dedup done nodes=%d edges=%d", len(deduped_nodes), len(deduped_edges))
    return {"nodes": deduped_nodes, "edges": deduped_edges}


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
