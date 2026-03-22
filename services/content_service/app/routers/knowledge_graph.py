from __future__ import annotations

import logging
from fastapi import APIRouter
from pydantic import BaseModel
from ..json_utils import parse_json_payload, safe_sample, top_keywords
from ..llm import chat

router = APIRouter()
logger = logging.getLogger(__name__)


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


def _fallback_knowledge_graph(text: str) -> KnowledgeGraphResponse:
    keywords = top_keywords(text, limit=8)
    nodes = [
        KGNode(id=f"node_{index + 1}", label=keyword[:60], type="concept")
        for index, keyword in enumerate(keywords)
    ]
    edges = [
        KGEdge(source=nodes[index].id, target=nodes[index + 1].id, label="связано с")
        for index in range(max(0, len(nodes) - 1))
    ]
    return KnowledgeGraphResponse(nodes=nodes, edges=edges)


@router.post(
    "/knowledge-graph",
    response_model=KnowledgeGraphResponse,
    summary="Граф знаний",
    description="""
Извлекает сущности и связи между ними из текста и возвращает интерактивный граф знаний.

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
    system = (
        "Ты — эксперт по извлечению знаний и построению графов. "
        "Извлекай только смысловые сущности из содержания текста — понятия, явления, процессы, организации, персоны. "
        "Игнорируй библиографические метаданные: ISBN, УДК, ББК, даты публикации, издательства, грифы. "
        "Все подписи (label) и рёбра (label) — строго на русском языке. "
        "Отвечай строго в формате JSON без лишнего текста."
    )
    user = (
        "Извлеки граф знаний из текста. Верни JSON:\n"
        '{"nodes": [{"id": "slug_латиницей", "label": "название на русском", '
        '"type": "concept|person|org|event|place|term"}], '
        '"edges": [{"source": "id_узла", "target": "id_узла", "label": "связь на русском"}]}\n'
        "Правила:\n"
        "- Минимум 8 узлов из содержания документа (не из титульной страницы)\n"
        "- id — уникальный slug латиницей без пробелов\n"
        "- label ребра — глагол или фраза на русском: 'определяет', 'является частью', 'влияет на'\n"
        "- Каждый узел участвует хотя бы в одной связи\n"
        "- Не включать: авторов, ISBN, УДК, ББК, издательства, даты выхода книги\n\n"
        f"Текст:\n{req.text}"
    )

    try:
        raw = await chat(system=system, user=user, temperature=0.3)
    except Exception as exc:
        logger.warning("Knowledge graph generation failed before parsing: %s", exc)
        return _fallback_knowledge_graph(req.text)

    data = parse_json_payload(raw)
    if not isinstance(data, dict):
        logger.warning("Knowledge graph parse failed, using fallback. raw_sample=%r", safe_sample(raw))
        return _fallback_knowledge_graph(req.text)

    try:
        return KnowledgeGraphResponse(
            nodes=[KGNode(**n) for n in data.get("nodes", [])],
            edges=[KGEdge(**e) for e in data.get("edges", [])],
        )
    except Exception as exc:
        logger.warning("Knowledge graph normalization failed, using fallback: %s; data=%r", exc, data)
        return _fallback_knowledge_graph(req.text)
