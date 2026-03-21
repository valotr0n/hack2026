from __future__ import annotations

import json
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from ..llm import chat

router = APIRouter()


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
        "Извлекай ключевые сущности и осмысленные связи между ними. "
        "Отвечай строго в формате JSON без лишнего текста."
    )
    user = (
        "Извлеки граф знаний из текста. Верни JSON:\n"
        '{"nodes": [{"id": "slug_en", "label": "название на языке текста", '
        '"type": "concept|person|org|event|date|place|term"}], '
        '"edges": [{"source": "id_узла", "target": "id_узла", "label": "тип связи"}]}\n'
        "Правила: минимум 8 узлов, id — slug на английском (без пробелов), "
        "каждый узел должен участвовать хотя бы в одной связи.\n\n"
        f"Текст:\n{req.text}"
    )

    raw = await chat(system=system, user=user)

    try:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        data = json.loads(raw[start:end])
        return KnowledgeGraphResponse(
            nodes=[KGNode(**n) for n in data.get("nodes", [])],
            edges=[KGEdge(**e) for e in data.get("edges", [])],
        )
    except Exception:
        raise HTTPException(status_code=500, detail="Не удалось разобрать ответ модели")
