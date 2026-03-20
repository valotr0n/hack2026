from __future__ import annotations

import asyncio
import inspect
import json
from io import BytesIO
from pathlib import Path
from typing import Any, AsyncIterator, Literal
from uuid import uuid4

import httpx
from docx import Document as DocxDocument
from fastapi import HTTPException, UploadFile, status
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document
from openai import AsyncOpenAI
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from .config import settings
from .schemas import ChatHistoryMessage

CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
TOP_K = 5
EmbeddingPromptName = Literal["search_query", "search_document"]

FRAGMENT_LABEL = "[\u0424\u0440\u0430\u0433\u043c\u0435\u043d\u0442 {index}]"
SOURCE_LABEL = "\u0418\u0441\u0442\u043e\u0447\u043d\u0438\u043a: {source}"
CHUNK_INDEX_LABEL = "\u041d\u043e\u043c\u0435\u0440 \u0447\u0430\u043d\u043a\u0430: {chunk_index}"
SYSTEM_PROMPT = (
    "\u0422\u044b \u0430\u0441\u0441\u0438\u0441\u0442\u0435\u043d\u0442, "
    "\u043e\u0442\u0432\u0435\u0447\u0430\u044e\u0449\u0438\u0439 \u0422\u041e\u041b\u042c\u041a\u041e \u043d\u0430 \u043e\u0441\u043d\u043e\u0432\u0435 "
    "\u043f\u0440\u0435\u0434\u043e\u0441\u0442\u0430\u0432\u043b\u0435\u043d\u043d\u043e\u0433\u043e \u043a\u043e\u043d\u0442\u0435\u043a\u0441\u0442\u0430.\n"
    "\u041e\u0442\u0432\u0435\u0447\u0430\u0439 \u043d\u0430 \u0440\u0443\u0441\u0441\u043a\u043e\u043c \u044f\u0437\u044b\u043a\u0435. "
    "\u0415\u0441\u043b\u0438 \u043e\u0442\u0432\u0435\u0442\u0430 \u043d\u0435\u0442 \u0432 \u043a\u043e\u043d\u0442\u0435\u043a\u0441\u0442\u0435 - "
    "\u0441\u043a\u0430\u0436\u0438 \u043e\u0431 \u044d\u0442\u043e\u043c.\n"
    "\u0412\u0441\u0435\u0433\u0434\u0430 \u0443\u043a\u0430\u0437\u044b\u0432\u0430\u0439, "
    "\u043d\u0430 \u043a\u0430\u043a\u043e\u043c \u0444\u0440\u0430\u0433\u043c\u0435\u043d\u0442\u0435 \u043e\u0441\u043d\u043e\u0432\u0430\u043d \u043e\u0442\u0432\u0435\u0442.\n\n"
    "\u041a\u043e\u043d\u0442\u0435\u043a\u0441\u0442:\n{context}"
)


async def extract_text_from_upload(file: UploadFile) -> str:
    filename = file.filename or "document"
    suffix = Path(filename).suffix.lower()
    payload = await file.read()

    if suffix == ".pdf":
        reader = PdfReader(BytesIO(payload))
        text = "\n".join((page.extract_text() or "").strip() for page in reader.pages)
    elif suffix == ".docx":
        document = DocxDocument(BytesIO(payload))
        text = "\n".join(paragraph.text.strip() for paragraph in document.paragraphs)
    elif suffix == ".txt":
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="TXT files must be UTF-8 encoded.",
            ) from exc
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported file type. Use PDF, DOCX, or TXT.",
        )

    normalized_text = "\n".join(line for line in text.splitlines() if line.strip()).strip()
    if not normalized_text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file does not contain readable text.",
        )

    return normalized_text


def split_text_into_chunks(full_text: str) -> list[str]:
    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    nodes = splitter.get_nodes_from_documents([Document(text=full_text)])
    chunks = [content for node in nodes if (content := node.get_content().strip())]

    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unable to split the document into chunks.",
        )

    return chunks


def get_embedding_dimension(embedding_model: SentenceTransformer) -> int:
    dimension = embedding_model.get_sentence_embedding_dimension()
    if not isinstance(dimension, int) or dimension <= 0:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Embedding model returned an invalid embedding dimension.",
        )
    return dimension


def _encode_texts(
    embedding_model: SentenceTransformer,
    texts: list[str],
    prompt_name: EmbeddingPromptName | None,
) -> list[list[float]]:
    encode_kwargs: dict[str, Any] = {
        "convert_to_numpy": True,
        "normalize_embeddings": True,
        "show_progress_bar": False,
    }
    if prompt_name is not None:
        encode_kwargs["prompt_name"] = prompt_name

    embeddings = embedding_model.encode(texts, **encode_kwargs)
    return embeddings.tolist()


async def fetch_embeddings(
    embedding_model: SentenceTransformer,
    texts: list[str],
    prompt_name: EmbeddingPromptName | None = None,
) -> list[list[float]]:
    try:
        embeddings = await asyncio.to_thread(
            _encode_texts,
            embedding_model,
            texts,
            prompt_name,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Local embedding generation failed: {exc}",
        ) from exc

    if len(embeddings) != len(texts):
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Embedding model returned an unexpected number of vectors.",
        )

    vector_size = get_embedding_dimension(embedding_model)
    for embedding in embeddings:
        if not isinstance(embedding, list) or len(embedding) != vector_size:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Embedding model returned an invalid vector size.",
            )

    return embeddings


async def _qdrant_request(
    method: str,
    path: str,
    body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    url = f"{settings.qdrant_url.rstrip('/')}{path}"
    request_kwargs: dict[str, Any] = {}
    if body is not None:
        request_kwargs["json"] = body

    try:
        async with httpx.AsyncClient(timeout=30.0, trust_env=False) as client:
            response = await client.request(method, url, **request_kwargs)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=(
                f"Qdrant request failed for {url} with status "
                f"{exc.response.status_code}: {exc.response.text!r}"
            ),
        ) from exc
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Qdrant request failed for {url}: {exc}",
        ) from exc


async def _create_qdrant_collection_via_rest(doc_id: str, embedding_dimension: int) -> None:
    await _qdrant_request(
        "PUT",
        f"/collections/{doc_id}",
        {"vectors": {"size": embedding_dimension, "distance": "Cosine"}},
    )


async def _upsert_qdrant_points_via_rest(
    doc_id: str,
    chunks: list[str],
    embeddings: list[list[float]],
    filename: str,
) -> None:
    points = [
        {
            "id": index,
            "vector": embedding,
            "payload": {
                "text": chunk,
                "source": filename,
                "chunk_index": index,
            },
        }
        for index, (chunk, embedding) in enumerate(zip(chunks, embeddings))
    ]
    await _qdrant_request(
        "PUT",
        f"/collections/{doc_id}/points?wait=true",
        {"points": points},
    )


async def create_document_collection(
    embedding_dimension: int,
    filename: str,
    chunks: list[str],
    embeddings: list[list[float]],
) -> str:
    doc_id = str(uuid4())
    await _create_qdrant_collection_via_rest(doc_id, embedding_dimension)
    await _upsert_qdrant_points_via_rest(doc_id, chunks, embeddings, filename)
    return doc_id


async def search_document_chunks(
    doc_id: str,
    query_embedding: list[float],
) -> list[dict[str, Any]]:
    await _qdrant_request("GET", f"/collections/{doc_id}")
    query_result = await _qdrant_request(
        "POST",
        f"/collections/{doc_id}/points/query",
        {
            "query": query_embedding,
            "limit": TOP_K,
            "with_payload": True,
        },
    )

    result = query_result.get("result") or {}
    points = result.get("points") or []

    chunks: list[dict[str, Any]] = []
    for point in points:
        payload = point.get("payload") or {}
        text = payload.get("text")
        if isinstance(text, str) and text.strip():
            chunks.append(
                {
                    "text": text,
                    "source": payload.get("source", "unknown"),
                    "chunk_index": payload.get("chunk_index"),
                }
            )

    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No relevant chunks were found for the requested document.",
        )

    return chunks


def build_system_prompt(chunks: list[dict[str, Any]]) -> tuple[str, list[str]]:
    sources = [chunk["text"] for chunk in chunks]
    context_parts = []
    for index, chunk in enumerate(chunks, start=1):
        context_parts.append(
            "\n".join(
                [
                    FRAGMENT_LABEL.format(index=index),
                    SOURCE_LABEL.format(source=chunk["source"]),
                    CHUNK_INDEX_LABEL.format(chunk_index=chunk["chunk_index"]),
                    chunk["text"],
                ]
            )
        )

    context = "\n\n".join(context_parts)
    prompt = SYSTEM_PROMPT.format(context=context)
    return prompt, sources


def build_chat_messages(
    system_prompt: str,
    history: list[ChatHistoryMessage],
    query: str,
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    for item in history:
        if item.role in {"assistant", "user"} and item.content.strip():
            messages.append({"role": item.role, "content": item.content})
    messages.append({"role": "user", "content": query})
    return messages


async def create_llm_stream(
    llm_client: AsyncOpenAI,
    messages: list[dict[str, str]],
):
    try:
        return await llm_client.chat.completions.create(
            model=settings.llm_model,
            messages=messages,
            stream=True,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"LLM API request failed: {exc}",
        ) from exc


async def stream_chat_response(stream: Any, sources: list[str]) -> AsyncIterator[str]:
    try:
        async for chunk in stream:
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue

            delta = choices[0].delta.content or ""
            if not delta:
                continue

            payload = json.dumps(
                {"delta": delta, "sources": sources},
                ensure_ascii=False,
            )
            yield f"data: {payload}\n\n"
    except Exception as exc:
        error_payload = json.dumps(
            {"delta": "", "sources": sources, "error": f"LLM stream failed: {exc}"},
            ensure_ascii=False,
        )
        yield f"data: {error_payload}\n\n"
    finally:
        close = getattr(stream, "close", None)
        if callable(close):
            result = close()
            if inspect.isawaitable(result):
                await result

    yield "data: [DONE]\n\n"