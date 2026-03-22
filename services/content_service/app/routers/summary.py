from __future__ import annotations

import asyncio
import logging
import time

from fastapi import APIRouter
from pydantic import BaseModel
from ..llm import chat

logger = logging.getLogger("content_service.summary")

router = APIRouter()

# Размер одного чанка для первого прохода (символы)
_CHUNK_SIZE = 20_000
# Максимальный размер объединённых промежуточных саммари для финального прохода
_META_CHUNK_SIZE = 40_000


class SummaryRequest(BaseModel):
    text: str
    style: str = "official"  # official | popular


class SummaryResponse(BaseModel):
    summary: str


def _style_prompt(style: str) -> str:
    if style == "official":
        return (
            "Составь краткое официальное резюме (саммари) по предоставленному тексту. "
            "Используй деловой стиль, структурированные абзацы. "
            "Объём: 150-250 слов."
        )
    return (
        "Составь краткий и понятный пересказ текста в популярном стиле. "
        "Пиши просто и доступно, избегай сложных терминов. "
        "Объём: 150-250 слов."
    )


_METADATA_EXCLUSION = (
    "Игнорируй и не упоминай: технические характеристики издания (формат, бумага, тираж, "
    "типография), копирайты и авторские права (© ...), метаданные файлов (timestamps, indd, pdf), "
    "выходные данные книги (ISBN, дата изготовления, издательство), оглавления и списки глав. "
)

_SYSTEM = (
    "Ты — ассистент по анализу документов. "
    "Используй ТОЛЬКО информацию из предоставленного текста. "
    "Не добавляй знания из других источников, не додумывай факты. "
    "Если какой-то аспект не освещён в тексте — не упоминай его. "
    + _METADATA_EXCLUSION +
    "Отвечай на русском языке."
)

_CHUNK_SYSTEM = (
    "Ты — ассистент по анализу документов. "
    "Создай краткое промежуточное резюме ТОЛЬКО по предоставленному фрагменту. "
    "Сохраняй все ключевые факты, цифры и имена точно как в тексте. "
    "Не добавляй интерпретации и выводы которых нет во фрагменте. "
    + _METADATA_EXCLUSION +
    "Отвечай на русском языке."
)

_META_SYSTEM = (
    "Ты — ассистент по синтезу документов. "
    "Перед тобой набор промежуточных резюме разных частей одного большого документа. "
    "Составь единое связное итоговое резюме по всем частям. "
    "Используй только то, что есть в резюме — не добавляй от себя. "
    "Отвечай на русском языке."
)


def _split_text(text: str, chunk_size: int) -> list[str]:
    """Делит текст на части по chunk_size символов, не разрывая слова."""
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end >= len(text):
            chunks.append(text[start:].strip())
            break
        # Ищем ближайший перенос строки или пробел назад
        cut = text.rfind("\n", start, end)
        if cut == -1 or cut <= start:
            cut = text.rfind(" ", start, end)
        if cut == -1 or cut <= start:
            cut = end
        chunks.append(text[start:cut].strip())
        start = cut + 1
    return [c for c in chunks if c]


async def _summarize_chunk(chunk: str, part_label: str, style_prompt: str) -> str:
    user = (
        f"{part_label}\n\n"
        f"Создай промежуточное резюме этого фрагмента. "
        f"Стиль итогового документа: {style_prompt}\n\n"
        f"Фрагмент:\n{chunk}"
    )
    return await chat(system=_CHUNK_SYSTEM, user=user, temperature=0.3)


async def _hierarchical_summary(text: str, style: str) -> str:
    sp = _style_prompt(style)

    if len(text) <= _CHUNK_SIZE:
        logger.info("Summary single-pass chars=%d style=%s", len(text), style)
        return await chat(system=_SYSTEM, user=f"{sp}\n\nТекст:\n{text}", temperature=0.3)

    # Первый проход: параллельная суммаризация каждого чанка
    chunks = _split_text(text, _CHUNK_SIZE)
    n = len(chunks)
    logger.info("Summary map-reduce started chars=%d chunks=%d style=%s", len(text), n, style)
    tasks = [
        _summarize_chunk(chunk, f"Часть {i + 1} из {n}:", sp)
        for i, chunk in enumerate(chunks)
    ]
    partial_summaries: list[str] = await asyncio.gather(*tasks)

    # Объединяем промежуточные саммари
    combined = "\n\n---\n\n".join(
        f"Резюме части {i + 1}:\n{s}" for i, s in enumerate(partial_summaries)
    )

    # Если объединённый текст всё ещё большой — ещё один уровень
    if len(combined) > _META_CHUNK_SIZE:
        meta_chunks = _split_text(combined, _META_CHUNK_SIZE)
        meta_tasks = [
            chat(
                system=_META_SYSTEM,
                user=f"Промежуточные резюме (группа {i + 1} из {len(meta_chunks)}):\n\n{mc}",
                temperature=0.3,
            )
            for i, mc in enumerate(meta_chunks)
        ]
        combined = "\n\n---\n\n".join(await asyncio.gather(*meta_tasks))

    # Финальный проход: синтез
    user_final = (
        f"{sp}\n\n"
        "Ниже — резюме всех частей большого документа. "
        "Составь единое итоговое саммари, устрани повторы, сохрани все ключевые факты.\n\n"
        f"{combined}"
    )
    return await chat(system=_META_SYSTEM, user=user_final, temperature=0.3)


@router.post(
    "/summary",
    response_model=SummaryResponse,
    summary="Саммари документа",
    description="""
Генерирует краткое изложение. Для больших документов (книги, отчёты) автоматически
применяется **иерархическое саммари** (map-reduce):

1. Текст делится на части по 20 000 символов
2. Каждая часть суммаризируется параллельно
3. Промежуточные резюме объединяются в финальное

**Параметр `style`:** `official` (деловой) · `popular` (простым языком)
    """,
)
async def generate_summary(req: SummaryRequest) -> SummaryResponse:
    started_at = time.perf_counter()
    logger.info("Summary request started chars=%d style=%s", len(req.text), req.style)
    summary = await _hierarchical_summary(req.text, req.style)
    logger.info("Summary done %.2fs", time.perf_counter() - started_at)
    return SummaryResponse(summary=summary)
