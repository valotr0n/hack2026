from fastapi import APIRouter
from pydantic import BaseModel
from ..llm import chat

router = APIRouter()


class SummaryRequest(BaseModel):
    text: str
    style: str = "official"  # official | popular


class SummaryResponse(BaseModel):
    summary: str


@router.post("/summary", response_model=SummaryResponse)
async def generate_summary(req: SummaryRequest) -> SummaryResponse:
    if req.style == "official":
        style_prompt = (
            "Составь краткое официальное резюме (саммари) по предоставленному тексту. "
            "Используй деловой стиль, структурированные абзацы. "
            "Объём: 150-250 слов."
        )
    else:
        style_prompt = (
            "Составь краткий и понятный пересказ текста в популярном стиле. "
            "Пиши просто и доступно, избегай сложных терминов. "
            "Объём: 150-250 слов."
        )

    system = (
        "Ты — ассистент по анализу документов. "
        "Используй ТОЛЬКО информацию из предоставленного текста. "
        "Не добавляй знания из других источников. "
        "Отвечай на русском языке."
    )

    user = f"{style_prompt}\n\nТекст:\n{req.text}"

    summary = await chat(system=system, user=user)
    return SummaryResponse(summary=summary)
