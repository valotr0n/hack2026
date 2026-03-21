from __future__ import annotations

import io
import json
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from ..llm import chat

router = APIRouter()


class PresentationRequest(BaseModel):
    text: str
    title: str = ""
    style: str = "business"  # business | academic | popular


class PresentationResponse(BaseModel):
    slides: list[dict]  # структура для превью в браузере


# ── JSON структура слайдов ────────────────────────────────────────────────────

async def _generate_slides(text: str, title: str, style: str) -> list[dict]:
    style_prompt = {
        "business": "деловой корпоративный стиль, чёткие тезисы, минимум воды",
        "academic": "академический стиль, подробные объяснения, термины",
        "popular": "простой понятный язык, примеры, аналогии",
    }.get(style, "деловой стиль")

    system = (
        "Ты — эксперт по созданию презентаций. "
        "Структурируй материал в чёткие, ёмкие слайды. "
        "Отвечай строго в формате JSON без лишнего текста."
    )
    user = (
        f"Создай презентацию в {style_prompt} по тексту ниже.\n"
        + (f'Заголовок презентации: "{title}"\n' if title else "")
        + "Верни JSON:\n"
        '{"slides": [\n'
        '  {"type": "title", "title": "...", "subtitle": "..."},\n'
        '  {"type": "content", "title": "Заголовок слайда", "bullets": ["тезис 1", "тезис 2", "тезис 3"]},\n'
        '  {"type": "content", "title": "...", "bullets": [...]},\n'
        '  {"type": "summary", "title": "Итоги", "bullets": ["вывод 1", "вывод 2"]}\n'
        "]}\n\n"
        "Правила:\n"
        "- Первый слайд всегда type=title\n"
        "- Последний слайд всегда type=summary\n"
        "- 5–10 слайдов итого\n"
        "- В каждом content-слайде 3–5 bullets\n"
        "- Bullets — короткие тезисы, не предложения\n\n"
        f"Текст:\n{text}"
    )

    raw = await chat(system=system, user=user)

    try:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        data = json.loads(raw[start:end])
        return data.get("slides", [])
    except Exception:
        raise HTTPException(status_code=500, detail="Не удалось разобрать структуру презентации")


# ── Сборка PPTX ───────────────────────────────────────────────────────────────

def _build_pptx(slides: list[dict]) -> bytes:
    from pptx import Presentation
    from pptx.util import Inches, Pt
    from pptx.dml.color import RGBColor
    from pptx.enum.text import PP_ALIGN

    prs = Presentation()
    prs.slide_width = Inches(13.33)
    prs.slide_height = Inches(7.5)

    # Цвета в стиле банка
    COLOR_BG = RGBColor(0x00, 0x3A, 0x70)       # тёмно-синий
    COLOR_ACCENT = RGBColor(0xFF, 0xB8, 0x00)    # золотой
    COLOR_WHITE = RGBColor(0xFF, 0xFF, 0xFF)
    COLOR_LIGHT = RGBColor(0xF0, 0xF4, 0xF8)
    COLOR_TEXT = RGBColor(0x1A, 0x1A, 0x2E)

    def _set_bg(slide, color: RGBColor) -> None:
        from pptx.util import Emu
        fill = slide.background.fill
        fill.solid()
        fill.fore_color.rgb = color

    def _add_textbox(slide, text: str, left, top, width, height,
                     font_size: int, bold: bool = False,
                     color: RGBColor = COLOR_WHITE,
                     align=PP_ALIGN.LEFT) -> None:
        txBox = slide.shapes.add_textbox(left, top, width, height)
        tf = txBox.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.alignment = align
        run = p.add_run()
        run.text = text
        run.font.size = Pt(font_size)
        run.font.bold = bold
        run.font.color.rgb = color

    for slide_data in slides:
        slide_type = slide_data.get("type", "content")
        title_text = slide_data.get("title", "")

        if slide_type == "title":
            layout = prs.slide_layouts[6]  # blank
            slide = prs.slides.add_slide(layout)
            _set_bg(slide, COLOR_BG)

            # Акцентная полоса слева
            from pptx.util import Emu
            bar = slide.shapes.add_shape(
                1,  # MSO_SHAPE_TYPE.RECTANGLE
                Inches(0), Inches(0),
                Inches(0.3), Inches(7.5),
            )
            bar.fill.solid()
            bar.fill.fore_color.rgb = COLOR_ACCENT
            bar.line.fill.background()

            _add_textbox(slide, title_text,
                         Inches(0.7), Inches(2.2), Inches(11.5), Inches(1.8),
                         font_size=40, bold=True, color=COLOR_WHITE,
                         align=PP_ALIGN.LEFT)

            subtitle = slide_data.get("subtitle", "")
            if subtitle:
                _add_textbox(slide, subtitle,
                             Inches(0.7), Inches(4.2), Inches(11.5), Inches(1.0),
                             font_size=22, bold=False, color=COLOR_ACCENT,
                             align=PP_ALIGN.LEFT)

        else:  # content / summary
            layout = prs.slide_layouts[6]
            slide = prs.slides.add_slide(layout)
            _set_bg(slide, COLOR_LIGHT)

            # Заголовочная полоса
            header = slide.shapes.add_shape(
                1,
                Inches(0), Inches(0),
                Inches(13.33), Inches(1.3),
            )
            header.fill.solid()
            header.fill.fore_color.rgb = COLOR_BG
            header.line.fill.background()

            _add_textbox(slide, title_text,
                         Inches(0.4), Inches(0.1), Inches(12.5), Inches(1.1),
                         font_size=28, bold=True, color=COLOR_WHITE)

            bullets = slide_data.get("bullets", [])
            bullet_text = "\n".join(f"• {b}" for b in bullets)
            _add_textbox(slide, bullet_text,
                         Inches(0.6), Inches(1.5), Inches(12.0), Inches(5.5),
                         font_size=20, bold=False, color=COLOR_TEXT)

    buf = io.BytesIO()
    prs.save(buf)
    buf.seek(0)
    return buf.read()


# ── Эндпоинты ─────────────────────────────────────────────────────────────────

@router.post(
    "/presentation/preview",
    response_model=PresentationResponse,
    summary="Структура презентации (превью)",
    description="""
Генерирует структуру слайдов в JSON — для отображения превью в браузере.

**Параметр `style`:** `business` · `academic` · `popular`

**Ответ:**
```json
{
  "slides": [
    {"type": "title", "title": "Анализ кредитного портфеля", "subtitle": "Q1 2026"},
    {"type": "content", "title": "Ключевые показатели", "bullets": ["NPL 2.3%", "ROE 18%"]},
    {"type": "summary", "title": "Итоги", "bullets": ["Портфель вырос на 12%"]}
  ]
}
```
    """,
)
async def presentation_preview(req: PresentationRequest) -> PresentationResponse:
    slides = await _generate_slides(req.text, req.title, req.style)
    return PresentationResponse(slides=slides)


@router.post(
    "/presentation/download",
    summary="Скачать презентацию (PPTX)",
    description="""
Генерирует и возвращает готовый файл `.pptx` для скачивания.

**Параметр `style`:** `business` · `academic` · `popular`

Возвращает бинарный файл `presentation.pptx`.
    """,
)
async def presentation_download(req: PresentationRequest) -> StreamingResponse:
    slides = await _generate_slides(req.text, req.title, req.style)
    pptx_bytes = _build_pptx(slides)
    filename = "presentation.pptx"
    return StreamingResponse(
        io.BytesIO(pptx_bytes),
        media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
