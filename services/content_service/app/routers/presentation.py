from __future__ import annotations

import ast
import io
import json
import logging
import re
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from ..config import settings
from ..llm import chat

router = APIRouter()
logger = logging.getLogger(__name__)


class PresentationRequest(BaseModel):
    text: str = ""
    title: str = ""
    style: str = "business"  # business | academic | popular
    slides: list[dict] | None = None
    domain: str = "general"
    doc_types: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


class PresentationResponse(BaseModel):
    slides: list[dict]  # структура для превью в браузере
    domain: str = "general"


# ── JSON структура слайдов ────────────────────────────────────────────────────

_DOMAIN_PROFILES = {
    "general": {
        "label": "универсальный материал",
        "subtitle": "Контекст, ключевые идеи и практический вывод",
        "summary_title": "Итог и следующий шаг",
        "fallback_titles": [
            "Контекст и предмет",
            "Ключевые идеи",
            "Сигналы и ограничения",
            "Что делать дальше",
        ],
        "flow": "контекст → ключевые тезисы → сигналы/ограничения → вывод",
        "bg": (0x00, 0x3A, 0x70),
        "accent": (0xFF, 0xB8, 0x00),
        "light": (0xF0, 0xF4, 0xF8),
        "text": (0x1A, 0x1A, 0x2E),
        "muted": (0xD9, 0xE2, 0xEC),
    },
    "legal": {
        "label": "юридический материал",
        "subtitle": "Стороны, обязательства, сроки и риски",
        "summary_title": "Юридические выводы и next steps",
        "fallback_titles": [
            "Суть документа и стороны",
            "Ключевые условия и сроки",
            "Риски, ограничения и спорные зоны",
            "Решения и действия",
        ],
        "flow": "предмет и стороны → условия и сроки → риски/неясности → действия",
        "bg": (0x2A, 0x2F, 0x45),
        "accent": (0xC1, 0x7B, 0x42),
        "light": (0xF7, 0xF9, 0xFC),
        "text": (0x24, 0x28, 0x35),
        "muted": (0xD7, 0xD9, 0xE2),
    },
    "finance": {
        "label": "финансовый материал",
        "subtitle": "Метрики, динамика, факторы и решения",
        "summary_title": "Финансовый вывод и решение",
        "fallback_titles": [
            "Картина показателей",
            "Драйверы и отклонения",
            "Риски для денег и результата",
            "Решение и приоритеты",
        ],
        "flow": "метрики → драйверы → риски → решение",
        "bg": (0x0B, 0x39, 0x5B),
        "accent": (0x18, 0xA9, 0x57),
        "light": (0xF2, 0xF8, 0xF5),
        "text": (0x18, 0x2B, 0x33),
        "muted": (0xD3, 0xE8, 0xD9),
    },
    "research": {
        "label": "исследовательский материал",
        "subtitle": "Проблема, метод, результаты и ограничения",
        "summary_title": "Научный вывод и ограничения",
        "fallback_titles": [
            "Проблема и постановка",
            "Подход и метод",
            "Ключевые результаты",
            "Ограничения и вывод",
        ],
        "flow": "проблема → метод → результаты → ограничения и вывод",
        "bg": (0x1F, 0x2A, 0x44),
        "accent": (0x2F, 0x80, 0xED),
        "light": (0xF7, 0xF9, 0xFC),
        "text": (0x21, 0x2B, 0x36),
        "muted": (0xC9, 0xD6, 0xE3),
    },
    "operations": {
        "label": "операционный материал",
        "subtitle": "Процесс, роли, контрольные точки и сбои",
        "summary_title": "Операционный вывод и действия",
        "fallback_titles": [
            "Цель и рамки процесса",
            "Как устроен поток работ",
            "Узкие места и контроль",
            "Что менять дальше",
        ],
        "flow": "цель → процесс → узкие места → действия",
        "bg": (0x3F, 0x4A, 0x5A),
        "accent": (0xF2, 0x8C, 0x28),
        "light": (0xF7, 0xF6, 0xF3),
        "text": (0x2B, 0x32, 0x3A),
        "muted": (0xDF, 0xD8, 0xCF),
    },
    "briefing": {
        "label": "информационный брифинг",
        "subtitle": "Контекст, событие, влияние и дальнейшие шаги",
        "summary_title": "Что это значит дальше",
        "fallback_titles": [
            "Что произошло",
            "Контекст и участники",
            "Последствия и риски",
            "Что отслеживать дальше",
        ],
        "flow": "событие → контекст → влияние → что дальше",
        "bg": (0x5E, 0x2A, 0x1F),
        "accent": (0xE2, 0xA4, 0x58),
        "light": (0xFC, 0xF7, 0xF2),
        "text": (0x32, 0x26, 0x22),
        "muted": (0xE7, 0xD6, 0xC8),
    },
    "education": {
        "label": "учебный материал",
        "subtitle": "Ключевые понятия, логика и примеры применения",
        "summary_title": "Главное к запоминанию",
        "fallback_titles": [
            "Ключевая идея",
            "Как это работает",
            "Пример и применение",
            "Что нужно запомнить",
        ],
        "flow": "идея → механизм → пример → закрепление",
        "bg": (0x0C, 0x5B, 0x63),
        "accent": (0xF3, 0xC9, 0x44),
        "light": (0xF1, 0xFA, 0xF8),
        "text": (0x1A, 0x30, 0x32),
        "muted": (0xD6, 0xEA, 0xE5),
    },
}

_TONE_PROMPTS = {
    "business": "деловой и управленческий тон: выводы короткие, формулировки строгие, акцент на решениях",
    "academic": "аналитический тон: допускаются термины, нужна причинно-следственная логика и аккуратные формулировки",
    "popular": "понятный живой тон: меньше канцелярита, больше ясности и объяснений простыми словами",
}

_DOMAIN_KEYWORDS: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = [
    ("legal", ("договор", "судебный документ"), ("договор", "соглаш", "иск", "арбитраж", "суд", "неустой", "обязатель", "право")),
    ("finance", ("отчёт",), ("банк", "кредит", "заем", "ставк", "выручк", "прибыл", "ebitda", "денежн", "бюджет", "финанс")),
    ("research", ("научная статья",), ("исслед", "метод", "гипотез", "выборк", "эксперимент", "результат", "науч", "статист")),
    ("operations", ("инструкция",), ("инструк", "регламент", "процесс", "процедур", "этап", "чек-лист", "внедрен", "sla")),
    ("briefing", ("новость", "письмо"), ("новост", "обновлен", "анонс", "письмо", "меморандум", "пресс", "релиз")),
    ("education", ("книга",), ("курс", "лекц", "глава", "учеб", "объяснен", "пример", "понят")),
]

_GENERIC_SLIDE_TITLES = {
    "",
    "ключевые тезисы",
    "основная информация",
    "общая информация",
    "главное",
    "детали",
    "содержание",
    "анализ",
    "выводы",
    "итоги",
}


def _fit_bullet_font_size(bullets: list[str]) -> int:
    if not bullets:
        return 20

    longest = max(len(item) for item in bullets)
    total = sum(len(item) for item in bullets)
    if len(bullets) >= 5 or longest > 120 or total > 360:
        return 16
    if len(bullets) >= 4 or longest > 90 or total > 260:
        return 18
    if longest > 60 or total > 180:
        return 20
    return 22


def _resolve_domain(domain: str, doc_types: list[str], tags: list[str], title: str, text: str = "") -> str:
    if domain in _DOMAIN_PROFILES:
        return domain

    lookup_text = " ".join([title, *doc_types, *tags, text[:1200]]).lower()
    for candidate, matched_doc_types, keywords in _DOMAIN_KEYWORDS:
        if any(doc_type in matched_doc_types for doc_type in doc_types):
            return candidate
        if any(keyword in lookup_text for keyword in keywords):
            return candidate
    return "general"


def _domain_profile(domain: str) -> dict:
    return _DOMAIN_PROFILES.get(domain, _DOMAIN_PROFILES["general"])


def _default_subtitle(domain: str, tags: list[str]) -> str:
    profile = _domain_profile(domain)
    if tags:
        return " · ".join(tags[:3])
    return profile["subtitle"]


def _fallback_slide_title(domain: str, index: int, slide_type: str) -> str:
    profile = _domain_profile(domain)
    if slide_type == "summary":
        return profile["summary_title"]
    titles = profile["fallback_titles"]
    if not titles:
        return "Ключевой вывод"
    return titles[min(index, len(titles) - 1)]


def _is_generic_title(title: str) -> bool:
    normalized = title.strip().lower()
    return normalized in _GENERIC_SLIDE_TITLES


def _strip_code_fences(raw: str) -> str:
    return re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", raw.strip(), flags=re.IGNORECASE | re.DOTALL).strip()


def _parse_presentation_payload(raw: str) -> list[dict] | None:
    cleaned = _strip_code_fences(raw)
    if not cleaned:
        return None

    candidates: list[str] = [cleaned]
    start = cleaned.find("{")
    end = cleaned.rfind("}") + 1
    if start >= 0 and end > start:
        candidates.append(cleaned[start:end])

    for candidate in candidates:
        for loader in (json.loads, ast.literal_eval):
            try:
                data = loader(candidate)
            except Exception:
                continue

            if isinstance(data, dict):
                slides = data.get("slides")
                if isinstance(slides, list):
                    return slides
            if isinstance(data, list):
                return data
    return None


def _extract_candidate_bullets(raw: str) -> list[str]:
    cleaned = _strip_code_fences(raw)
    if not cleaned:
        return []

    candidates: list[str] = []
    for line in cleaned.splitlines():
        value = line.strip().strip(",")
        if not value or value in {"{", "}", "[", "]"}:
            continue
        if value.lower().startswith(("slides", '"slides"', "'slides'")):
            continue
        value = re.sub(r'^[\-\*\u2022\d\.\)\(\"\']+\s*', "", value)
        value = value.strip(" \"'")
        if len(value) < 12:
            continue
        if ":" in value and len(value) < 28:
            continue
        candidates.append(value)

    if len(candidates) < 4:
        sentence_candidates = [
            chunk.strip()
            for chunk in re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", cleaned))
            if len(chunk.strip()) >= 20
        ]
        candidates.extend(sentence_candidates)

    unique: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        normalized = candidate.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        unique.append(candidate[:220])
    return unique


def _fallback_slides_from_text(
    text: str,
    title: str,
    domain: str,
    tags: list[str],
    raw: str = "",
) -> list[dict]:
    bullets = _extract_candidate_bullets(raw)
    if len(bullets) < 6:
        bullets.extend(_extract_candidate_bullets(text))

    if not bullets:
        bullets = [
            "Материал загружен, но автоматическая структура получилась неполной.",
            "Проверьте полноту исходного текста и повторите генерацию при необходимости.",
            "Ключевые выводы лучше формировать после очистки или детализации источников.",
        ]

    content_bullets = bullets[:12]
    slides: list[dict] = [
        {
            "type": "title",
            "title": title.strip() or "Презентация",
            "subtitle": _default_subtitle(domain, tags),
        }
    ]

    for chunk_index, start in enumerate(range(0, len(content_bullets), 3)):
        chunk = content_bullets[start:start + 3]
        if not chunk:
            continue
        slides.append(
            {
                "type": "content",
                "title": _fallback_slide_title(domain, chunk_index, "content"),
                "bullets": chunk,
            }
        )
        if len(slides) >= 5:
            break

    summary_bullets = bullets[:2] if len(bullets) >= 2 else content_bullets[:2]
    if not summary_bullets:
        summary_bullets = ["Ключевые тезисы требуют ручной проверки."]
    slides.append(
        {
            "type": "summary",
            "title": _fallback_slide_title(domain, len(slides), "summary"),
            "bullets": summary_bullets[:2],
        }
    )

    return _normalize_slides(slides, title, domain, tags)


async def _generate_slides(
    text: str,
    title: str,
    style: str,
    domain: str,
    doc_types: list[str],
    tags: list[str],
) -> tuple[list[dict], str]:
    resolved_domain = _resolve_domain(domain, doc_types, tags, title, text)
    profile = _domain_profile(resolved_domain)
    tone_prompt = _TONE_PROMPTS.get(style, _TONE_PROMPTS["business"])
    doc_types_prompt = ", ".join(doc_types[:3]) if doc_types else "не определены"
    tags_prompt = ", ".join(tags[:6]) if tags else "не определены"

    system = (
        "Ты — сильный редактор презентаций для руководителей и клиентов. "
        "Собирай не конспект документа, а осмысленный story-driven deck. "
        "Отвечай строго в формате JSON без лишнего текста."
    )
    user = (
        f"Подготовь презентацию по материалу домена «{profile['label']}».\n"
        f"Тон подачи: {tone_prompt}.\n"
        f"Рекомендуемый ход презентации: {profile['flow']}.\n"
        f"Опознанные типы источников: {doc_types_prompt}.\n"
        f"Тематические теги: {tags_prompt}.\n"
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
        "- 5–8 слайдов итого\n"
        "- В каждом content-слайде 3–4 bullets\n"
        "- Это не пересказ по абзацам: собирай управленческую историю\n"
        "- Заголовок каждого content-слайда должен быть выводом или смыслом, а не общим словом\n"
        "- Избегай названий вроде 'Основная информация', 'Ключевые тезисы', 'Анализ'\n"
        "- В bullets используй конкретику: роли, суммы, сроки, факты, последствия, решения\n"
        "- Если в материале есть пробелы или неоднозначность, выдели это отдельным слайдом или bullet\n"
        "- Для домена держи фокус на этом: "
        f"{profile['subtitle']}\n\n"
        f"Текст:\n{text}"
    )

    try:
        raw = await chat(
            system=system,
            user=user,
            temperature=0.2,
            max_tokens=settings.presentation_max_tokens,
        )
    except Exception as exc:
        logger.warning(
            "Presentation generation failed before parsing: domain=%s style=%s error=%s",
            resolved_domain,
            style,
            exc,
        )
        return _fallback_slides_from_text(text, title, resolved_domain, tags), resolved_domain

    parsed_slides = _parse_presentation_payload(raw)
    if parsed_slides is None:
        logger.warning(
            "Presentation parse failed: domain=%s style=%s raw_sample=%r",
            resolved_domain,
            style,
            raw[:500],
        )
        return _fallback_slides_from_text(text, title, resolved_domain, tags, raw=raw), resolved_domain

    return _normalize_slides(parsed_slides, title, resolved_domain, tags), resolved_domain


def _normalize_slides(slides: list[dict], title: str, domain: str, tags: list[str]) -> list[dict]:
    normalized: list[dict] = []
    safe_title = title.strip() or "Презентация"
    profile = _domain_profile(domain)

    for idx, slide in enumerate(slides):
        if not isinstance(slide, dict):
            continue

        slide_type = slide.get("type", "content")
        if idx == 0:
            slide_type = "title"
        elif slide_type not in {"content", "summary"}:
            slide_type = "content"

        if slide_type == "title":
            normalized.append(
                {
                    "type": "title",
                    "title": str(slide.get("title") or safe_title).strip() or safe_title,
                    "subtitle": str(slide.get("subtitle") or _default_subtitle(domain, tags)).strip(),
                }
            )
            continue

        bullets = [
            str(item).strip()
            for item in (slide.get("bullets") or [])
            if str(item).strip()
        ][:5]
        if not bullets:
            continue

        normalized.append(
            {
                "type": "summary" if slide_type == "summary" else "content",
                "title": str(slide.get("title") or "").strip() or _fallback_slide_title(domain, len(normalized), slide_type),
                "bullets": bullets,
            }
        )

    if not normalized:
        return [
            {"type": "title", "title": safe_title, "subtitle": _default_subtitle(domain, tags)},
            {"type": "summary", "title": profile["summary_title"], "bullets": ["Недостаточно данных для генерации структуры"]},
        ]

    normalized[0]["type"] = "title"
    normalized[0].setdefault("title", safe_title)
    normalized[0]["subtitle"] = str(normalized[0].get("subtitle") or _default_subtitle(domain, tags)).strip()

    if len(normalized) == 1:
        normalized.append(
            {
                "type": "summary",
                "title": profile["summary_title"],
                "bullets": ["Основные тезисы вынесены на титульный слайд"],
            }
        )
    else:
        normalized[-1]["type"] = "summary"
        normalized[-1]["title"] = (
            str(normalized[-1].get("title") or "").strip()
            if not _is_generic_title(str(normalized[-1].get("title") or ""))
            else profile["summary_title"]
        ) or profile["summary_title"]

    for idx, slide in enumerate(normalized[1:], start=0):
        if _is_generic_title(str(slide.get("title") or "")):
            slide["title"] = _fallback_slide_title(domain, idx, slide.get("type", "content"))

    return normalized


async def _resolve_slides(req: PresentationRequest) -> tuple[list[dict], str]:
    resolved_domain = _resolve_domain(req.domain, req.doc_types, req.tags, req.title, req.text)
    if req.slides is not None:
        return _normalize_slides(req.slides, req.title, resolved_domain, req.tags), resolved_domain
    return await _generate_slides(req.text, req.title, req.style, resolved_domain, req.doc_types, req.tags)


# ── Сборка PPTX ───────────────────────────────────────────────────────────────

def _build_pptx(
    slides: list[dict],
    style: str = "business",
    domain: str = "general",
    tags: list[str] | None = None,
) -> bytes:
    from pptx import Presentation
    from pptx.util import Inches, Pt
    from pptx.dml.color import RGBColor
    from pptx.enum.text import PP_ALIGN

    prs = Presentation()
    prs.slide_width = Inches(13.33)
    prs.slide_height = Inches(7.5)

    profile = _domain_profile(domain)
    theme = profile

    # Цветовая тема
    COLOR_BG = RGBColor(*theme["bg"])
    COLOR_ACCENT = RGBColor(*theme["accent"])
    COLOR_WHITE = RGBColor(0xFF, 0xFF, 0xFF)
    COLOR_LIGHT = RGBColor(*theme["light"])
    COLOR_TEXT = RGBColor(*theme["text"])
    COLOR_MUTED = RGBColor(*theme["muted"])

    def _set_bg(slide, color: RGBColor) -> None:
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

    for index, slide_data in enumerate(slides, start=1):
        slide_type = slide_data.get("type", "content")
        title_text = slide_data.get("title", "")

        if slide_type == "title":
            layout = prs.slide_layouts[6]  # blank
            slide = prs.slides.add_slide(layout)
            _set_bg(slide, COLOR_BG)

            # Акцентная полоса слева
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

            badge = f"{profile['label']} · {style}"
            _add_textbox(slide, badge,
                         Inches(8.6), Inches(6.55), Inches(3.8), Inches(0.35),
                         font_size=12, bold=False, color=COLOR_MUTED,
                         align=PP_ALIGN.RIGHT)

            if tags:
                _add_textbox(slide, " · ".join(tags[:3]),
                             Inches(6.9), Inches(6.95), Inches(5.5), Inches(0.3),
                             font_size=11, bold=False, color=COLOR_MUTED,
                             align=PP_ALIGN.RIGHT)

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
            bullet_font_size = _fit_bullet_font_size(bullets)
            _add_textbox(slide, bullet_text,
                         Inches(0.6), Inches(1.5), Inches(12.0), Inches(5.5),
                         font_size=bullet_font_size, bold=False, color=COLOR_TEXT)

            footer = slide.shapes.add_shape(
                1,
                Inches(0), Inches(7.05),
                Inches(13.33), Inches(0.45),
            )
            footer.fill.solid()
            footer.fill.fore_color.rgb = COLOR_MUTED
            footer.line.fill.background()

            _add_textbox(slide, f"{index}/{len(slides)}",
                         Inches(12.0), Inches(7.07), Inches(0.9), Inches(0.25),
                         font_size=11, bold=True, color=COLOR_BG,
                         align=PP_ALIGN.RIGHT)

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
Определяет тон подачи. Домен и визуальная тема определяются автоматически по содержанию.

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
    slides, domain = await _resolve_slides(req)
    return PresentationResponse(slides=slides, domain=domain)


@router.post(
    "/presentation/download",
    summary="Скачать презентацию (PPTX)",
    description="""
Генерирует и возвращает готовый файл `.pptx` для скачивания.

**Параметр `style`:** `business` · `academic` · `popular`
Определяет тон подачи. Домен и визуальная тема определяются автоматически по содержанию.

Возвращает бинарный файл `presentation.pptx`.
    """,
)
async def presentation_download(req: PresentationRequest) -> StreamingResponse:
    slides, domain = await _resolve_slides(req)
    pptx_bytes = _build_pptx(slides, style=req.style, domain=domain, tags=req.tags)
    filename = "presentation.pptx"
    return StreamingResponse(
        io.BytesIO(pptx_bytes),
        media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
