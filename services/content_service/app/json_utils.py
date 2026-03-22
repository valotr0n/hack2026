from __future__ import annotations

import ast
import json
import re
from collections import Counter

_STOPWORDS = {
    "и", "в", "во", "не", "что", "он", "на", "я", "с", "со", "как", "а", "то", "все",
    "она", "так", "его", "но", "да", "ты", "к", "у", "же", "вы", "за", "бы", "по",
    "ее", "мне", "было", "вот", "от", "меня", "еще", "нет", "о", "из", "ему", "теперь",
    "когда", "даже", "ну", "вдруг", "ли", "если", "уже", "или", "ни", "быть", "был",
    "него", "до", "вас", "нибудь", "опять", "уж", "вам", "ведь", "там", "потом", "себя",
    "ничего", "ей", "может", "они", "тут", "где", "есть", "надо", "ней", "для", "мы",
    "тебя", "их", "чем", "была", "сам", "чтоб", "без", "будто", "чего", "раз", "тоже",
    "себе", "под", "будет", "ж", "тогда", "кто", "этот", "того", "потому", "этого",
    "какой", "совсем", "ним", "здесь", "этом", "один", "почти", "мой", "тем", "чтобы",
    "нее", "сейчас", "были", "куда", "зачем", "всех", "никогда", "можно", "при", "наконец",
    "два", "об", "другой", "хоть", "после", "над", "больше", "тот", "через", "эти",
    "нас", "про", "всего", "них", "какая", "много", "разве", "три", "эту", "моя", "впрочем",
    "хорошо", "свою", "этой", "перед", "иногда", "лучше", "чуть", "том", "нельзя", "такой",
    "им", "более", "всегда", "конечно", "всю", "между",
}


def strip_code_fences(raw: str) -> str:
    return re.sub(
        r"^\s*```(?:json)?\s*|\s*```\s*$",
        "",
        raw.strip(),
        flags=re.IGNORECASE | re.DOTALL,
    ).strip()


def parse_json_payload(raw: str) -> object | None:
    cleaned = strip_code_fences(raw)
    if not cleaned:
        return None

    candidates: list[str] = [cleaned]
    for open_char, close_char in (("{", "}"), ("[", "]")):
        start = cleaned.find(open_char)
        end = cleaned.rfind(close_char) + 1
        if start >= 0 and end > start:
            candidates.append(cleaned[start:end])

    for candidate in candidates:
        for loader in (json.loads, ast.literal_eval):
            try:
                return loader(candidate)
            except Exception:
                continue
    return None


def split_sentences(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", text or "").strip()
    if not normalized:
        return []
    return [
        chunk.strip()
        for chunk in re.split(r"(?<=[.!?])\s+", normalized)
        if chunk.strip()
    ]


def candidate_sentences(text: str, *, min_length: int = 24, max_items: int = 12) -> list[str]:
    items: list[str] = []
    seen: set[str] = set()
    for sentence in split_sentences(text):
        normalized = sentence.lower()
        if len(sentence) < min_length or normalized in seen:
            continue
        seen.add(normalized)
        items.append(sentence[:320])
        if len(items) >= max_items:
            break
    return items


def top_keywords(text: str, *, limit: int = 8) -> list[str]:
    words = re.findall(r"[A-Za-zА-Яа-яЁё][A-Za-zА-Яа-яЁё0-9-]{3,}", text or "")
    counts = Counter(
        word.lower()
        for word in words
        if word.lower() not in _STOPWORDS
    )
    return [word for word, _ in counts.most_common(limit)]
