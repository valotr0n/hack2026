from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

import aiosqlite

_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS users (
    id            TEXT PRIMARY KEY,
    username      TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS notebooks (
    id         TEXT PRIMARY KEY,
    user_id    TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title      TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sources (
    id           TEXT PRIMARY KEY,
    notebook_id  TEXT NOT NULL REFERENCES notebooks(id) ON DELETE CASCADE,
    filename     TEXT NOT NULL,
    chunks_count INTEGER NOT NULL DEFAULT 0,
    created_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chat_messages (
    id          TEXT PRIMARY KEY,
    notebook_id TEXT NOT NULL REFERENCES notebooks(id) ON DELETE CASCADE,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    sources     TEXT NOT NULL DEFAULT '[]',
    created_at  TEXT NOT NULL
);
"""


async def init_db(path: str) -> None:
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    async with aiosqlite.connect(path, timeout=30) as db:
        await db.execute("PRAGMA busy_timeout = 10000")
        await db.executescript(_SCHEMA)
        # Миграции — добавляем колонки для кэша контента если их ещё нет
        # Миграции sources
        for col, typedef in [
            ("status", "TEXT DEFAULT 'ready'"),
            ("error", "TEXT"),
        ]:
            try:
                await db.execute(f"ALTER TABLE sources ADD COLUMN {col} {typedef}")
            except Exception:
                pass
        # Миграции notebooks
        for col, typedef in [
            ("summary", "TEXT"),
            ("mindmap", "TEXT"),
            ("flashcards", "TEXT"),
            ("podcast_url", "TEXT"),
            ("podcast_script", "TEXT"),
            ("contract", "TEXT"),
            ("knowledge_graph", "TEXT"),
            ("timeline", "TEXT"),
            ("questions", "TEXT"),
            ("presentation", "TEXT"),
            ("contour", "TEXT DEFAULT 'open'"),
        ]:
            try:
                await db.execute(f"ALTER TABLE notebooks ADD COLUMN {col} {typedef}")
            except Exception:
                pass
        await db.commit()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id() -> str:
    return str(uuid.uuid4())


# ── Users ─────────────────────────────────────────────────────────────────────

async def create_user(path: str, username: str, password_hash: str) -> dict[str, Any]:
    uid = _new_id()
    now = _now()
    async with aiosqlite.connect(path, timeout=30) as db:
        await db.execute(
            "INSERT INTO users (id, username, password_hash, created_at) VALUES (?, ?, ?, ?)",
            (uid, username, password_hash, now),
        )
        await db.commit()
    return {"id": uid, "username": username, "created_at": now}


async def get_user_by_username(path: str, username: str) -> dict[str, Any] | None:
    async with aiosqlite.connect(path, timeout=30) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT id, username, password_hash, created_at FROM users WHERE username = ?",
            (username,),
        ) as cur:
            row = await cur.fetchone()
            return dict(row) if row else None


async def get_user_by_id(path: str, user_id: str) -> dict[str, Any] | None:
    async with aiosqlite.connect(path, timeout=30) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT id, username, created_at FROM users WHERE id = ?",
            (user_id,),
        ) as cur:
            row = await cur.fetchone()
            return dict(row) if row else None


# ── Notebooks ─────────────────────────────────────────────────────────────────

async def create_notebook(path: str, user_id: str, title: str, contour: str = "open") -> dict[str, Any]:
    nid = _new_id()
    now = _now()
    contour = contour if contour in ("open", "closed") else "open"
    async with aiosqlite.connect(path, timeout=30) as db:
        await db.execute(
            "INSERT INTO notebooks (id, user_id, title, created_at, contour) VALUES (?, ?, ?, ?, ?)",
            (nid, user_id, title, now, contour),
        )
        await db.commit()
    return {"id": nid, "user_id": user_id, "title": title, "created_at": now, "contour": contour}


async def update_notebook_contour(path: str, notebook_id: str, contour: str) -> None:
    contour = contour if contour in ("open", "closed") else "open"
    async with aiosqlite.connect(path, timeout=30) as db:
        await db.execute(
            "UPDATE notebooks SET contour = ? WHERE id = ?",
            (contour, notebook_id),
        )
        await db.commit()


async def list_notebooks(path: str, user_id: str) -> list[dict[str, Any]]:
    async with aiosqlite.connect(path, timeout=30) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT id, title, created_at FROM notebooks WHERE user_id = ? ORDER BY created_at DESC",
            (user_id,),
        ) as cur:
            return [dict(r) for r in await cur.fetchall()]


async def get_notebook(path: str, notebook_id: str) -> dict[str, Any] | None:
    async with aiosqlite.connect(path, timeout=30) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT id, user_id, title, created_at, contour, summary, mindmap, flashcards, podcast_url, podcast_script, contract, knowledge_graph, timeline, questions, presentation FROM notebooks WHERE id = ?",
            (notebook_id,),
        ) as cur:
            row = await cur.fetchone()
            return dict(row) if row else None


async def save_notebook_content(path: str, notebook_id: str, field: str, value: str) -> None:
    allowed = {"summary", "mindmap", "flashcards", "podcast_url", "podcast_script", "contract", "knowledge_graph", "timeline", "questions", "presentation"}
    if field not in allowed:
        raise ValueError(f"Unknown field: {field}")
    async with aiosqlite.connect(path, timeout=30) as db:
        await db.execute(
            f"UPDATE notebooks SET {field} = ? WHERE id = ?",
            (value, notebook_id),
        )
        await db.commit()


async def update_notebook_title(path: str, notebook_id: str, title: str) -> None:
    async with aiosqlite.connect(path, timeout=30) as db:
        await db.execute(
            "UPDATE notebooks SET title = ? WHERE id = ?",
            (title, notebook_id),
        )
        await db.commit()


async def delete_notebook(path: str, notebook_id: str) -> None:
    async with aiosqlite.connect(path, timeout=30) as db:
        await db.execute("DELETE FROM notebooks WHERE id = ?", (notebook_id,))
        await db.commit()


# ── Sources ───────────────────────────────────────────────────────────────────

async def create_source(
    path: str,
    notebook_id: str,
    filename: str,
    chunks_count: int = 0,
    status: str = "processing",
    source_id: str | None = None,
) -> dict[str, Any]:
    sid = source_id or _new_id()
    now = _now()
    async with aiosqlite.connect(path, timeout=30) as db:
        await db.execute(
            "INSERT INTO sources (id, notebook_id, filename, chunks_count, created_at, status) VALUES (?, ?, ?, ?, ?, ?)",
            (sid, notebook_id, filename, chunks_count, now, status),
        )
        await db.commit()
    return {
        "id": sid,
        "notebook_id": notebook_id,
        "filename": filename,
        "chunks_count": chunks_count,
        "created_at": now,
        "status": status,
        "error": None,
    }


async def update_source_status(
    path: str,
    source_id: str,
    status: str,
    chunks_count: int = 0,
    error: str | None = None,
) -> None:
    async with aiosqlite.connect(path, timeout=30) as db:
        await db.execute(
            "UPDATE sources SET status = ?, chunks_count = ?, error = ? WHERE id = ?",
            (status, chunks_count, error, source_id),
        )
        await db.commit()


async def list_sources(path: str, notebook_id: str) -> list[dict[str, Any]]:
    async with aiosqlite.connect(path, timeout=30) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT id, filename, chunks_count, created_at, status, error FROM sources WHERE notebook_id = ? ORDER BY created_at ASC",
            (notebook_id,),
        ) as cur:
            return [dict(r) for r in await cur.fetchall()]


async def get_source(path: str, source_id: str) -> dict[str, Any] | None:
    async with aiosqlite.connect(path, timeout=30) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT id, notebook_id, filename, chunks_count, created_at, status, error FROM sources WHERE id = ?",
            (source_id,),
        ) as cur:
            row = await cur.fetchone()
            return dict(row) if row else None


async def delete_source(path: str, source_id: str) -> None:
    async with aiosqlite.connect(path, timeout=30) as db:
        await db.execute("DELETE FROM sources WHERE id = ?", (source_id,))
        await db.commit()


async def save_chat_message(
    path: str,
    notebook_id: str,
    role: str,
    content: str,
    sources: list[str] | None = None,
) -> dict[str, Any]:
    import json
    mid = _new_id()
    now = _now()
    sources_json = json.dumps(sources or [], ensure_ascii=False)
    async with aiosqlite.connect(path, timeout=30) as db:
        await db.execute(
            "INSERT INTO chat_messages (id, notebook_id, role, content, sources, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (mid, notebook_id, role, content, sources_json, now),
        )
        await db.commit()
    return {"id": mid, "notebook_id": notebook_id, "role": role, "content": content, "sources": sources or [], "created_at": now}


async def get_chat_history(path: str, notebook_id: str) -> list[dict[str, Any]]:
    import json
    async with aiosqlite.connect(path, timeout=30) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT id, role, content, sources, created_at FROM chat_messages WHERE notebook_id = ? ORDER BY created_at ASC",
            (notebook_id,),
        ) as cur:
            rows = await cur.fetchall()
    result = []
    for row in rows:
        d = dict(row)
        try:
            d["sources"] = json.loads(d["sources"])
        except Exception:
            d["sources"] = []
        result.append(d)
    return result


async def clear_chat_history(path: str, notebook_id: str) -> None:
    async with aiosqlite.connect(path, timeout=30) as db:
        await db.execute("DELETE FROM chat_messages WHERE notebook_id = ?", (notebook_id,))
        await db.commit()


async def clear_notebook_cache(path: str, notebook_id: str) -> None:
    """Сбрасывает весь кэш сгенерированного контента блокнота."""
    async with aiosqlite.connect(path, timeout=30) as db:
        await db.execute(
            """UPDATE notebooks SET
               summary = NULL, mindmap = NULL, flashcards = NULL,
               podcast_url = NULL, podcast_script = NULL,
               contract = NULL, knowledge_graph = NULL,
               timeline = NULL, questions = NULL, presentation = NULL
               WHERE id = ?
            """,
            (notebook_id,),
        )
        await db.commit()
