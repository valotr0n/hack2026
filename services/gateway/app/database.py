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
"""


async def init_db(path: str) -> None:
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    async with aiosqlite.connect(path) as db:
        await db.executescript(_SCHEMA)
        # Миграции — добавляем колонки для кэша контента если их ещё нет
        for col, typedef in [
            ("summary", "TEXT"),
            ("mindmap", "TEXT"),
            ("flashcards", "TEXT"),
            ("podcast_url", "TEXT"),
            ("podcast_script", "TEXT"),
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
    async with aiosqlite.connect(path) as db:
        await db.execute(
            "INSERT INTO users (id, username, password_hash, created_at) VALUES (?, ?, ?, ?)",
            (uid, username, password_hash, now),
        )
        await db.commit()
    return {"id": uid, "username": username, "created_at": now}


async def get_user_by_username(path: str, username: str) -> dict[str, Any] | None:
    async with aiosqlite.connect(path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT id, username, password_hash, created_at FROM users WHERE username = ?",
            (username,),
        ) as cur:
            row = await cur.fetchone()
            return dict(row) if row else None


async def get_user_by_id(path: str, user_id: str) -> dict[str, Any] | None:
    async with aiosqlite.connect(path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT id, username, created_at FROM users WHERE id = ?",
            (user_id,),
        ) as cur:
            row = await cur.fetchone()
            return dict(row) if row else None


# ── Notebooks ─────────────────────────────────────────────────────────────────

async def create_notebook(path: str, user_id: str, title: str) -> dict[str, Any]:
    nid = _new_id()
    now = _now()
    async with aiosqlite.connect(path) as db:
        await db.execute(
            "INSERT INTO notebooks (id, user_id, title, created_at) VALUES (?, ?, ?, ?)",
            (nid, user_id, title, now),
        )
        await db.commit()
    return {"id": nid, "user_id": user_id, "title": title, "created_at": now}


async def list_notebooks(path: str, user_id: str) -> list[dict[str, Any]]:
    async with aiosqlite.connect(path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT id, title, created_at FROM notebooks WHERE user_id = ? ORDER BY created_at DESC",
            (user_id,),
        ) as cur:
            return [dict(r) for r in await cur.fetchall()]


async def get_notebook(path: str, notebook_id: str) -> dict[str, Any] | None:
    async with aiosqlite.connect(path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT id, user_id, title, created_at, summary, mindmap, flashcards, podcast_url, podcast_script FROM notebooks WHERE id = ?",
            (notebook_id,),
        ) as cur:
            row = await cur.fetchone()
            return dict(row) if row else None


async def save_notebook_content(path: str, notebook_id: str, field: str, value: str) -> None:
    allowed = {"summary", "mindmap", "flashcards", "podcast_url", "podcast_script"}
    if field not in allowed:
        raise ValueError(f"Unknown field: {field}")
    async with aiosqlite.connect(path) as db:
        await db.execute(
            f"UPDATE notebooks SET {field} = ? WHERE id = ?",
            (value, notebook_id),
        )
        await db.commit()


async def update_notebook_title(path: str, notebook_id: str, title: str) -> None:
    async with aiosqlite.connect(path) as db:
        await db.execute(
            "UPDATE notebooks SET title = ? WHERE id = ?",
            (title, notebook_id),
        )
        await db.commit()


async def delete_notebook(path: str, notebook_id: str) -> None:
    async with aiosqlite.connect(path) as db:
        await db.execute("DELETE FROM notebooks WHERE id = ?", (notebook_id,))
        await db.commit()


# ── Sources ───────────────────────────────────────────────────────────────────

async def create_source(
    path: str,
    notebook_id: str,
    filename: str,
    chunks_count: int,
) -> dict[str, Any]:
    sid = _new_id()
    now = _now()
    async with aiosqlite.connect(path) as db:
        await db.execute(
            "INSERT INTO sources (id, notebook_id, filename, chunks_count, created_at) VALUES (?, ?, ?, ?, ?)",
            (sid, notebook_id, filename, chunks_count, now),
        )
        await db.commit()
    return {
        "id": sid,
        "notebook_id": notebook_id,
        "filename": filename,
        "chunks_count": chunks_count,
        "created_at": now,
    }


async def list_sources(path: str, notebook_id: str) -> list[dict[str, Any]]:
    async with aiosqlite.connect(path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT id, filename, chunks_count, created_at FROM sources WHERE notebook_id = ? ORDER BY created_at ASC",
            (notebook_id,),
        ) as cur:
            return [dict(r) for r in await cur.fetchall()]


async def get_source(path: str, source_id: str) -> dict[str, Any] | None:
    async with aiosqlite.connect(path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT id, notebook_id, filename, chunks_count, created_at FROM sources WHERE id = ?",
            (source_id,),
        ) as cur:
            row = await cur.fetchone()
            return dict(row) if row else None


async def delete_source(path: str, source_id: str) -> None:
    async with aiosqlite.connect(path) as db:
        await db.execute("DELETE FROM sources WHERE id = ?", (source_id,))
        await db.commit()
