"""SQLite database: init + CRUD for projects/images/results."""

import aiosqlite
import os
import uuid
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "delo_color.db")


async def get_db() -> aiosqlite.Connection:
    db = await aiosqlite.connect(DB_PATH)
    db.row_factory = aiosqlite.Row
    await db.execute("PRAGMA journal_mode=WAL")
    await db.execute("PRAGMA foreign_keys=ON")
    return db


async def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    db = await get_db()
    await db.executescript("""
        CREATE TABLE IF NOT EXISTS projects (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS images (
            id TEXT PRIMARY KEY,
            project_id TEXT REFERENCES projects(id) ON DELETE CASCADE,
            filename TEXT NOT NULL,
            has_mask BOOLEAN DEFAULT 0,
            status TEXT DEFAULT 'uploaded'
        );
        CREATE TABLE IF NOT EXISTS recolor_results (
            id TEXT PRIMARY KEY,
            image_id TEXT REFERENCES images(id) ON DELETE CASCADE,
            ral_code TEXT NOT NULL,
            filename TEXT NOT NULL
        );
    """)
    await db.commit()
    await db.close()


# ─── Projects ─────────────────────────────────────────────────────────────

async def create_project(name: str) -> dict:
    db = await get_db()
    pid = str(uuid.uuid4())
    await db.execute(
        "INSERT INTO projects (id, name) VALUES (?, ?)", (pid, name)
    )
    await db.commit()
    await db.close()
    return {"id": pid, "name": name}


async def list_projects() -> list[dict]:
    db = await get_db()
    cursor = await db.execute("""
        SELECT p.id, p.name, p.created_at,
               COUNT(i.id) as image_count
        FROM projects p
        LEFT JOIN images i ON i.project_id = p.id
        GROUP BY p.id
        ORDER BY p.created_at DESC
    """)
    rows = await cursor.fetchall()
    await db.close()
    return [dict(r) for r in rows]


async def get_project(project_id: str) -> dict | None:
    db = await get_db()
    cursor = await db.execute("SELECT * FROM projects WHERE id = ?", (project_id,))
    row = await cursor.fetchone()
    await db.close()
    return dict(row) if row else None


async def delete_project(project_id: str):
    db = await get_db()
    await db.execute("DELETE FROM projects WHERE id = ?", (project_id,))
    await db.commit()
    await db.close()


# ─── Images ───────────────────────────────────────────────────────────────

async def add_image(project_id: str, filename: str) -> dict:
    db = await get_db()
    iid = str(uuid.uuid4())
    await db.execute(
        "INSERT INTO images (id, project_id, filename) VALUES (?, ?, ?)",
        (iid, project_id, filename),
    )
    await db.commit()
    await db.close()
    return {"id": iid, "filename": filename}


async def get_images(project_id: str) -> list[dict]:
    db = await get_db()
    cursor = await db.execute(
        "SELECT * FROM images WHERE project_id = ? ORDER BY filename",
        (project_id,),
    )
    rows = await cursor.fetchall()
    await db.close()
    return [dict(r) for r in rows]


async def update_image_status(image_id: str, status: str, has_mask: bool | None = None):
    db = await get_db()
    if has_mask is not None:
        await db.execute(
            "UPDATE images SET status = ?, has_mask = ? WHERE id = ?",
            (status, has_mask, image_id),
        )
    else:
        await db.execute(
            "UPDATE images SET status = ? WHERE id = ?", (status, image_id)
        )
    await db.commit()
    await db.close()


async def delete_image(image_id: str) -> dict | None:
    db = await get_db()
    cursor = await db.execute("SELECT * FROM images WHERE id = ?", (image_id,))
    row = await cursor.fetchone()
    if row:
        await db.execute("DELETE FROM images WHERE id = ?", (image_id,))
        await db.commit()
    await db.close()
    return dict(row) if row else None


# ─── Results ──────────────────────────────────────────────────────────────

async def add_result(image_id: str, ral_code: str, filename: str):
    db = await get_db()
    rid = str(uuid.uuid4())
    await db.execute(
        "INSERT INTO recolor_results (id, image_id, ral_code, filename) VALUES (?, ?, ?, ?)",
        (rid, image_id, ral_code, filename),
    )
    await db.commit()
    await db.close()


async def get_results(project_id: str) -> list[dict]:
    db = await get_db()
    cursor = await db.execute("""
        SELECT r.id, r.image_id, r.ral_code, r.filename, i.filename as original
        FROM recolor_results r
        JOIN images i ON i.id = r.image_id
        WHERE i.project_id = ?
        ORDER BY i.filename, r.ral_code
    """, (project_id,))
    rows = await cursor.fetchall()
    await db.close()
    return [dict(r) for r in rows]


async def clear_results(project_id: str):
    db = await get_db()
    await db.execute("""
        DELETE FROM recolor_results WHERE image_id IN (
            SELECT id FROM images WHERE project_id = ?
        )
    """, (project_id,))
    await db.commit()
    await db.close()
