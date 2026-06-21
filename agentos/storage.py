from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

from agentos.models import MemoryRecord


class AgentOSStore:
    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)

    def connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def migrate(self) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id TEXT NOT NULL,
                  memory_type TEXT NOT NULL,
                  content TEXT NOT NULL,
                  source TEXT NOT NULL,
                  importance REAL NOT NULL,
                  confidence REAL NOT NULL,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL,
                  is_active INTEGER NOT NULL,
                  tags_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_memories_user_active
                ON memories(user_id, is_active, memory_type)
                """
            )

    def add_memory(self, record: MemoryRecord) -> MemoryRecord:
        with self.connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO memories (
                  user_id, memory_type, content, source, importance, confidence,
                  created_at, updated_at, is_active, tags_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.user_id,
                    record.memory_type,
                    record.content,
                    record.source,
                    record.importance,
                    record.confidence,
                    record.created_at.isoformat(),
                    record.updated_at.isoformat(),
                    1 if record.is_active else 0,
                    json.dumps(record.tags, ensure_ascii=False),
                ),
            )
            memory_id = int(cursor.lastrowid)
        return record.model_copy(update={"id": memory_id})

    def get_memory(self, memory_id: int | None) -> MemoryRecord | None:
        if memory_id is None:
            return None
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
        return self._row_to_memory(row) if row else None

    def list_active_memories(self, user_id: str) -> list[MemoryRecord]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM memories
                WHERE user_id = ? AND is_active = 1
                ORDER BY importance DESC, confidence DESC, updated_at DESC
                """,
                (user_id,),
            ).fetchall()
        return [self._row_to_memory(row) for row in rows]

    def _row_to_memory(self, row: sqlite3.Row) -> MemoryRecord:
        return MemoryRecord(
            id=row["id"],
            user_id=row["user_id"],
            memory_type=row["memory_type"],
            content=row["content"],
            source=row["source"],
            importance=row["importance"],
            confidence=row["confidence"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            is_active=bool(row["is_active"]),
            tags=json.loads(row["tags_json"]),
        )
