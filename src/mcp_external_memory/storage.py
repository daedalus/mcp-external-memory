from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import lz4.frame


def _compress(data: str) -> bytes:
    return lz4.frame.compress(data.encode("utf-8"))  # type: ignore[no-any-return]


def _decompress(data: bytes) -> str:
    return lz4.frame.decompress(data).decode("utf-8")  # type: ignore[no-any-return]


DB_PATH = Path(os.environ.get("MEMORY_DB_PATH", "~/.semantic_memory.db")).expanduser()


SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id          TEXT PRIMARY KEY,
    content     BLOB NOT NULL,
    namespace   TEXT NOT NULL DEFAULT 'default',
    tags        TEXT NOT NULL DEFAULT '[]',
    embedding   BLOB NOT NULL,
    metadata    TEXT NOT NULL DEFAULT '{}',
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL,
    access_count INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_namespace ON memories(namespace);
CREATE INDEX IF NOT EXISTS idx_created   ON memories(created_at);
"""


@dataclass
class MemoryEntry:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    namespace: str = "default"
    tags: list[str] = field(default_factory=list)
    embedding: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float | None = None
    updated_at: float | None = None
    access_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert MemoryEntry to dictionary representation.

        Returns:
            Dictionary with all memory fields.

        Examples:
            >>> entry = MemoryEntry(id="123", content="test")
            >>> entry.to_dict()["id"]
            '123'
        """
        return {
            "id": self.id,
            "content": self.content,
            "namespace": self.namespace,
            "tags": self.tags,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "access_count": self.access_count,
        }


class MemoryStore:
    """SQLite-backed memory store with LZ4 compression."""

    def __init__(self, db_path: Path) -> None:
        """Initialize memory store with database at given path.

        Args:
            db_path: Path to SQLite database file.
        """
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def add(self, entry: MemoryEntry) -> None:
        """Add or update a memory entry in the store.

        Args:
            entry: MemoryEntry to store.
        """
        now = time.time()
        self._conn.execute(
            """INSERT INTO memories
               (id, content, namespace, tags, embedding, metadata, created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?)
               ON CONFLICT(id) DO UPDATE SET
                 content=excluded.content,
                 embedding=excluded.embedding,
                 tags=excluded.tags,
                 metadata=excluded.metadata,
                 updated_at=excluded.updated_at""",
            (
                entry.id,
                _compress(entry.content),
                entry.namespace,
                json.dumps(entry.tags),
                _compress(json.dumps(entry.embedding or [])),
                json.dumps(entry.metadata),
                entry.created_at or now,
                now,
            ),
        )
        self._conn.commit()

    def get(self, mem_id: str) -> MemoryEntry | None:
        """Retrieve a memory entry by ID.

        Args:
            mem_id: Unique identifier of memory.

        Returns:
            MemoryEntry if found, None otherwise.
        """
        row = self._conn.execute(
            "SELECT * FROM memories WHERE id=?", (mem_id,)
        ).fetchone()
        if row is None:
            return None
        self._conn.execute(
            "UPDATE memories SET access_count=access_count+1 WHERE id=?", (mem_id,)
        )
        self._conn.commit()
        return self._row_to_entry(row)

    def update(self, entry: MemoryEntry) -> None:
        now = time.time()
        self._conn.execute(
            """UPDATE memories SET
               content=?, namespace=?, tags=?, embedding=?, metadata=?, updated_at=?
               WHERE id=?""",
            (
                _compress(entry.content),
                entry.namespace,
                json.dumps(entry.tags),
                _compress(json.dumps(entry.embedding or [])),
                json.dumps(entry.metadata),
                now,
                entry.id,
            ),
        )
        self._conn.commit()

    def delete(self, mem_id: str) -> bool:
        """Delete a memory entry by ID.

        Args:
            mem_id: Unique identifier of memory to delete.

        Returns:
            True if entry was deleted, False if not found.
        """
        cur = self._conn.execute("DELETE FROM memories WHERE id=?", (mem_id,))
        self._conn.commit()
        return cur.rowcount > 0

    def list(
        self,
        namespace: str | None = None,
        tags: list[str] | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[MemoryEntry]:
        """List memory entries with optional filtering.

        Args:
            namespace: Filter by namespace.
            tags: Filter by tags (OR logic - returns entries with any matching tag).
            limit: Maximum number of results.
            offset: Number of results to skip (for pagination).

        Returns:
            List of matching MemoryEntry objects.
        """
        query = "SELECT * FROM memories"
        params: list[object] = []
        clauses: list[str] = []
        if namespace:
            clauses.append("namespace=?")
            params.append(namespace)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY updated_at DESC LIMIT ? OFFSET ?"
        params += [limit, offset]
        rows = self._conn.execute(query, params).fetchall()
        results = [self._row_to_entry(r) for r in rows]
        if tags:
            tag_set = set(tags)
            results = [r for r in results if tag_set & set(r.tags)]
        return results

    def all_texts(self) -> list[str]:  # type: ignore[valid-type]
        """Get all text content from stored memories.

        Returns:
            List of all memory content strings.
        """
        rows = self._conn.execute("SELECT content FROM memories").fetchall()
        return [_decompress(r["content"]) for r in rows]

    def all_with_embeddings(self) -> list[tuple[str, list[float], MemoryEntry]]:  # type: ignore[valid-type]
        """Get all memories that have embeddings.

        Returns:
            List of tuples (id, embedding, MemoryEntry).
        """
        rows = self._conn.execute("SELECT * FROM memories").fetchall()
        result: list[tuple[str, list[float], MemoryEntry]] = []
        for r in rows:
            emb_bytes = r["embedding"]
            if emb_bytes:
                emb = json.loads(_decompress(emb_bytes))
                result.append((r["id"], emb, self._row_to_entry(r)))
        return result

    def count(self) -> int:
        """Get total count of stored memories.

        Returns:
            Number of memory entries in store.
        """
        return self._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]  # type: ignore[no-any-return]

    def stats(self) -> dict[str, Any]:
        """Get statistics about the memory store.

        Returns:
            Dictionary with total_memories, namespaces dict, and db_path.
        """
        total = self._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        ns = self._conn.execute(
            "SELECT namespace, COUNT(*) as c FROM memories GROUP BY namespace"
        ).fetchall()
        return {
            "total_memories": total,
            "namespaces": {r["namespace"]: r["c"] for r in ns},
            "db_path": str(DB_PATH),
        }

    @staticmethod
    def _row_to_entry(row: sqlite3.Row) -> MemoryEntry:
        return MemoryEntry(
            id=row["id"],
            content=_decompress(row["content"]),
            namespace=row["namespace"],
            tags=json.loads(row["tags"]),
            embedding=json.loads(_decompress(row["embedding"])),
            metadata=json.loads(row["metadata"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            access_count=row["access_count"],
        )
