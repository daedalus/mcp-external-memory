from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import time
import uuid
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from mcp.server.fastmcp import FastMCP

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

DB_PATH = Path(os.environ.get("MEMORY_DB_PATH", "~/.semantic_memory.db")).expanduser()
BACKEND = os.environ.get("MEMORY_EMBED_BACKEND", "tfidf").lower()
TOP_K = int(os.environ.get("MEMORY_DEFAULT_TOP_K", "5"))
SCORE_MIN = float(os.environ.get("MEMORY_MIN_SCORE", "0.0"))


_STOPWORDS = {
    "a",
    "an",
    "the",
    "is",
    "it",
    "in",
    "on",
    "at",
    "to",
    "for",
    "of",
    "and",
    "or",
    "but",
    "be",
    "as",
    "by",
    "with",
    "from",
    "that",
    "this",
    "was",
    "are",
    "were",
    "has",
    "have",
    "had",
    "not",
    "so",
    "do",
    "did",
    "does",
    "he",
    "she",
    "they",
    "we",
    "you",
    "i",
    "me",
    "my",
    "your",
    "his",
    "her",
    "its",
    "our",
    "their",
    "what",
    "which",
    "who",
    "whom",
    "when",
    "where",
    "why",
    "how",
    "all",
    "each",
    "every",
    "both",
    "few",
    "more",
    "most",
    "other",
    "some",
    "such",
    "no",
    "nor",
    "only",
    "own",
    "same",
    "than",
    "too",
    "very",
    "just",
    "because",
    "if",
    "then",
    "else",
    "also",
    "about",
    "up",
    "out",
    "into",
    "after",
}


def _tokenize(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOPWORDS]


def _tfidf_vector(tokens: list[str], idf: dict[str, float]) -> dict[str, float]:
    tf = Counter(tokens)
    n = len(tokens) or 1
    return {t: (cnt / n) * idf.get(t, 1.0) for t, cnt in tf.items()}


def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
    dot = sum(a.get(k, 0.0) * v for k, v in b.items())
    na = math.sqrt(sum(v * v for v in a.values())) or 1.0
    nb = math.sqrt(sum(v * v for v in b.values())) or 1.0
    return dot / (na * nb)


class EmbeddingProvider(ABC):
    @abstractmethod
    async def generate(self, text: str) -> list[float]:
        pass


class TFIDFEmbedder(EmbeddingProvider):
    def __init__(self, store: MemoryStore) -> None:
        self._store = store

    async def generate(self, text: str) -> list[float]:
        idf = self._build_idf()
        toks = _tokenize(text)
        vec = _tfidf_vector(toks, idf)
        return self._to_dense(vec, idf)

    def _build_idf(self) -> dict[str, float]:
        texts = self._store.all_texts()
        n = len(texts) or 1
        df: Counter[str] = Counter()
        for t in texts:
            df.update(set(_tokenize(t)))
        return {w: math.log((n + 1) / (cnt + 1)) + 1 for w, cnt in df.items()}

    def _vocab(self, idf: dict[str, float]) -> list[str]:
        return sorted(idf)

    def _to_dense(self, sparse: dict[str, float], idf: dict[str, float]) -> list[float]:
        vocab = self._vocab(idf)
        return [sparse.get(w, 0.0) for w in vocab]


class OpenAIEmbedder(EmbeddingProvider):
    def __init__(self) -> None:
        import httpx

        self._api_key = os.environ["OPENAI_API_KEY"]
        self._model = os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-small")
        self._base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self._http_client = httpx

    async def generate(self, text: str) -> list[float]:
        from tenacity import retry, stop_after_attempt, wait_exponential

        @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
        async def _call() -> list[float]:
            async with self._http_client.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self._base_url}/embeddings",
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    },
                    json={"input": text, "model": self._model},
                )
                response.raise_for_status()
                data = response.json()
                return data["data"][0]["embedding"]  # type: ignore[no-any-return]

        return await _call()


class OllamaEmbedder(EmbeddingProvider):
    def __init__(self) -> None:
        import httpx

        self._url = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self._model = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
        self._http_client = httpx

    async def generate(self, text: str) -> list[float]:
        from tenacity import retry, stop_after_attempt, wait_exponential

        @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
        async def _call() -> list[float]:
            async with self._http_client.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self._url}/api/embeddings",
                    json={"model": self._model, "prompt": text},
                )
                response.raise_for_status()
                data = response.json()
                return data["embedding"]  # type: ignore[no-any-return]

        return await _call()


def create_embedding_provider(store: MemoryStore) -> EmbeddingProvider:
    if BACKEND == "openai":
        return OpenAIEmbedder()
    elif BACKEND == "ollama":
        return OllamaEmbedder()
    else:
        if BACKEND != "tfidf":
            import logging

            logging.warning("Unknown backend %r — falling back to tfidf", BACKEND)
        return TFIDFEmbedder(store)


SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id          TEXT PRIMARY KEY,
    content     TEXT NOT NULL,
    namespace   TEXT NOT NULL DEFAULT 'default',
    tags        TEXT NOT NULL DEFAULT '[]',
    embedding   TEXT NOT NULL DEFAULT '[]',
    metadata    TEXT NOT NULL DEFAULT '{}',
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL,
    access_count INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_namespace ON memories(namespace);
CREATE INDEX IF NOT EXISTS idx_created   ON memories(created_at);
"""


class MemoryStore:
    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def add(self, entry: MemoryEntry) -> None:
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
                entry.content,
                entry.namespace,
                json.dumps(entry.tags),
                json.dumps(entry.embedding or []),
                json.dumps(entry.metadata),
                entry.created_at or now,
                now,
            ),
        )
        self._conn.commit()

    def get(self, mem_id: str) -> MemoryEntry | None:
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
                entry.content,
                entry.namespace,
                json.dumps(entry.tags),
                json.dumps(entry.embedding or []),
                json.dumps(entry.metadata),
                now,
                entry.id,
            ),
        )
        self._conn.commit()

    def delete(self, mem_id: str) -> bool:
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

    def all_texts(self) -> list[str]:
        rows = self._conn.execute("SELECT content FROM memories").fetchall()
        return [r["content"] for r in rows]

    def all_with_embeddings(self) -> list[tuple[str, list[float], MemoryEntry]]:
        rows = self._conn.execute("SELECT * FROM memories").fetchall()
        return [
            (r["id"], json.loads(r["embedding"]), self._row_to_entry(r))
            for r in rows
            if json.loads(r["embedding"])
        ]

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]  # type: ignore[no-any-return]

    def stats(self) -> dict[str, object]:
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
            content=row["content"],
            namespace=row["namespace"],
            tags=json.loads(row["tags"]),
            embedding=json.loads(row["embedding"]),
            metadata=json.loads(row["metadata"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            access_count=row["access_count"],
        )


@dataclass
class MemoryEntry:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    namespace: str = "default"
    tags: list[str] = field(default_factory=list)
    embedding: list[float] | None = None
    metadata: dict = field(default_factory=dict)
    created_at: float | None = None
    updated_at: float | None = None
    access_count: int = 0

    def to_dict(self) -> dict:
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


class AppContext:
    def __init__(
        self, store: MemoryStore, embedding_provider: EmbeddingProvider
    ) -> None:
        self.store = store
        self.embedding_provider = embedding_provider


@asynccontextmanager
async def app_lifespan(_server: FastMCP) -> AsyncIterator[AppContext]:
    store = MemoryStore(DB_PATH)
    embedding_provider = create_embedding_provider(store)
    yield AppContext(store=store, embedding_provider=embedding_provider)


mcp = FastMCP(
    "External Memory", dependencies=[], json_response=True, lifespan=app_lifespan
)


@mcp.tool()
async def memory_store(
    content: str,
    namespace: str = "default",
    tags: list[str] | None = None,
    metadata: dict | None = None,
    id: str | None = None,
) -> dict:
    """
    Persist a piece of information to long-term semantic memory.

    Args:
        content: The text content to store in memory
        namespace: Logical collection (default: 'default')
        tags: Optional tags for filtering
        metadata: Arbitrary JSON metadata
        id: Optional stable ID (upserts if exists)
    """
    ctx = mcp.get_context()
    app_ctx: AppContext = ctx.request_context.lifespan_context

    entry = MemoryEntry(
        id=id or str(uuid.uuid4()),
        content=content,
        namespace=namespace,
        tags=tags or [],
        metadata=metadata or {},
    )

    entry.embedding = await app_ctx.embedding_provider.generate(content)
    app_ctx.store.add(entry)

    return {"id": entry.id, "status": "stored"}


@mcp.tool()
async def memory_search(
    query: str,
    namespace: str | None = None,
    tags: list[str] | None = None,
    top_k: int = TOP_K,
    min_score: float = SCORE_MIN,
) -> dict:
    """
    Semantic search over stored memories.

    Args:
        query: Natural language search query
        namespace: Restrict search to this namespace
        tags: Require at least one of these tags
        top_k: Maximum number of results (default: 5)
        min_score: Minimum cosine similarity threshold (0.0 to 1.0)
    """
    ctx = mcp.get_context()
    app_ctx: AppContext = ctx.request_context.lifespan_context

    candidates = app_ctx.store.all_with_embeddings()
    if not candidates:
        return {"results": [], "count": 0}

    qemb = await app_ctx.embedding_provider.generate(query)

    scored: list[tuple[float, MemoryEntry]] = []
    for mid, emb, entry in candidates:
        if namespace and entry.namespace != namespace:
            continue
        if tags and not (set(tags) & set(entry.tags)):
            continue
        score = _cosine_dense(qemb, emb)
        if score >= min_score:
            scored.append((score, entry))

    scored.sort(key=lambda x: x[0], reverse=True)
    results = []
    for score, entry in scored[:top_k]:
        d = entry.to_dict()
        d["score"] = round(score, 4)
        results.append(d)

    return {"results": results, "count": len(results)}


def _cosine_dense(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (na * nb)


@mcp.tool()
async def memory_get(id: str) -> dict:
    """
    Retrieve a specific memory by its ID.

    Args:
        id: The memory ID returned by memory_store
    """
    ctx = mcp.get_context()
    app_ctx: AppContext = ctx.request_context.lifespan_context

    entry = app_ctx.store.get(id)
    if entry is None:
        return {"error": f"Memory {id!r} not found"}
    return entry.to_dict()


@mcp.tool()
async def memory_delete(id: str) -> dict:
    """
    Delete a memory by its ID.

    Args:
        id: Memory ID to delete
    """
    ctx = mcp.get_context()
    app_ctx: AppContext = ctx.request_context.lifespan_context

    deleted = app_ctx.store.delete(id)
    return {"deleted": deleted}


@mcp.tool()
async def memory_list(
    namespace: str | None = None,
    tags: list[str] | None = None,
    limit: int = 50,
    offset: int = 0,
) -> dict:
    """
    List stored memories, optionally filtered by namespace or tags.

    Args:
        namespace: Filter by namespace
        tags: Filter by tags (returns memories with at least one matching tag)
        limit: Maximum number of results (default: 50)
        offset: Pagination offset
    """
    ctx = mcp.get_context()
    app_ctx: AppContext = ctx.request_context.lifespan_context

    entries = app_ctx.store.list(
        namespace=namespace, tags=tags, limit=limit, offset=offset
    )
    return {"memories": [e.to_dict() for e in entries], "count": len(entries)}


@mcp.tool()
async def memory_stats() -> dict:
    """
    Return memory store statistics: total memories, namespace counts, DB path.
    """
    ctx = mcp.get_context()
    app_ctx: AppContext = ctx.request_context.lifespan_context

    return app_ctx.store.stats()


@mcp.tool()
async def memory_update(
    id: str,
    content: str | None = None,
    namespace: str | None = None,
    tags: list[str] | None = None,
    metadata: dict | None = None,
) -> dict:
    """
    Update an existing memory's content, namespace, tags, and/or metadata.

    Args:
        id: The unique identifier of the memory to update
        content: New text content (optional, regenerates embedding)
        namespace: New namespace (optional)
        tags: New tags list (optional, replaces existing)
        metadata: Additional metadata (optional, merges with existing)
    """
    ctx = mcp.get_context()
    app_ctx: AppContext = ctx.request_context.lifespan_context

    entry = app_ctx.store.get(id)
    if entry is None:
        return {"error": f"Memory {id!r} not found"}

    if content is not None:
        entry.content = content
        entry.embedding = await app_ctx.embedding_provider.generate(content)
    if namespace is not None:
        entry.namespace = namespace
    if tags is not None:
        entry.tags = tags
    if metadata is not None:
        entry.metadata = {**entry.metadata, **metadata}

    app_ctx.store.update(entry)
    return entry.to_dict()


def main() -> int:

    transport = os.getenv("MCP_TRANSPORT", "stdio")
    if transport == "streamable-http":
        mcp.run(transport="streamable-http", host="0.0.0.0", port=8080)
    elif transport == "sse":
        mcp.run(transport="sse")
    else:
        mcp.run(transport="stdio")
    return 0
