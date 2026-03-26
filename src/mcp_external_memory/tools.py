from __future__ import annotations

import os
import uuid
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from mcp.server.fastmcp import FastMCP

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from .embeddings import EmbeddingProvider
    from .storage import MemoryEntry, MemoryStore


TOP_K = int(os.environ.get("MEMORY_DEFAULT_TOP_K", "5"))
SCORE_MIN = float(os.environ.get("MEMORY_MIN_SCORE", "0.0"))


class AppContext:
    def __init__(
        self, store: MemoryStore, embedding_provider: EmbeddingProvider
    ) -> None:
        self.store = store
        self.embedding_provider = embedding_provider


@asynccontextmanager
async def app_lifespan(_server: FastMCP) -> AsyncIterator[AppContext]:
    from .embeddings import create_embedding_provider
    from .storage import DB_PATH, MemoryStore

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
    from .storage import MemoryEntry

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
    from .embeddings import _cosine_dense

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
