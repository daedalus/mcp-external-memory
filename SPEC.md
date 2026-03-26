# SPEC.md — mcp-external-memory

## Purpose

An MCP server that gives LLMs persistent, searchable semantic memory using SQLite for storage and pluggable embedding backends (TF-IDF, OpenAI, Ollama).

## Scope

### In Scope
- MCP protocol server with stdio/SSE/streamable-http transports
- SQLite-based memory persistence with namespaces and tags
- Semantic search via cosine similarity with multiple embedding providers
- Memory CRUD operations (store, get, update, delete, list, search)
- Memory statistics and metadata support

### Not Out of Scope
- User authentication/authorization
- Multi-instance sync
- Graph-based memory relationships
- Vector database backends (pgvector, etc.)

## Public API / Interface

### MCP Tools

| Tool | Args | Returns | Description |
|------|------|---------|-------------|
| `memory_store` | `content: str`, `namespace: str = "default"`, `tags: list[str]?`, `metadata: dict?`, `id: str?` | `{"id": str, "status": "stored"}` | Persist text to semantic memory |
| `memory_search` | `query: str`, `namespace: str?`, `tags: list[str]?`, `top_k: int = 5`, `min_score: float = 0.0` | `{"results": list, "count": int}` | Semantic search over memories |
| `memory_get` | `id: str` | `dict` (memory entry or error) | Retrieve single memory by ID |
| `memory_delete` | `id: str` | `{"deleted": bool}` | Delete memory by ID |
| `memory_list` | `namespace: str?`, `tags: list[str]?`, `limit: int = 50`, `offset: int = 0` | `{"memories": list, "count": int}` | List memories with filtering |
| `memory_stats` | — | `dict` | Return total count, namespace counts, DB path |
| `memory_update` | `id: str`, `content: str?`, `namespace: str?`, `tags: list[str]?`, `metadata: dict?` | `dict` | Update existing memory |

### Configuration (Environment Variables)

| Variable | Default | Description |
|----------|---------|-------------|
| `MEMORY_DB_PATH` | `~/.semantic_memory.db` | SQLite database path |
| `MEMORY_EMBED_BACKEND` | `tfidf` | Embedding backend: `tfidf`/`openai`/`ollama` |
| `MEMORY_DEFAULT_TOP_K` | `5` | Default search result count |
| `MEMORY_MIN_SCORE` | `0.0` | Minimum cosine similarity threshold |
| `OPENAI_API_KEY` | — | Required for openai backend |
| `OPENAI_EMBED_MODEL` | `text-embedding-3-small` | OpenAI model |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | OpenAI API base URL |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama base URL |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Ollama model |
| `MCP_TRANSPORT` | `stdio` | Transport: `stdio`, `streamable-http`, `sse` |

## Data Formats

- **Database**: SQLite with table `memories` (id, content, namespace, tags, embedding, metadata, created_at, updated_at, access_count)
- **Embeddings**: JSON-serialized list of floats in SQLite
- **Tags/Metadata**: JSON-serialized lists/dicts

## Edge Cases

1. Empty query string in search — should return empty results
2. Non-existent memory ID in get/update/delete — get/update returns error, delete returns `deleted: false`
3. Namespace with no memories — search returns empty, list returns empty
4. Memory with empty content — allowed, generates embedding from empty string
5. Very long content (>100KB) — handled by embedding provider (may timeout or truncate)
6. Multiple tags matching (OR logic) — returns memories matching any tag
7. Concurrent access — SQLite with check_same_thread=False handles basic concurrency

## Performance & Constraints

- TF-IDF backend: O(n) per search where n = total memories
- Embedding generation: Async, with 3 retries and exponential backoff
- Default top_k: 5, max list limit: 50
- No pagination for search results (only list has offset)