---
name: external-memory
description: Use this skill whenever the user wants to store, retrieve, search, or manage persistent semantic memory. This includes requests related to memory management, knowledge retention, context caching, storing information across sessions, semantic search over past conversations, tagging and organizing stored information, namespaces, memory statistics, or any MCP external-memory operations. Trigger on mentions like "store in memory", "remember this", "what do I remember about", "search my memories", "semantic search", "forget this", "delete memory", "list memories", "memory stats", "update memory", or working with the mcp-external-memory package.
---

# External Memory Skill

This skill provides access to a persistent, searchable semantic memory system via the MCP (Model Context Protocol) server.

## Tools Overview

The external-memory MCP server provides these tools:

| Tool | Description |
|------|-------------|
| `memory_store` | Persist text with optional namespace/tags/metadata |
| `memory_search` | Semantic search using cosine similarity |
| `memory_get` | Retrieve a single memory by ID |
| `memory_delete` | Delete a memory by ID |
| `memory_list` | List memories with filtering and pagination |
| `memory_stats` | Get statistics (counts, namespaces) |
| `memory_update` | Update an existing memory |

## Storing Memories

Call `memory_store` to persist information:

```python
# Basic storage
memory_store(content="Alice prefers dark mode", namespace="users")

# With tags
memory_store(
    content="The API endpoint is https://api.example.com/v1",
    namespace="config",
    tags=["api", "production"]
)

# With metadata
memory_store(
    content="Meeting with Bob at 3pm",
    namespace="meetings",
    metadata={"date": "2026-04-13", "attendees": ["alice", "bob"]}
)
```

Parameters:
- `content` (required): The text to store
- `namespace`: Logical collection (default: "default")
- `tags`: Optional list of tags for filtering
- `metadata`: Arbitrary JSON metadata
- `id`: Optional stable ID for upserts

## Searching Memories

Call `memory_search` for semantic similarity search:

```python
# Basic search
memory_search(query="what does Alice prefer?")

# With filters
memory_search(
    query="API configuration details",
    namespace="config",
    tags=["api"],
    top_k=10,
    min_score=0.7
)
```

Returns results sorted by cosine similarity score.

## Retrieving Specific Memories

```python
# By ID
memory_get(id="uuid-here")

# List with filtering
memory_list(namespace="users", tags=["alice"], limit=20, offset=0)
```

## Updating and Deleting

```python
# Update
memory_update(id="uuid", content="new content", tags=["updated"])

# Delete
memory_delete(id="uuid")
```

## Getting Statistics

```python
memory_stats()
# Returns: {"total": 150, "namespaces": {"users": 50, "config": 100}, "db_path": "..."}
```

## Configuration

Set environment variables to customize behavior:

- `MEMORY_DB_PATH`: Database file path (default: `~/.memory/memory.db`)
- `MEMORY_EMBED_BACKEND`: Embedding backend (tfidf, openai, ollama)
- `MEMORY_OPENAI_MODEL`: OpenAI model (default: text-embedding-3-small)
- `MEMORY_OLLAMA_URL`: Ollama server URL
- `MEMORY_DEFAULT_TOP_K`: Default search results (default: 5)
- `MEMORY_MIN_SCORE`: Minimum similarity threshold (default: 0.0)

## Embedding Backends

- **TF-IDF** (default): Pure Python, no external dependencies
- **OpenAI**: Uses `text-embedding-3-small` model
- **Ollama**: Local embeddings with Ollama