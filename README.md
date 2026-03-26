# mcp-external-memory

> An MCP server that gives LLMs persistent, searchable semantic memory.

[![PyPI](https://img.shields.io/pypi/v/mcp-external-memory.svg)](https://pypi.org/project/mcp-external-memory/)
[![Python](https://img.shields.io/pypi/pyversions/mcp-external-memory.svg)](https://pypi.org/project/mcp-external-memory/)
[![Coverage](https://codecov.io/gh/daedalus/mcp-external-memory/branch/main/graph/badge.svg)](https://codecov.io/gh/daedalus/mcp-external-memory)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

## Install

```bash
pip install mcp-external-memory
```

## Usage

```python
from mcp_external_memory import memory_store, memory_search

# Store a memory
result = memory_store(content="Alice prefers dark mode", namespace="users", tags=["alice", "ui"])

# Search memories
results = memory_search(query="what does Alice prefer?", namespace="users")
```

## CLI

```bash
mcp-external-memory --help
```

## API

### Tools

| Tool | Description |
|------|-------------|
| `memory_store` | Persist text + optional namespace/tags/metadata |
| `memory_search` | Semantic search (cosine similarity) over all memories |
| `memory_get` | Retrieve a single memory by ID |
| `memory_delete` | Delete a memory by ID |
| `memory_list` | List memories with optional namespace/tag filter + pagination |
| `memory_stats` | Count of memories, namespaces, DB path |
| `memory_update` | Update an existing memory |

### Embedding Backends

The server supports multiple embedding backends:

- **TF-IDF** (default): Pure Python, no external dependencies
- **OpenAI**: Uses `text-embedding-3-small` model
- **Ollama**: Local embeddings with Ollama

Set via `MEMORY_EMBED_BACKEND` environment variable.

## Development

```bash
git clone https://github.com/daedalus/mcp-external-memory.git
cd mcp-external-memory
pip install -e ".[test]"

# run tests
pytest

# format
ruff format src/ tests/

# lint
ruff check src/ tests/

# type check
mypy src/
```

## MCP Registry

mcp-name: io.github.daedalus/mcp-external-memory