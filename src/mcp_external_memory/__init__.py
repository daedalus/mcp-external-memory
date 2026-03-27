__version__ = "0.1.1"
__all__ = [
    "MemoryStore",
    "MemoryEntry",
    "EmbeddingProvider",
    "TFIDFEmbedder",
    "OpenAIEmbedder",
    "OllamaEmbedder",
    "create_embedding_provider",
    "mcp",
    "memory_store",
    "memory_search",
    "memory_get",
    "memory_delete",
    "memory_list",
    "memory_stats",
    "memory_update",
    "main",
]

from .embeddings import (
    EmbeddingProvider,
    OllamaEmbedder,
    OpenAIEmbedder,
    TFIDFEmbedder,
    create_embedding_provider,
)
from .storage import MemoryEntry, MemoryStore
from .tools import (
    main,
    mcp,
    memory_delete,
    memory_get,
    memory_list,
    memory_search,
    memory_stats,
    memory_store,
    memory_update,
)
