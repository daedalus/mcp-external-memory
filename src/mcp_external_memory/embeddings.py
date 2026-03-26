from __future__ import annotations

import math
import os
import re
from abc import ABC, abstractmethod
from collections import Counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .storage import MemoryStore


_BACKEND = os.environ.get("MEMORY_EMBED_BACKEND", "tfidf").lower()


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


def _cosine_dense(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
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
    if _BACKEND == "openai":
        return OpenAIEmbedder()
    elif _BACKEND == "ollama":
        return OllamaEmbedder()
    else:
        if _BACKEND != "tfidf":
            import logging

            logging.warning("Unknown backend %r — falling back to tfidf", _BACKEND)
        return TFIDFEmbedder(store)
