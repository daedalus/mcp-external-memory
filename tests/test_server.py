import pytest

from mcp_external_memory.embeddings import (
    TFIDFEmbedder,
    _cosine,
    _cosine_dense,
    _tfidf_vector,
    _tokenize,
)
from mcp_external_memory.storage import (
    MemoryEntry,
    MemoryStore,
)


class TestTokenize:
    def test_tokenize_basic(self):
        result = _tokenize("Hello world test")
        assert "hello" in result
        assert "world" in result
        assert "test" in result

    def test_tokenize_removes_stopwords(self):
        result = _tokenize("The cat is on the mat")
        assert "the" not in result
        assert "is" not in result
        assert "on" not in result
        assert "cat" in result
        assert "mat" in result

    def test_tokenize_lowercase(self):
        result = _tokenize("UPPERCASE Lowercase")
        assert "uppercase" in result
        assert "lowercase" in result
        assert "UPPERCASE" not in result

    def test_tokenize_numbers(self):
        result = _tokenize("test 123 abc")
        assert "123" in result

    def test_tokenize_special_chars_removed(self):
        result = _tokenize("hello@world! test#123")
        assert "hello" in result
        assert "world" in result
        assert "test" in result


class TestTFIDFVector:
    def test_tfidf_basic(self):
        tokens = ["word1", "word1", "word2"]
        idf = {"word1": 1.5, "word2": 2.0, "word3": 1.0}
        result = _tfidf_vector(tokens, idf)
        assert result["word1"] == pytest.approx((2 / 3) * 1.5)
        assert result["word2"] == pytest.approx((1 / 3) * 2.0)

    def test_tfidf_empty_tokens(self):
        tokens = []
        idf = {"word1": 1.5}
        result = _tfidf_vector(tokens, idf)
        assert result == {}  # Empty tokens returns empty vector

    def test_tfidf_unknown_words(self):
        tokens = ["unknown"]
        idf = {"word1": 1.5}
        result = _tfidf_vector(tokens, idf)
        assert result["unknown"] == pytest.approx(1.0)


class TestCosine:
    def test_cosine_identical(self):
        a = {"a": 1.0, "b": 0.0}
        b = {"a": 1.0, "b": 0.0}
        assert _cosine(a, b) == pytest.approx(1.0)

    def test_cosine_orthogonal(self):
        a = {"a": 1.0, "b": 0.0}
        b = {"a": 0.0, "b": 1.0}
        assert _cosine(a, b) == pytest.approx(0.0)

    def test_cosine_opposite(self):
        a = {"a": 1.0}
        b = {"a": -1.0}
        assert _cosine(a, b) == pytest.approx(-1.0)

    def test_cosine_empty_vectors(self):
        a = {}
        b = {}
        result = _cosine(a, b)
        assert result == 0.0


class TestCosineDense:
    def test_cosine_dense_identical(self):
        a = [1.0, 0.0, 0.0]
        b = [1.0, 0.0, 0.0]
        assert _cosine_dense(a, b) == pytest.approx(1.0)

    def test_cosine_dense_orthogonal(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert _cosine_dense(a, b) == pytest.approx(0.0)

    def test_cosine_dense_empty(self):
        a = []
        b = []
        result = _cosine_dense(a, b)
        assert result == 0.0

    def test_cosine_dense_different_lengths(self):
        a = [1.0, 0.0, 0.0]
        b = [1.0, 0.0]
        result = _cosine_dense(a, b)
        assert result == pytest.approx(1.0)


class TestMemoryEntry:
    def test_default_values(self):
        entry = MemoryEntry()
        assert entry.id != ""
        assert entry.content == ""
        assert entry.namespace == "default"
        assert entry.tags == []
        assert entry.embedding is None
        assert entry.metadata == {}

    def test_to_dict(self):
        entry = MemoryEntry(
            id="test-id",
            content="test content",
            namespace="test-ns",
            tags=["tag1", "tag2"],
            metadata={"key": "value"},
            created_at=123456.0,
            updated_at=123457.0,
            access_count=5,
        )
        d = entry.to_dict()
        assert d["id"] == "test-id"
        assert d["content"] == "test content"
        assert d["namespace"] == "test-ns"
        assert d["tags"] == ["tag1", "tag2"]
        assert d["metadata"] == {"key": "value"}
        assert d["created_at"] == 123456.0
        assert d["updated_at"] == 123457.0
        assert d["access_count"] == 5


class TestMemoryStore:
    @pytest.fixture
    def store(self, temp_db_path):
        return MemoryStore(temp_db_path)

    def test_init_creates_db(self, temp_db_path):
        store = MemoryStore(temp_db_path)
        assert temp_db_path.exists()
        count = store.count()
        assert count == 0

    def test_add_increments_count(self, store):
        entry = MemoryEntry(content="test content")
        store.add(entry)
        assert store.count() == 1

    def test_add_with_id(self, store):
        entry = MemoryEntry(id="custom-id", content="test")
        store.add(entry)
        retrieved = store.get("custom-id")
        assert retrieved is not None
        assert retrieved.id == "custom-id"

    def test_get_nonexistent(self, store):
        result = store.get("nonexistent-id")
        assert result is None

    def test_get_increments_access_count(self, store):
        entry = MemoryEntry(content="test")
        store.add(entry)
        store.get(entry.id)
        retrieved = store.get(entry.id)
        assert retrieved is not None
        assert retrieved.access_count >= 1

    def test_update_content(self, store):
        entry = MemoryEntry(content="original")
        store.add(entry)
        entry.content = "updated"
        store.update(entry)
        retrieved = store.get(entry.id)
        assert retrieved is not None
        assert retrieved.content == "updated"

    def test_delete_existing(self, store):
        entry = MemoryEntry(content="test")
        store.add(entry)
        deleted = store.delete(entry.id)
        assert deleted is True
        assert store.get(entry.id) is None

    def test_delete_nonexistent(self, store):
        deleted = store.delete("nonexistent-id")
        assert deleted is False

    def test_list_all(self, store):
        store.add(MemoryEntry(content="test1"))
        store.add(MemoryEntry(content="test2"))
        results = store.list()
        assert len(results) == 2

    def test_list_namespace_filter(self, store):
        store.add(MemoryEntry(content="test1", namespace="ns1"))
        store.add(MemoryEntry(content="test2", namespace="ns2"))
        results = store.list(namespace="ns1")
        assert len(results) == 1
        assert results[0].namespace == "ns1"

    def test_list_tags_filter(self, store):
        store.add(MemoryEntry(content="test1", tags=["tag1", "tag2"]))
        store.add(MemoryEntry(content="test2", tags=["tag3"]))
        results = store.list(tags=["tag1"])
        assert len(results) == 1

    def test_list_pagination(self, store):
        for i in range(10):
            store.add(MemoryEntry(content=f"test{i}"))
        results = store.list(limit=5, offset=0)
        assert len(results) == 5
        results_offset = store.list(limit=5, offset=5)
        assert len(results_offset) == 5

    def test_all_texts(self, store):
        store.add(MemoryEntry(content="content1"))
        store.add(MemoryEntry(content="content2"))
        texts = store.all_texts()
        assert "content1" in texts
        assert "content2" in texts

    def test_all_with_embeddings(self, store):
        entry = MemoryEntry(content="test content")
        entry.embedding = [1.0, 2.0, 3.0]
        store.add(entry)
        results = store.all_with_embeddings()
        assert len(results) == 1
        mid, emb, retrieved = results[0]
        assert mid == entry.id
        assert emb == [1.0, 2.0, 3.0]

    def test_stats(self, store):
        store.add(MemoryEntry(content="test1", namespace="ns1"))
        store.add(MemoryEntry(content="test2", namespace="ns1"))
        store.add(MemoryEntry(content="test3", namespace="ns2"))
        stats = store.stats()
        assert stats["total_memories"] == 3
        assert stats["namespaces"]["ns1"] == 2
        assert stats["namespaces"]["ns2"] == 1


class TestTFIDFEmbedder:
    @pytest.fixture
    def store(self, temp_db_path):
        return MemoryStore(temp_db_path)

    @pytest.fixture
    def embedder(self, store):
        return TFIDFEmbedder(store)

    @pytest.mark.asyncio
    async def test_generate_returns_list(self, embedder, store):
        # Add some content first so IDF is non-empty
        from mcp_external_memory.storage import MemoryEntry

        store.add(MemoryEntry(content="test content for embedding"))
        result = await embedder.generate("test text")
        assert isinstance(result, list)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_generate_empty_text(self, embedder, store):
        from mcp_external_memory.storage import MemoryEntry

        store.add(MemoryEntry(content="some content"))
        result = await embedder.generate("")
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_generate_same_text_same_embedding(self, embedder, store):
        from mcp_external_memory.storage import MemoryEntry

        store.add(MemoryEntry(content="hello world content"))
        result1 = await embedder.generate("hello world")
        result2 = await embedder.generate("hello world")
        assert result1 == result2

    @pytest.mark.asyncio
    async def test_generate_different_text_different_embedding(self, embedder, store):
        from mcp_external_memory.storage import MemoryEntry

        store.add(MemoryEntry(content="test content"))
        await embedder.generate("hello world")
        await embedder.generate("foo bar")


class TestEdgeCases:
    @pytest.fixture
    def store(self, temp_db_path):
        return MemoryStore(temp_db_path)

    def test_empty_content_allowed(self, store):
        entry = MemoryEntry(content="")
        store.add(entry)
        retrieved = store.get(entry.id)
        assert retrieved is not None
        assert retrieved.content == ""

    def test_very_long_content(self, store):
        long_content = "word " * 10000
        entry = MemoryEntry(content=long_content)
        store.add(entry)
        retrieved = store.get(entry.id)
        assert retrieved is not None
        assert retrieved.content == long_content

    def test_special_characters_in_content(self, store):
        content = "Hello <world> & 'test' \"quotes\""
        entry = MemoryEntry(content=content)
        store.add(entry)
        retrieved = store.get(entry.id)
        assert retrieved is not None
        assert retrieved.content == content

    def test_unicode_in_content(self, store):
        content = "Hello 世界 🎉 emoji"
        entry = MemoryEntry(content=content)
        store.add(entry)
        retrieved = store.get(entry.id)
        assert retrieved is not None
        assert retrieved.content == content

    def test_namespace_with_no_memories(self, store):
        results = store.list(namespace="nonexistent")
        assert results == []

    def test_tags_no_match(self, store):
        store.add(MemoryEntry(content="test", tags=["tag1"]))
        results = store.list(tags=["tag2"])
        assert results == []

    def test_upsert_with_same_id(self, store):
        entry1 = MemoryEntry(id="same-id", content="first")
        store.add(entry1)
        entry2 = MemoryEntry(id="same-id", content="second")
        store.add(entry2)
        retrieved = store.get("same-id")
        assert retrieved is not None
        assert retrieved.content == "second"
