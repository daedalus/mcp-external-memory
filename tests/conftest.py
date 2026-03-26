import shutil
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def temp_db_path() -> Path:
    tmpdir = tempfile.mkdtemp()
    db_path = Path(tmpdir) / "test.db"
    yield db_path
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def mock_env(monkeypatch):
    monkeypatch.setenv("MEMORY_DB_PATH", "")
    monkeypatch.setenv("MEMORY_EMBED_BACKEND", "tfidf")
    return monkeypatch
