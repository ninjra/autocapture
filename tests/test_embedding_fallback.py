import importlib.util

import pytest

from autocapture.config import EmbedConfig
from autocapture.embeddings.service import EmbeddingService


def _disable_embedding_packages(monkeypatch: pytest.MonkeyPatch) -> None:
    original = importlib.util.find_spec

    def fake_find_spec(name: str, *args, **kwargs):
        if name in {"fastembed", "sentence_transformers"}:
            return None
        return original(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)


def test_embedding_fallback_in_dev(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_ENV", "dev")
    _disable_embedding_packages(monkeypatch)
    embedder = EmbeddingService(EmbedConfig(text_model="BAAI/bge-base-en-v1.5"))
    vectors = embedder.embed_texts(["hello"])
    assert len(vectors) == 1
    assert len(vectors[0]) == embedder.dim


def test_embedding_backend_missing_fails_in_prod(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("APP_ENV", raising=False)
    monkeypatch.delenv("AUTOCAPTURE_ENV", raising=False)

    def fake_find_spec(_name: str, *args, **kwargs):
        return None

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    with pytest.raises(RuntimeError, match="No embedding backend available"):
        EmbeddingService(EmbedConfig(text_model="BAAI/bge-base-en-v1.5"))
