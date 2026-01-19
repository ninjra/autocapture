from __future__ import annotations

from pathlib import Path

import pytest

from autocapture.config import (
    MemoryCompilerConfig,
    MemoryConfig,
    MemoryPolicyConfig,
    MemoryRetrievalConfig,
    MemorySpanConfig,
    MemoryStorageConfig,
)
from autocapture.memory.compiler import ContextCompiler
from autocapture.memory.models import ArtifactMeta
from autocapture.memory.store import MemoryStore


def _memory_config(tmp_path: Path, *, policy: MemoryPolicyConfig | None = None) -> MemoryConfig:
    storage = MemoryStorageConfig(root_dir=tmp_path)
    return MemoryConfig(
        storage=storage,
        policy=policy or MemoryPolicyConfig(),
        spans=MemorySpanConfig(max_chars=400, min_chars=50),
        retrieval=MemoryRetrievalConfig(default_k=5, max_k=10),
        compiler=MemoryCompilerConfig(max_total_chars=2000, max_chars_per_span=300, max_spans=5),
    )


def test_memory_ingest_query_compile_deterministic(tmp_path: Path) -> None:
    config = _memory_config(tmp_path)
    store = MemoryStore(config)
    meta = ArtifactMeta(source_uri="file://test.txt", title="Test Doc")
    store.ingest_text("Alpha\n\nBeta Gamma", meta)

    if not store.fts_available:
        pytest.skip("FTS5 unavailable")

    query = store.query_spans("Beta", k=3)
    assert query.spans

    compiler = ContextCompiler(store, config)
    first = compiler.compile("Beta")
    second = compiler.compile("Beta")

    assert first.snapshot_id == second.snapshot_id
    first_context = (Path(first.output_dir) / "context.md").read_text(encoding="utf-8")
    second_context = (Path(second.output_dir) / "context.md").read_text(encoding="utf-8")
    assert first_context == second_context


def test_memory_policy_redact_and_exclude(tmp_path: Path) -> None:
    policy = MemoryPolicyConfig(
        exclude_patterns=["secret"],
        redact_patterns=["token"],
        redact_token="[REDACTED]",
    )
    config = _memory_config(tmp_path, policy=policy)
    store = MemoryStore(config)
    meta = ArtifactMeta(source_uri="stdin", title="Policy Doc")

    excluded = store.ingest_text("this is a SECRET", meta)
    assert excluded.excluded

    redacted = store.ingest_text("token=abcd", meta)
    assert redacted.redacted
    artifact_path = store.artifacts_dir / f"{redacted.artifact_id}.txt"
    payload = artifact_path.read_text(encoding="utf-8")
    assert "[REDACTED]" in payload


def test_memory_item_promotion_requires_citation(tmp_path: Path) -> None:
    config = _memory_config(tmp_path)
    store = MemoryStore(config)
    meta = ArtifactMeta(source_uri="stdin", title="Citation Doc")
    ingest = store.ingest_text("Alpha\n\nBeta", meta)
    assert ingest.span_ids

    item = store.propose_item(key="favorite_color", value="blue", item_type="fact")
    with pytest.raises(ValueError):
        store.promote_item(item_id=item.item_id)

    promoted = store.promote_item(item_id=item.item_id, span_ids=[ingest.span_ids[0]])
    assert promoted.status == "active"


def test_memory_verify_detects_tamper(tmp_path: Path) -> None:
    config = _memory_config(tmp_path)
    store = MemoryStore(config)
    meta = ArtifactMeta(source_uri="stdin", title="Verify Doc")
    ingest = store.ingest_text("Alpha\n\nBeta", meta)
    compiler = ContextCompiler(store, config)
    compiler.compile("Alpha")

    artifact_path = store.artifacts_dir / f"{ingest.artifact_id}.txt"
    artifact_path.write_text("tampered", encoding="utf-8")
    result = store.verify()
    assert not result.ok
    assert any("artifact checksum mismatch" in err for err in result.errors)
