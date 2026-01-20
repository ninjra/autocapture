from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from autocapture.config import (
    MemoryCompilerConfig,
    MemoryConfig,
    MemoryHotnessConfig,
    MemoryHotnessHalfLivesConfig,
    MemoryHotnessRateLimitConfig,
    MemoryHotnessWeightsConfig,
    MemoryPolicyConfig,
    MemoryRetrievalConfig,
    MemorySpanConfig,
    MemoryStorageConfig,
)
from autocapture.memory.compiler import ContextCompiler
from autocapture.memory.models import ArtifactMeta
from autocapture.memory.hotness import HotnessPlugin
from autocapture.memory.store import MemoryStore


def _timestamp(offset_seconds: int) -> str:
    base = dt.datetime(2025, 1, 1, tzinfo=dt.timezone.utc)
    return (base + dt.timedelta(seconds=offset_seconds)).strftime("%Y-%m-%dT%H:%M:%SZ")


def _memory_config(tmp_path: Path, *, hotness: MemoryHotnessConfig | None = None) -> MemoryConfig:
    storage = MemoryStorageConfig(root_dir=tmp_path, require_fts=False)
    return MemoryConfig(
        storage=storage,
        policy=MemoryPolicyConfig(),
        spans=MemorySpanConfig(max_chars=400, min_chars=50),
        retrieval=MemoryRetrievalConfig(default_k=5, max_k=10),
        compiler=MemoryCompilerConfig(max_total_chars=2000, max_chars_per_span=300, max_spans=5),
        hotness=hotness or MemoryHotnessConfig(enabled=True),
    )


def _create_item(store: MemoryStore, key: str) -> str:
    item = store.propose_item(key=key, value=f"value:{key}", item_type="fact")
    store.promote_item(item_id=item.item_id, user_asserted=True)
    return item.item_id


def test_hotness_decay_half_life_bins(tmp_path: Path) -> None:
    hotness = MemoryHotnessConfig(
        enabled=True,
        half_lives=MemoryHotnessHalfLivesConfig(
            fast_seconds=10,
            mid_seconds=10,
            warm_seconds=10,
            cool_seconds=10,
        ),
        weights=MemoryHotnessWeightsConfig(fast=1.0, mid=0.0, warm=0.0, cool=0.0),
    )
    config = _memory_config(tmp_path, hotness=hotness)
    store = MemoryStore(config)
    item_id = _create_item(store, "alpha")
    plugin = HotnessPlugin(store, config.hotness)

    t0 = _timestamp(0)
    plugin.touch(
        scope="default",
        item_id=item_id,
        ts_utc=t0,
        signal="manual_touch",
        weight=1.0,
        source="cli",
    )
    t1 = _timestamp(10)
    ranked = plugin.rank(scope="default", as_of_utc=t1, limit=1)
    assert ranked.selected
    entry = ranked.selected[0]
    assert entry.item_id == item_id
    assert entry.score == pytest.approx(0.5, abs=1e-4)


def test_hotness_pins_override_budget_deterministic(tmp_path: Path) -> None:
    config = _memory_config(tmp_path)
    store = MemoryStore(config)
    plugin = HotnessPlugin(store, config.hotness)
    ts = _timestamp(0)
    item_ids = [_create_item(store, f"item-{idx}") for idx in range(5)]
    for idx, item_id in enumerate(item_ids):
        plugin.pin(
            scope="default",
            item_id=item_id,
            level="hard",
            rank=idx,
            ts_utc=ts,
            source="cli",
        )

    ranked = plugin.rank(scope="default", as_of_utc=ts, limit=3)
    assert [entry.item_id for entry in ranked.selected] == item_ids[:3]
    assert ranked.pinned_over_budget_hard == 2


def test_compile_hotness_as_of_deterministic(tmp_path: Path) -> None:
    config = _memory_config(tmp_path)
    store = MemoryStore(config)
    if not store.fts_available:
        pytest.skip("FTS5 unavailable")
    store.ingest_text("Alpha\n\nBeta Gamma", meta=ArtifactMeta(source_uri="stdin"))
    item_id = _create_item(store, "beta")
    plugin = HotnessPlugin(store, config.hotness)
    ts = _timestamp(0)
    plugin.touch(
        scope="default",
        item_id=item_id,
        ts_utc=ts,
        signal="manual_touch",
        weight=1.0,
        source="cli",
    )
    compiler = ContextCompiler(store, config)
    first = compiler.compile(
        "Beta",
        memory_hotness_mode="as_of",
        memory_hotness_as_of_utc=ts,
    )
    second = compiler.compile(
        "Beta",
        memory_hotness_mode="as_of",
        memory_hotness_as_of_utc=ts,
    )
    assert first.snapshot_id == second.snapshot_id
    first_context = (Path(first.output_dir) / "context.md").read_text(encoding="utf-8")
    second_context = (Path(second.output_dir) / "context.md").read_text(encoding="utf-8")
    assert first_context == second_context


def test_hotness_gc_event_cap_deterministic(tmp_path: Path) -> None:
    hotness = MemoryHotnessConfig(
        enabled=True, rate_limit=MemoryHotnessRateLimitConfig(enabled=False)
    )
    config = _memory_config(tmp_path, hotness=hotness)
    store = MemoryStore(config)
    plugin = HotnessPlugin(store, config.hotness)
    item_id = _create_item(store, "gamma")
    for idx in range(5):
        plugin.touch(
            scope="default",
            item_id=item_id,
            ts_utc=_timestamp(idx),
            signal="manual_touch",
            weight=1.0,
            source="cli",
        )
    result = plugin.gc(scope="default", as_of_utc=_timestamp(10), max_events=3)
    assert result.events_remaining == 3
    with store._connect() as conn:
        rows = conn.execute(
            """
            SELECT ts_utc
            FROM memory_hotness_events
            WHERE scope = ?
            ORDER BY ts_utc DESC, event_id DESC
            """,
            ("default",),
        ).fetchall()
    remaining = [row["ts_utc"] for row in rows]
    assert remaining == [_timestamp(4), _timestamp(3), _timestamp(2)]


def test_hotness_schema_privacy_columns(tmp_path: Path) -> None:
    config = _memory_config(tmp_path)
    store = MemoryStore(config)
    with store._connect() as conn:
        events_cols = [
            row["name"] for row in conn.execute("PRAGMA table_info(memory_hotness_events)")
        ]
        state_cols = [
            row["name"] for row in conn.execute("PRAGMA table_info(memory_hotness_state)")
        ]
        pins_cols = [row["name"] for row in conn.execute("PRAGMA table_info(memory_hotness_pins)")]
    assert events_cols == [
        "event_id",
        "scope",
        "item_id",
        "ts_utc",
        "signal",
        "weight",
        "source",
    ]
    assert state_cols == [
        "scope",
        "item_id",
        "last_ts_utc",
        "h_fast",
        "h_mid",
        "h_warm",
        "h_cool",
    ]
    assert pins_cols == ["scope", "item_id", "pin_level", "pin_rank", "set_ts_utc"]
