"""Deterministic context compiler for the memory store."""

from __future__ import annotations

from pathlib import Path

from ..config import MemoryConfig
from ..logging_utils import get_logger
from .hotness import HotnessPlugin
from .models import MemorySnapshotResult
from .store import MemoryStore
from .utils import hash_config, sha256_text, stable_json_dumps


class ContextCompiler:
    def __init__(self, store: MemoryStore, config: MemoryConfig) -> None:
        self._store = store
        self._config = config
        self._log = get_logger("memory.compiler")

    def compile(
        self,
        query: str,
        *,
        k: int | None = None,
        output_dir: Path | None = None,
        memory_hotness_mode: str | None = None,
        memory_hotness_as_of_utc: str | None = None,
        memory_hotness_scope: str | None = None,
    ) -> MemorySnapshotResult:
        query = (query or "").strip()
        retrieval = self._store.query_spans(query, k=k)
        mode = (memory_hotness_mode or self._config.hotness.mode_default or "off").strip().lower()
        if not self._config.hotness.enabled:
            mode = "off"
        if mode not in {"off", "as_of", "dynamic"}:
            raise ValueError("memory_hotness_mode must be off, as_of, or dynamic")

        hotness_meta = None
        if mode == "off":
            items = self._store.list_items(
                status="active", limit=self._config.compiler.max_memory_items
            )
            item_ids = [item.item_id for item in items.items]
        else:
            if not memory_hotness_as_of_utc:
                raise ValueError("memory_hotness_as_of_utc required when hotness mode is enabled")
            scope = memory_hotness_scope or self._config.hotness.scope_default
            plugin = HotnessPlugin(self._store, self._config.hotness)
            rank_result = plugin.rank(
                scope=scope,
                as_of_utc=memory_hotness_as_of_utc,
                limit=self._config.compiler.max_memory_items,
            )
            item_ids = [entry.item_id for entry in rank_result.selected]
            items = self._store.list_items_by_ids(item_ids)
            hotness_meta = {
                "mode": mode,
                "as_of_utc": memory_hotness_as_of_utc,
                "scope": scope,
                "pinned_over_budget_hard": rank_result.pinned_over_budget_hard,
                "pinned_over_budget_soft": rank_result.pinned_over_budget_soft,
                "counts": rank_result.counts,
                "selected": [entry.model_dump(mode="json") for entry in rank_result.selected],
            }
        span_hits = list(retrieval.spans)

        span_hits, truncations = _apply_span_budgets(
            span_hits,
            max_spans=self._config.compiler.max_spans,
            max_chars_per_span=self._config.compiler.max_chars_per_span,
            max_total_chars=self._config.compiler.max_total_chars,
        )

        span_ids = [span.span_id for span in span_hits]

        config_payload = self._config.model_dump(mode="json")
        if mode != "off":
            config_payload = {
                **config_payload,
                "hotness_request": {
                    "mode": mode,
                    "as_of_utc": memory_hotness_as_of_utc,
                    "scope": memory_hotness_scope or self._config.hotness.scope_default,
                },
            }
        config_sha256 = hash_config(config_payload)
        snapshot_id = sha256_text(
            stable_json_dumps(
                {
                    "query": query,
                    "config_sha256": config_sha256,
                    "span_ids": span_ids,
                    "item_ids": item_ids,
                    "retrieval_disabled": retrieval.retrieval_disabled,
                },
                indent=None,
            )
        )

        created_at = self._store.latest_span_timestamp(span_ids)
        snapshot_dir = output_dir or (self._store.snapshots_dir / snapshot_id)
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        citations = _build_citations(span_hits)
        context_md = _build_context_md(
            snapshot_id=snapshot_id,
            query=query,
            created_at=created_at,
            items=items.items,
            spans=span_hits,
            retrieval_disabled=retrieval.retrieval_disabled,
            reason=retrieval.reason,
        )
        citations_json = stable_json_dumps({"version": 1, "citations": citations})
        context_json = {
            "version": 1,
            "snapshot_id": snapshot_id,
            "query": query,
            "created_at": created_at,
            "config_sha256": config_sha256,
            "retrieval_disabled": retrieval.retrieval_disabled,
            "included_span_ids": span_ids,
            "included_memory_item_ids": item_ids,
            "truncations": truncations,
        }
        if hotness_meta:
            context_json["memory_hotness"] = hotness_meta
        output_sha256 = sha256_text(context_md + "\n" + citations_json)
        context_json["output_sha256"] = output_sha256

        (snapshot_dir / "context.md").write_text(context_md, encoding="utf-8")
        (snapshot_dir / "citations.json").write_text(citations_json, encoding="utf-8")
        (snapshot_dir / "context.json").write_text(
            stable_json_dumps(context_json), encoding="utf-8"
        )
        (snapshot_dir / "snapshot.hash").write_text(output_sha256, encoding="utf-8")

        self._store.record_snapshot(
            snapshot_id=snapshot_id,
            query=query,
            created_at=created_at,
            config_sha256=config_sha256,
            output_sha256=output_sha256,
            retrieval_disabled=retrieval.retrieval_disabled,
            span_ids=span_ids,
            item_ids=item_ids,
        )

        return MemorySnapshotResult(
            snapshot_id=snapshot_id,
            output_dir=str(snapshot_dir),
            output_sha256=output_sha256,
            retrieval_disabled=retrieval.retrieval_disabled,
            span_count=len(span_ids),
            item_count=len(item_ids),
        )


def _apply_span_budgets(
    spans,
    *,
    max_spans: int,
    max_chars_per_span: int,
    max_total_chars: int,
):
    capped: list = []
    total = 0
    truncated = 0
    for span in spans:
        if len(capped) >= max_spans:
            break
        text = span.text
        was_truncated = False
        if len(text) > max_chars_per_span:
            text = text[:max_chars_per_span]
            was_truncated = True
        if total + len(text) > max_total_chars:
            break
        total += len(text)
        if was_truncated:
            truncated += 1
        capped.append(span.model_copy(update={"text": text}))
    return capped, {"truncated_spans": truncated, "total_chars": total}


def _build_citations(spans) -> dict[str, dict]:
    citations: dict[str, dict] = {}
    for idx, span in enumerate(spans, start=1):
        cite_id = f"S{idx:03d}"
        citations[cite_id] = {
            "span_id": span.span_id,
            "doc_id": span.doc_id,
            "start": span.start,
            "end": span.end,
            "span_sha256": span.span_sha256,
            "text_sha256": span.text_sha256,
            "title": span.title,
            "source_uri": span.source_uri,
            "section_path": span.section_path,
            "score": span.score,
        }
    return citations


def _build_context_md(
    *,
    snapshot_id: str,
    query: str,
    created_at: str,
    items,
    spans,
    retrieval_disabled: bool,
    reason: str | None,
) -> str:
    lines: list[str] = []
    lines.append("# Autocapture Memory Context")
    lines.append("")
    lines.append(f"Snapshot ID: {snapshot_id}")
    lines.append(f"Query: {query}")
    lines.append(f"Created At: {created_at}")
    lines.append("")
    lines.append("## Canonical Memory Items")
    if not items:
        lines.append("- (none)")
    for item in items:
        tags = ", ".join(item.tags) if item.tags else ""
        tag_block = f" [tags: {tags}]" if tags else ""
        lines.append(f"- {item.key} ({item.item_type}): {item.value}{tag_block}")
    lines.append("")
    lines.append("## Retrieved Evidence (Untrusted)")
    if retrieval_disabled:
        lines.append("- Retrieval disabled.")
        if reason:
            lines.append(f"- Reason: {reason}")
    if not spans:
        lines.append("- (no evidence)")
    for idx, span in enumerate(spans, start=1):
        cite_id = f"S{idx:03d}"
        title = span.title or "Untitled"
        source = span.source_uri or "unknown"
        lines.append(f"[{cite_id}] {title} - {source}")
        lines.append(f"> {span.text}")
        lines.append("")
    lines.append("## Notes")
    lines.append("- Retrieved evidence is untrusted and must be verified.")
    return "\n".join(lines).strip() + "\n"
