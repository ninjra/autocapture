# Deterministic Memory Store

This module adds a CLI-only, local-first memory store backed by SQLite + FTS5. It is
separate from the existing retrieval pipeline and is designed for deterministic,
auditable outputs.

## Storage layout

Default root:
- Windows: `%LOCALAPPDATA%/Autocapture/memory`
- Other: `~/.autocapture/memory`

Within the root:
- `memory.sqlite3` - SQLite store
- `artifacts/` - payload files (redacted if policy triggers)
- `snapshots/` - deterministic context snapshots

Override the root with `AUTOCAPTURE_MEMORY_DIR` or `memory.storage.root_dir`.

## Determinism contract

- No wall-clock timestamps are injected by default.
- Snapshot IDs are derived from query + config hash + included IDs.
- Snapshot timestamps derive from included documents (or the epoch when none).
- Output hashing uses `context.md` + `citations.json` in a fixed order.

## CLI usage

- `autocapture memory ingest --path path/to/file.txt`
- `autocapture memory query "my search"`
- `autocapture memory compile "my search"`
- `autocapture memory items list --status active`
- `autocapture memory items propose --key foo --value bar --type fact`
- `autocapture memory promote --item-id <id> --span-id <span_id>`
- `autocapture memory verify`
- `autocapture memory gc --retention-days 90`
- `autocapture memory hotness top --as-of 2025-01-01T00:00:00Z`
- `autocapture memory hotness touch --item-id <id> --timestamp 2025-01-01T00:00:00Z`
- `autocapture memory hotness pin --item-id <id> --level hard --timestamp 2025-01-01T00:00:00Z`
- `autocapture memory hotness unpin --item-id <id> --timestamp 2025-01-01T00:00:00Z`
- `autocapture memory hotness gc --event-days 30 --event-cap 50000 --as-of 2025-01-01T00:00:00Z`
- `autocapture memory hotness state --item-id <id>`

All commands support `--json` for stable output.

## Memory hotness (opt-in)

Hotness is an additive, opt-in ranking plugin for memory items. It never changes
the default compile path unless enabled and explicitly requested.

Configuration (defaults shown):

```
memory:
  hotness:
    enabled: false
    mode_default: "off" # off|as_of|dynamic
    scope_default: "default"
    half_lives:
      fast_seconds: 3600
      mid_seconds: 21600
      warm_seconds: 86400
      cool_seconds: 604800
    weights:
      fast: 0.4
      mid: 0.3
      warm: 0.2
      cool: 0.1
    thresholds:
      hot: 0.75
      recent: 0.5
      warm: 0.25
      cool: 0.1
    quotas:
      hot: 0.4
      recent: 0.3
      warm: 0.2
      cool: 0.1
    rate_limit:
      enabled: true
      min_interval_ms: 60000
    retention:
      event_max_age_days: 30
      event_max_count: 50000
    allowed_signals:
      manual_touch: ["cli", "api"]
      pin_set: ["cli", "api"]
      pin_unset: ["cli", "api"]
```

Determinism rules:
- Hotness is off by default (`mode_default=off`).
- When enabled, the compiler requires an explicit `as_of_utc` timestamp; no wall-clock
  time is read inside the compiler.
- API `memory_hotness_mode=dynamic` sets `as_of_utc` explicitly at the API boundary.

Timestamp rules:
- Only `YYYY-MM-DDTHH:MM:SSZ` (UTC, second precision) is accepted.
- Use `--now` in CLI commands to supply a current timestamp explicitly.

Privacy:
- Hotness tables store only item IDs and minimal metadata (no queries, values, or free-form text).

Pins and truncation:
- Hard pins are selected first, then soft pins, then hotness-ranked items.
- Over-budget pins are reported in `memory_hotness.pinned_over_budget_*` metadata.

## API integration

Memory snapshots in `/api/context-pack` are enabled by default. To override:

```
memory:
  api_context_pack_enabled: true
```

## Memory roadmap (deferred work)

- [x] M1: Integrate memory snapshots into `/api/context-pack` responses.
- [ ] M2: Add API endpoints for memory ingest/query/compile.
- [ ] M3: Provide optional vector index plugin (disabled by default).
- [ ] M4: Add migration strategy if/when the store is shared across hosts.

Deferred items are tracked in `TASKS.md` and design notes; deferred work is not tracked inline in code.
