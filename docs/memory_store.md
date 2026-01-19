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

All commands support `--json` for stable output.

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

Deferred items are tracked with TODO markers and (where appropriate) skipped tests.
