# Migrations and Backfill

Phase 0 uses additive schema changes and resumable backfills. Existing data remains queryable during upgrades.

## Migrations
- Apply schema upgrades using the existing migration workflow for your environment.
- Migrations are additive and do not remove data.

## Backfill
Use the built-in backfill command to populate new fields on existing data. The backfill is resumable and idempotent.

Common options:
- `--dry-run`: scan and report without persisting changes
- `--batch-size`: control batch size for incremental progress
- `--max-rows`: limit total rows processed in this run
- `--frame-hash-days`: compute frame_hash only for recent frames
- `--fill-monotonic`: fill monotonic_ts using captured_at ordering
- `--task`: choose specific tasks (captures, events, spans, embeddings)
- `--reset-checkpoints`: reset progress checkpoints

## Rollback
Rollback is configuration-based:
- Disable the relevant feature flags to revert to previous behavior.
- No destructive rollback is required for Phase 0.
