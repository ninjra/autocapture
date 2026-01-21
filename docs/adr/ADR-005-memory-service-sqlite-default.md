# ADR-005: Memory Service SQLite Default Store
Date: 2026-01-21
Status: Accepted

## Context
The Memory Service should run offline by default without requiring Postgres.

## Decision
- Default the Memory Service to a separate SQLite DB file when `memory_service.database_url`
  is unset (`${data_dir}/memory_service.db`).
- Keep Postgres opt-in via explicit database URL.
- Preserve deterministic retrieval and policy checks across storage backends.

## Consequences
- Memory Service can run locally without external infrastructure.
- Postgres deployments remain supported by setting the database URL explicitly.
