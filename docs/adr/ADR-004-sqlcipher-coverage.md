# ADR-004: SQLCipher Coverage for SQLite Stores
Date: 2026-01-21
Status: Accepted

## Context
SQLite is now the default backend for vector, spans_v2, and memory service storage.
At-rest encryption must remain available for these paths.

## Decision
- Route SQLite backends through `DatabaseManager` so SQLCipher PRAGMA keying applies.
- Add SQLCipher E2E tests for vector, spans_v2, and memory service stores.
- Keep Postgres opt-in for memory service via `memory_service.database_url`.

## Consequences
- Local-first storage can be encrypted consistently via SQLCipher.
- SQLCipher remains an optional dependency but is exercised in CI when available.
