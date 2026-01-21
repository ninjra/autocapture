# ADR-002: Grounded Schema SQL Artifacts
Date: 2026-01-21
Status: Accepted

## Context
The spec references explicit DDL that must be grounded in the repo for validation and
migration auditing. We must avoid inferring schema details that are not specified.

## Decision
- Add SQL schema artifacts under `docs/schemas/` for explicit DDL only.
- Annotate each file with the source MOD/section references and excluded placeholders.
- Add tests that execute each SQL file against in-memory SQLite and verify expected objects.

## Consequences
- Schema coverage is traceable to the spec and testable offline.
- Unspecified tables remain unimplemented until explicitly defined in the spec.
