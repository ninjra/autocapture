# ADR-003: SQLite Default Vector and Spans Backends
Date: 2026-01-21
Status: Accepted

## Context
Vector and spans_v2 retrieval must be local-first and not depend on external services by
default. Qdrant remains supported but should be opt-in.

## Decision
- Introduce routing knobs `routing.vector_backend` and `routing.spans_v2_backend`.
- Default routing to `local` (SQLite) for both backends.
- Keep Qdrant available via `routing.* = qdrant` and disable Qdrant by default in manifests.
- Log resolved backend routing at startup to make defaults visible.

## Consequences
- Local SQLite storage is the default retrieval backend.
- Deployments relying on Qdrant must opt in explicitly via routing config.
