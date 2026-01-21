# ADR-008: Table Extractor Plugin System
Date: 2026-01-21
Status: Accepted

## Context
Table extraction needs to be optional, policy-gated, and deterministic. Providers should
be pluggable to support local-first defaults and future expansions.

## Decision
- Add a `table.extractor` plugin kind and a table extraction service.
- Default the pipeline to disabled and require explicit enablement in config.
- Enforce local-only policy by default, with an `allow_cloud` override.
- Auto-embed `*_embedding` columns during inserts and wrap writes in a single transaction.

## Consequences
- Table extraction remains opt-in and policy-safe.
- Extracted tables can be stored deterministically with optional embeddings.
