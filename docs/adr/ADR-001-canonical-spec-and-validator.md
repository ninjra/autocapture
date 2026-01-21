# ADR-001: Canonical Spec Artifact and Validator
Date: 2026-01-21
Status: Accepted

## Context
The repo needs a stable, auditable spec artifact and a deterministic validator that can run
offline. The spec must preserve placeholders and be enforced with lightweight checks.

## Decision
- Treat `BLUEPRINT.md` as the canonical spec artifact.
- Enforce structural requirements via `tools/validate_blueprint.py` and tests.
- Keep validation stdlib-only and emit actionable diagnostics.

## Consequences
- CI can gate merges on spec validity.
- Spec edits must preserve `[MISSING_VALUE]` placeholders and required headings.
