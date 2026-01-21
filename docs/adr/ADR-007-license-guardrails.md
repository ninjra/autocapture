# ADR-007: Elastic License Guardrails
Date: 2026-01-21
Status: Accepted

## Context
We must avoid adding Elastic-licensed or SSPL-licensed dependencies or code.

## Decision
- Add `tools/license_guardrails.py` to scan dependencies and repository files.
- Enforce guardrails in CI and require explicit ADR exceptions if necessary.

## Consequences
- Elastic/SSPL artifacts are blocked unless a documented exception is approved.
- Dependency and code hygiene is enforceable in offline CI.
