# AutoCapture Critical Memory Canonical

## Project Identity
- AutoCapture is a local-first personal activity memory engine for Windows 11 + WSL2.
- Defaults prioritize privacy, GPU performance, deterministic outputs, and citations.

## Non-Negotiables
- Patch/append only; do not replace whole files unless new.
- Frozen surfaces must be unfrozen before edits and refrozen after with reasons.
- Tests and docs must be updated when behavior or workflows change.
- Local-first defaults; cloud usage requires explicit, per-stage opt-in.

## Prohibited Content
- Secrets, keys, tokens, credentials, or private identifiers.
- PII or medical data (emails, phone numbers, addresses).
- Machine-specific paths or hostnames in repo memory.

## Agent Lifecycle
1) Before work: read `.memory/CANONICAL.md` and `.memory/STATE.json`.
2) During work: record new constraints/lessons/decisions as staged ledger entries.
3) End of work: append ledger entries safely, rebuild `STATE.json`, update `DECISIONS.md` when needed.

## Conflict Resolution (ADR)
- Use ADRs in `.memory/DECISIONS.md` to resolve conflicts.
- New ADRs must be linked to ledger entries and capture rationale.

## Definition of Done
- Stable UI, deterministic behavior, secure storage, and passing tests.
