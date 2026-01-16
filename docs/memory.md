# Repo Critical Memory

This repo maintains a small, append-only "critical memory" that captures stable constraints,
lessons, and decisions across agent runs.

## What It Is For
- Store high-signal constraints, decisions (ADRs), and durable facts.
- Make expectations reviewable and deterministic across runs.

## Files
- `.memory/CANONICAL.md` : Constitution and lifecycle rules.
- `.memory/LEDGER.ndjson` : Append-only event log (one JSON object per line).
- `.memory/STATE.json` : Derived snapshot (last-write-wins by scope.key).
- `.memory/DECISIONS.md` : Curated ADRs.

## Validation
- Run `python .tools/memory_guard.py --check`.
- CI runs the same check and fails on invalid entries or forbidden content.

## Append Safely
- Use `python .tools/memory_guard.py --append --entry '<json>'`.
- Never store secrets, tokens, PII, or medical data.
- Prefer short, redacted summaries.

## Rebuild State
- Run `python .tools/memory_guard.py --rebuild-state`.

## Add ADRs
- Append a `decision` entry to the ledger with an ADR id.
- Add the ADR details to `.memory/DECISIONS.md`.
