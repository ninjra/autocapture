# Decisions

## ADR-0001: Adopt event-sourced repo memory
Status: Accepted

Context:
- Repo needs durable, reviewable memory across agent runs.
- Decisions must be explicit and audit-friendly.

Decision:
- Use an append-only NDJSON ledger with a derived STATE snapshot and curated ADRs.

Consequences:
- Memory entries are validated in CI.
- Changes must be appended via the memory guard tooling.
