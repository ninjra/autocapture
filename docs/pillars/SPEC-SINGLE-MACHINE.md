# SPEC-SINGLE-MACHINE: Single-Machine Production Stack

This pillar documents the SPEC-SINGLE-MACHINE commitment for a production-ready, single-machine
stack that includes an OpenAI-compatible gateway, deterministic claim-level
citations with validation and entailment gates, retrieval fusion (lexical + vector
+ graph adapters), and first-class observability (OTel + Prometheus).

Key invariants:
- Gateway enforces stage policies and deterministic JSON schema outputs.
- Final answers require claim-level citations with line-range validation.
- Retrieval integrates SQLite FTS, Qdrant, and graph adapters with RRF fusion.
- Observability emits spans and metrics for all major stages and gates.
