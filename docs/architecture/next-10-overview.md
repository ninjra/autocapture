# Next-10 Overview (SPEC-260117)

This document maps SPEC-260117 requirements to repo modules and artifacts.

## Core contracts + storage
- Contracts: `autocapture/contracts_next10.py`
- Canonical JSON + hashing: `autocapture/contracts_utils.py`
- ORM tables + migrations: `autocapture/storage/models.py`, `alembic/versions/0012_next10_contracts.py`
- Ledger writer + validation: `autocapture/storage/ledger.py`

## Capture + privacy invariants
- Capture orchestration: `autocapture/capture/orchestrator.py`
- Privacy policy + masking: `autocapture/capture/privacy.py`, `autocapture/capture/privacy_filter.py`
- Frame records + privacy flags: `autocapture/capture/frame_record.py`
- Media encryption: `autocapture/media/store.py`, `autocapture/encryption.py`
- Secure-mode gate: `autocapture/storage/database.py`

## Extraction + evidence atoms
- OCR ingest: `autocapture/worker/event_worker.py`
- Citable spans + artifacts: `autocapture/storage/models.py` + `autocapture/storage/ledger.py`

## Retrieval tiers + budgets
- Tier planner + orchestrator: `autocapture/memory/tiered_retrieval.py`
- Budget manager: `autocapture/runtime/budgets.py`
- Tier stats: `autocapture/storage/models.py`

## Answer orchestration
- Answer pipeline: `autocapture/agents/answer_graph.py`
- Coverage + sentence splitting: `autocapture/answer/coverage.py`
- Conflict detection: `autocapture/answer/conflict.py`
- Citation integrity: `autocapture/answer/integrity.py`

## CI gates + reports
- Privacy regression scanner: `tools/privacy_scanner.py`
- Retrieval sensitivity ±1: `tools/retrieval_sensitivity.py`
- Coverage/latency/provenance gates: `tools/coverage_gate.py`, `tools/latency_gate.py`, `tools/provenance_gate.py`
- Pillar gate: `tools/pillar_gate.py`

## Config defaults
- Policy: `config/defaults/policy.json`
- Budgets: `config/defaults/budgets.json`
- Tier knobs: `config/defaults/tiers.json`

## UI/UX
- Web UI: `autocapture/ui/web/app.js`
- API contract: `autocapture/api/server.py`

