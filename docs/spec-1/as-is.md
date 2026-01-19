# SPEC-1 As-Is Inventory (2026-01-19)

This inventory captures the current implementation state aligned to SPEC-1 P0 requirements. It is
based on in-repo source inspection and local command snapshots.

## Snapshot
- Branch created for discovery: `spec-1-single-machine-prod`
- HEAD: `d8f92af Merge pull request #82 from ninjra/feat/runtime-watchdog-20260119`
- Primary stack: Python (Poetry-managed), FastAPI services, SQLAlchemy + Alembic, Qdrant client,
  Prometheus + OpenTelemetry instrumentation. (`pyproject.toml:L1-L63`, `pyproject.toml:L57-L73`)

## Stage Routing + Gateway
- StageRouter selects providers per stage, supports ModelRegistry stage policies, and records
  circuit-breaker state. (`autocapture/model_ops/router.py:L38-L210`)
- ModelRegistry materializes stage candidates from provider/model/stage config entries.
  (`autocapture/model_ops/registry.py:L22-L83`)
- GatewayRouter handles stage requests with registry candidates, decode backend concurrency,
  claims JSON validation, and citation enforcement. (`autocapture/gateway/service.py:L38-L361`)
- Gateway FastAPI app exposes OpenAI-compatible endpoints plus internal stage calls and `/metrics`.
  (`autocapture/gateway/app.py:L25-L149`)

## Evidence Pack / Context Pack
- ContextPack defines EvidenceItem/EvidenceSpan, sanitizes prompt-injection patterns, and renders
  JSON/TRON evidence payloads. (`autocapture/memory/context_pack.py:L49-L138`)
- API `/api/context-pack` builds context packs for retrieval results and memory snapshots.
  (`autocapture/api/server.py:L1486-L1578`)
- API `/api/answer` builds context packs and forwards to AnswerGraph for answer generation.
  (`autocapture/api/server.py:L1579-L1707`)

## Deterministic Verification + Citations
- ClaimValidator enforces deterministic claim-level citation rules.
  (`autocapture/answer/claim_validation.py:L1-L120`)
- Claims JSON parsing assigns stable claim IDs and renders citations inline.
  (`autocapture/answer/claims.py:L15-L101`)
- AnswerGraph validates citations, builds deterministic line maps, and persists evidence lineage.
  (`autocapture/agents/answer_graph.py:L1525-L1900`)
- AnswerGraph verifies claim-level JSON with citation validator and verifier hooks.
  (`autocapture/agents/answer_graph.py:L1525-L1620`)
- AnswerGraph maintains line offsets deterministically for evidence text.
  (`autocapture/agents/answer_graph.py:L1817-L1842`)
- Citation integrity checks verify span presence, provenance ledger, and media existence.
  (`autocapture/answer/integrity.py:L26-L74`)

## Entailment Verification
- Heuristic entailment compares claim numerics to evidence; LLM judge uses StageRouter for strict
  verdicts. (`autocapture/answer/entailment.py:L1-L156`)
- AnswerGraph applies entailment gating with bounded retries and regeneration logic.
  (`autocapture/agents/answer_graph.py:L1562-L1759`)

## Retrieval Stack
- RetrievalService supports lexical + vector + sparse + late interaction retrieval, RRF fusion,
  and graph adapters with tiered budgets. (`autocapture/memory/retrieval.py:L91-L1040`)
- LexicalIndex provides SQLite FTS5-backed event and span indexes.
  (`autocapture/indexing/lexical_index.py:L31-L214`)
- VectorIndex uses Qdrant backend with circuit breaker and collection management.
  (`autocapture/indexing/vector_index.py:L253-L520`)
- SpansV2Index supports dense/sparse/late vectors backed by Qdrant collections.
  (`autocapture/indexing/spans_v2.py:L32-L217`)

## Observability
- OpenTelemetry helpers include safe attribute allowlist, tracer/meter setup, and spans.
  (`autocapture/observability/otel.py:L1-L170`)
- Prometheus metrics cover gateway, graph, retrieval, verification, and system health.
  (`autocapture/observability/metrics.py:L1-L214`)
- API server initializes OTel on startup. (`autocapture/api/server.py:L395-L407`)

## API Services + Entrypoints
- CLI entrypoint defines subcommands for API, gateway, and graph worker services.
  (`autocapture/main.py:L36-L602`)
- Local API app created via `create_app` with retrieval + AnswerGraph wiring.
  (`autocapture/api/server.py:L395-L538`)
- Graph worker FastAPI app exposes `/index`, `/query`, and `/metrics`.
  (`autocapture/graph/app.py:L27-L146`)

## Config System
- AppConfig composes service configs (gateway, registry, verification, qdrant, observability).
  (`autocapture/config.py:L1244-L1575`)
- Config loader reads YAML and applies compatibility defaults.
  (`autocapture/config.py:L1642-L1705`)
- Example configs exist in `autocapture.yml` and `config/example.yml`.
  (`autocapture.yml:L1-L260`, `config/example.yml:L1-L360`)
- Prometheus scrape config includes API, gateway, and graph metrics endpoints.
  (`infra/prometheus.yml:L1-L19`)

## Persistence Layer + Migrations
- DatabaseManager sets SQLite pragmas, runs Alembic migrations, and manages sessions.
  (`autocapture/storage/database.py:L25-L214`)
- Alembic env defines migration context for SQLAlchemy models.
  (`alembic/env.py:L1-L52`)
- Alembic versions include SPEC-1 gateway claims migration and run logging schema.
  (`alembic/versions/0013_spec1_gateway_claims.py:L1-L119`,
   `alembic/versions/0014_spec1_runs_and_citations.py:L1-L150`)

## Tests + CI Entry Points
- Claim validation tests. (`tests/test_claim_validation.py:L1-L35`)
- Citation handling tests for AnswerGraph and API response payloads.
  (`tests/test_answer_citations.py:L1-L155`)
- Entailment tests (heuristic + judge). (`tests/test_entailment.py:L1-L60`)
- Gateway service validation tests. (`tests/test_gateway_service.py:L1-L102`)
- Citation integrity checks tests. (`tests/test_integrity_checker.py:L1-L88`)
- CI uses Poetry, `make check`, and gate scripts. (`.github/workflows/ci.yml:L1-L133`)
- Local checks defined via Make targets (black, ruff, pytest). (`Makefile:L1-L30`)
- Pytest markers and deps in `pytest.ini` + `pyproject.toml`.
  (`pytest.ini:L1-L3`, `pyproject.toml:L57-L73`)

## Command Snapshots (Discovery)
```
$ rg -n "FastAPI\(" autocapture
autocapture/api/server.py:531:    app = FastAPI(**app_kwargs)
autocapture/graph/app.py:33:    app = FastAPI(title="Autocapture Graph Workers")
autocapture/gateway/app.py:31:    app = FastAPI(title="Autocapture LLM Gateway")
```

```
$ rg -n "StageRouter" autocapture/model_ops/router.py
38:class StageRouter:
```

```
$ nvidia-smi
Mon Jan 19 12:36:45 2026
| NVIDIA-SMI 580.82.10              Driver Version: 581.29         CUDA Version: 13.0     |
|   0  NVIDIA GeForce RTX 4090        On  |   00000000:01:00.0  On |                  Off |
```
