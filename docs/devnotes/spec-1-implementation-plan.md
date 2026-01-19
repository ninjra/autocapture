# SPEC-1 implementation plan (2026-01-19)

## Scope
- Priority >=2 production architecture upgrades (WSL2-first deployment, multi-instance vLLM, in-repo LLM gateway, Qdrant + graph adapters, claim-level citations + deterministic validation + entailment gate, OTel + Prometheus + optional Loki).
- Reuse existing routing, evidence pack, and observability hooks where possible; keep diffs minimal.

## Repo evidence baseline (paths)
- Stage routing: `autocapture/model_ops/router.py` (StageRouter), used in `autocapture/agents/answer_graph.py` and `autocapture/api/server.py`.
- LLM providers: `autocapture/llm/providers.py` (+ built-in factories in `autocapture/plugins/builtin/factories.py`).
- Context packs + sanitizer: `autocapture/memory/context_pack.py`.
- Retrieval + RRF + late-stage: `autocapture/memory/retrieval.py` and Qdrant client in `autocapture/indexing/vector_index.py`.
- Citation integrity + provenance: `autocapture/answer/coverage.py`, `autocapture/answer/integrity.py`, `autocapture/answer/provenance.py`.
- Observability: `autocapture/observability/otel.py`, `autocapture/observability/metrics.py`.
- DB + migrations: `autocapture/storage/database.py`, `alembic/`, `alembic.ini`.
- Qdrant sidecar + compose: `autocapture/qdrant/sidecar.py`, `docker-compose.yml`.
- Config system: `autocapture/config.py`, `autocapture.yml`, `config/example.yml`.

## Gaps vs SPEC-1 Priority >=2 (initial baseline)
- No dedicated LLM gateway or OpenAI-compatible proxy service.
- No model registry or stage policy with fallback chains (StageRouter selects a single provider today).
- No GraphRAG/HyperGraphRAG/Hyper-RAG adapters present.
- No claim-level `claims_json_v1` schema or entailment gate (rules verifier and citation integrity exist, but no structured claim enforcement).

## Target component mapping (planned locations)
- LLM Gateway (OpenAI-compatible proxy + internal stage endpoint): new `autocapture/gateway/` package with FastAPI app, and a CLI entry in `autocapture/main.py`.
  - Decision: standalone app on its own port.
- Model registry + StagePolicy + fallback router:
  - New config models (e.g., `autocapture/model_ops/registry.py`) and wiring in `autocapture/config.py`.
  - Stage router upgrade in `autocapture/model_ops/router.py` to use registry + deterministic fallback chain.
- Retrieval hardening + graph adapters:
  - Extend `autocapture/memory/retrieval.py` with deterministic fusion tie-breaks (already partially present).
  - Add graph adapters (GraphRAG, HyperGraphRAG, Hyper-RAG) in `autocapture/memory/graph_adapters.py`, config-gated.
- Claim-level citations + deterministic validator + entailment gate:
  - Extend `autocapture/agents/answer_graph.py` to emit/validate `claims_json_v1` and enforce entailment policy.
  - Add validator modules (e.g., `autocapture/answer/claims.py`, `autocapture/answer/entailment.py`) that reuse existing provenance + citation checks.
- Observability updates:
  - Add spans around gateway calls, verification, and retrieval graph adapters.
  - Add low-cardinality metrics in `autocapture/observability/metrics.py`.
- Deployment skeleton:
  - Update `docker-compose.yml` to bind services to `127.0.0.1` by default.
  - Add Prometheus (and optional Loki profile) with localhost-only defaults.
  - Add scripts under `scripts/` to run gateway and vLLM instances (documented, not executed in tests).

## Config and safety plan
- Extend `autocapture/config.py` with Pydantic models:
  - ProviderSpec, ModelSpec, StagePolicy, DecodeConfig, CircuitBreakerConfig, GatewayConfig.
- Add config examples to `autocapture.yml` and `config/example.yml`; defaults keep new features disabled unless explicitly configured.
- Enforce localhost-only defaults for gateway/vLLM/Qdrant and preserve offline/privacy gates.
- Ensure deterministic routing: stable fallback order, fixed seed usage where applicable, and stable tie-breaks.

## Acceptance tests (new)
- Unit tests (pytest):
  - Config validation (unknown stage/model/provider, fallback cycles).
  - Gateway routing + fallback with mocked upstreams (timeouts/5xx).
  - Citation validator + claim schema enforcement.
  - Deterministic RRF ordering and sanitizer redaction stability.
  - Entailment policy (contradicted blocks, NEI policy behavior).
- Integration tests (opt-in):
  - Qdrant upsert/query persistence via docker when available.
  - `/metrics` endpoint exposes new metrics.

## Sequencing (PR-sized)
1) Add spec anchor + plan docs (this file only; method text not provided).
2) Add registry + StagePolicy config models; upgrade StageRouter (disabled by default).
3) Add gateway service + tests (mocked).
4) Add claim schema + citation validator + entailment gate + tests.
5) Add graph adapters + config gating + tests.
6) Update compose/scripts/docs.

## Open inputs needed
- None.
