# SPEC-260117 repo reality (snapshot)

- Date (UTC): 2026-01-20
- Branch: spec-260117-ship-architecture
- Commit: 1932b2f

## Frozen surfaces and invariants
- Frozen manifest: `autocapture/stability/frozen_manifest.json`
  - Frozen file: `autocapture/memory/context_pack.py` (context pack schema/serialization)
  - Frozen file: `docs/CONTEXT_PACK.md` (context pack documentation)
- Freeze tooling + gate: `tools/freeze_surfaces.py`, `tests/test_frozen_surfaces.py`

## Core pipeline components (paths)
- Stage routing: `autocapture/model_ops/router.py` (`StageRouter`, `StageDecision`)
- PolicyEnvelope boundary: `autocapture/policy/envelope.py` (LLM call enforcement wrapper)
- Cloud/offline policy gates:
  - Plugin policy: `autocapture/plugins/policy.py` (`guard_cloud_text`, `guard_cloud_images`)
  - Gateway gating: `autocapture/gateway/service.py` (`_cloud_allowed`)
  - Provider routing: `autocapture/memory/router.py` (openai/openai_compatible gating)
  - Offline egress guard: `autocapture/security/offline_guard.py` (applied in `autocapture/main.py`)
- Retrieval: `autocapture/memory/retrieval.py`
- Context pack builder: `autocapture/memory/context_pack.py` (`build_context_pack`)
- Answer orchestrator: `autocapture/agents/answer_graph.py` (`AnswerGraph`)
- Answer API surface: `autocapture/api/server.py` (`/api/answer` endpoint)
- Extractive fallback: `autocapture/memory/compression.py` (`extractive_answer`) and fallback branch in `autocapture/agents/answer_graph.py`
- Rules verifier: `autocapture/memory/verification.py` (`RulesVerifier`)

## Context pack schema / constructor
- Schema + serialization: `autocapture/memory/context_pack.py`
  - `EvidenceSpan`, `EvidenceItem`, `ContextPack`, `ContextPack.to_json()` (version=1)
  - `build_context_pack()` generates packs with privacy/routing/filters/aggregates
- TRON formatting: `autocapture/format/tron.py`
- API models: `autocapture/api/server.py` (`ContextPackRequest`, `ContextPackResponse`, `/api/context_pack`)
- Spec doc (frozen): `docs/CONTEXT_PACK.md`

## Answer JSON schema / emitter
- Response model: `autocapture/api/server.py` (`AnswerResponse`)
- Payload builder: `autocapture/api/server.py` (`_build_answer_payload`)
  - Fields: `answer`, `citations`, `warnings`, `used_llm`, `context_pack`, `evidence`
  - Optional TRON encoding via `autocapture/format/tron.py`
- Answer flow + citation enforcement: `autocapture/agents/answer_graph.py`

## Defaults and gating (current)
- Offline default: `AppConfig.offline` defaults to `True` and `load_config()` forces offline true if unset.
- Cloud defaults: `PrivacyConfig.cloud_enabled = False`, `PrivacyConfig.allow_cloud_images = False`.
- Stage-level cloud opt-in: `ModelStageConfig.allow_cloud = False` (default).
- Bind defaults (loopback):
  - API: `APIConfig.bind_host = 127.0.0.1`
  - Gateway: `GatewayConfig.bind_host = 127.0.0.1`
  - Graph Service: `GraphServiceConfig.bind_host = 127.0.0.1`
  - Memory Service: `MemoryServiceConfig.bind_host = 127.0.0.1`
  - Metrics: `ObservabilityConfig.prometheus_bind_host = 127.0.0.1`
- Non-loopback binding safeguards: `AppConfig.validate_api_security()` enforces API key + HTTPS when binding non-loopback in local mode.

## Observability / telemetry
- OTel initialization: `autocapture/observability/otel.py` (`init_otel`, `otel_span`)
  - Attribute allowlist only; no raw prompt/evidence payload capture.
- Prometheus metrics: `autocapture/observability/metrics.py` (`MetricsServer`, counters/gauges/histograms)
- OTel init is called by `autocapture/runtime.py` and `autocapture/api/server.py`.

## Tests and CI entrypoints
- Test runner: `pytest` (see `tests/` and `pytest.ini`).
- CI workflows: `.github/workflows/ci.yml`, `.github/workflows/dev-check.yml`, `.github/workflows/research-scout.yml`.
- Release/frozen surface gates: `tools/release_gate.py`, `tests/test_frozen_surfaces.py`.
- Existing regression tests relevant to SPEC-260117:
  - Offline/cloud gating: `tests/test_router_cloud_guard.py`, `tests/test_provider_router_policy.py`
  - Citation validation: `tests/test_answer_citations.py`, `tests/test_claim_validation.py`
  - Answer flow: `tests/test_answer_graph.py`, `tests/test_answer_speculative.py`

## Baseline regression fixtures (current)
- Retrieval eval datasets: `evals/phase0_retrieval.jsonl` and `evals/phase0_retrieval_baseline.json`.
- Bench baseline: `bench/baseline.json` (used by `bench/ci_gate.py`).
- No new fixtures added in this pass; existing tests above provide baseline gates for citations and offline/cloud policy.
