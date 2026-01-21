# SPEC-SINGLE-MACHINE To-Be Plan (Single-Machine Production Topology)

This plan documents the SPEC-SINGLE-MACHINE P0 implementation and its file-level anchors.

## Baseline Anchors (Existing)
- Stage routing and registry live in `StageRouter` and `ModelRegistry`.
  (`autocapture/model_ops/router.py:L38-L210`, `autocapture/model_ops/registry.py:L22-L83`)
- Gateway exposes OpenAI-compatible endpoints and internal stage routing with policy enforcement.
  (`autocapture/gateway/service.py:L38-L361`, `autocapture/gateway/app.py:L25-L149`)
- Claim-level JSON parsing/validation, entailment gates, and deterministic line maps are implemented.
  (`autocapture/answer/claims.py:L15-L101`, `autocapture/answer/claim_validation.py:L1-L120`,
   `autocapture/answer/entailment.py:L1-L156`, `autocapture/agents/answer_graph.py:L393-L1900`)
- Retrieval stack includes lexical (SQLite FTS), vector (Qdrant), fusion, and graph adapters.
  (`autocapture/indexing/lexical_index.py:L31-L214`, `autocapture/indexing/vector_index.py:L253-L520`,
   `autocapture/indexing/spans_v2.py:L32-L217`, `autocapture/memory/retrieval.py:L91-L1040`)
- Prompt-injection scanning is deterministic and persisted in evidence metadata.
  (`autocapture/memory/prompt_injection.py:L1-L61`, `autocapture/memory/context_pack.py:L49-L138`)
- Observability primitives (OTel + Prometheus) are wired into API/gateway/retrieval.
  (`autocapture/observability/otel.py:L1-L170`, `autocapture/observability/metrics.py:L1-L214`,
   `autocapture/api/server.py:L395-L410`)
- DB migrations include SPEC-SINGLE-MACHINE run logging + citations + FTS tables.
  (`alembic/versions/0014_spec_single_machine_runs_and_citations.py:L1-L150`)

## Port Map (Defaults)
- API: `api.bind_host`/`api.port` default 127.0.0.1:8008.
  (`autocapture/config.py:L647-L661`, `autocapture.yml:L12-L24`)
- Gateway: 127.0.0.1:8010.
  (`autocapture/config.py:L1329-L1338`, `autocapture.yml:L186-L195`)
- Graph workers: 127.0.0.1:8020.
  (`autocapture/config.py:L1341-L1347`, `autocapture.yml:L196-L200`)
- Prometheus metrics: 127.0.0.1:9005 (scraped in `infra/prometheus.yml`).
  (`autocapture/config.py:L598-L604`, `infra/prometheus.yml:L1-L19`)

## Security Posture (Target)
- Preserve loopback-first defaults and offline guard behavior from AppConfig validation.
  (`autocapture/config.py:L1507-L1537`)
- Harden internal stage endpoints (`/internal/stage/...`) with local-only or API-key enforcement
  (additive to existing gateway). (`autocapture/gateway/app.py:L112-L131`)

## Delta Plan (SPEC-SINGLE-MACHINE P0 Mapping)

### 1) Gateway + Registry Hardening
- Extend `GatewayRouter` to enforce internal auth and stage allowlists.
  (`autocapture/gateway/service.py:L83-L247`, `autocapture/gateway/app.py:L112-L131`)
- Add explicit provider/model capability metadata to registry config (if needed) to support
  vLLM multi-instance routing and decode strategies. (`autocapture/config.py:L1230-L1326`)

### 2) EvidencePack + Citation Validator Hard Gate
- Expand EvidencePack to include deterministic line ranges (persisted via line maps) and
  explicit claim-level citations.
  (`autocapture/memory/context_pack.py:L49-L138`,
   `autocapture/agents/answer_graph.py:L1525-L1855`,
   `alembic/versions/0014_spec_single_machine_runs_and_citations.py:L1-L150`)
- Enforce claim-level citations in final answers (no partial compliance), using
  `ClaimValidator` and `check_citations`. (`autocapture/answer/claim_validation.py:L1-L55`,
  `autocapture/answer/integrity.py:L26-L74`)

### 3) Retrieval Fusion + Worker Wrappers
- Keep lexical + vector + spans_v2 retrieval and extend fusion with explicit RRF parameters for
  SPEC-SINGLE-MACHINE defaults. (`autocapture/memory/retrieval.py:L160-L395`)
- Wire graph adapters to GraphRAG/HyperGraphRAG/Hyper-RAG workers via graph service API.
  (`autocapture/graph/app.py:L27-L146`, `autocapture/memory/retrieval.py:L984-L1038`)

### 4) Entailment Gate (Heuristic + Verifier)
- Use existing heuristic/judge pipeline and tighten policy decisions on `contradicted` / `nei`.
  (`autocapture/answer/entailment.py:L1-L156`, `autocapture/agents/answer_graph.py:L1562-L1759`)
- Ensure stage-level config uses `entailment_judge` with deterministic sampling.
  (`autocapture/config.py:L1356-L1368`, `autocapture/model_ops/router.py:L182-L193`)

### 5) Observability + Metrics
- Extend existing OTel spans and Prometheus counters for SPEC-SINGLE-MACHINE required names, reusing
  `otel_span` and metrics registry.
  (`autocapture/observability/otel.py:L110-L126`, `autocapture/observability/metrics.py:L88-L121`)

### 6) Infra + Runbook (Single-Machine)
- Add compose + scripts for local Qdrant + Prometheus + Loki + Grafana.
  (`infra/compose.yaml:L1-L44`, `infra/prometheus.yml:L1-L17`)
- Document gateway/API/graph service startup using existing CLI subcommands.
  (`autocapture/main.py:L548-L602`)

## File-Level Work Plan (Proposed)
- **Modify** `autocapture/gateway/service.py` and `autocapture/gateway/app.py`: internal auth,
  stricter policy enforcement, bounded retries.
- **Modify** `autocapture/agents/answer_graph.py`: tighten citation/entailment policy flow,
  record provider call metadata to match `provider_calls` schema.
- **Modify** `autocapture/memory/context_pack.py`: include deterministic line-span references
  required by claim citations (non-breaking additions).
- **Modify** `autocapture/memory/retrieval.py`: finalize RRF fusion + graph adapter integration
  toggles for SPEC-SINGLE-MACHINE.
- **Add** `infra/compose.yaml` + `infra/prometheus.yml` + `scripts/` run templates (WSL2).
- **Add** `docs/runbook-single-machine.md` + `docs/config-reference.md` +
  `docs/Observability.md` (grounded in existing config/metrics).

## Test Strategy (P0)
- Extend existing unit tests for claim validation, entailment, and gateway validation.
  (`tests/test_claim_validation.py:L1-L35`, `tests/test_entailment.py:L1-L60`,
   `tests/test_gateway_service.py:L1-L102`)
- Reuse `make check` / `pytest -q` as the default validation loop.
  (`Makefile:L1-L30`)

## Rollout & Safety
- Defaults remain offline + loopback-only unless explicitly configured.
  (`autocapture/config.py:L1438-L1537`, `config/example.yml:L1-L120`)
- Qdrant, graph workers, and vLLM instances are mandatory in SPEC-SINGLE-MACHINE configs and runbooks.
  (`autocapture.yml:L48-L210`, `config/example.yml:L114-L330`)
