# Priority >= 2 Blueprint Status Matrix (2026-01-20)

Scope: repo inspection only. No runtime changes applied.
Branch: feat/priority2-blueprint-pluginized
Head: 9c64ae9

## Doctrine sources (authoritative)
- `docs/ARCHITECTURE.md`
- `docs/architecture/plugin-system.md`
- `docs/pillars/SPEC-1.md`
- `docs/pillars/SPEC-260117.md`
- `docs/plugins.md`
- `docs/spec-1/as-is.md`
- `docs/spec-1/to-be.md`

## Plugin system contract (summary)
- Core plugin runtime and policy gates live in `autocapture/plugins/`.
- Extension kinds are enumerated in `autocapture/plugins/constants.py`.
- Built-in manifests live under `autocapture/plugins/builtin/**/plugin.yaml`.
- External plugins are enabled/locked via `settings.json` in the data dir, enforced by
  `autocapture/plugins/manager.py` + `autocapture/plugins/settings.py`.

## Priority >= 2 Status Matrix
Status legend: exists | partial | missing | incorrect

| ID | Capability | Status | Owner module / plugin | Evidence | Notes / gaps |
| --- | --- | --- | --- | --- | --- |
| SRV-01 | OpenAI-compatible gateway + internal stage routing + auth guardrails | exists | `autocapture/gateway/*` | `autocapture/gateway/app.py`<br>`autocapture/gateway/service.py` | Internal endpoints protected by loopback/token checks. |
| SRV-02 | Model registry (providers/models/stages) with fallback chain + circuit breaker | exists | `autocapture/model_ops/*` | `autocapture/model_ops/registry.py`<br>`autocapture/model_ops/router.py`<br>`autocapture/config.py` | Stage fallback uses `fallback_model_ids`; breaker enforced in router. |
| SRV-03 | Provider plugin system + policy gate for cloud/offline | exists | `autocapture/plugins/*` | `autocapture/plugins/manager.py`<br>`autocapture/plugins/policy.py`<br>`autocapture/plugins/builtin/autocapture.builtin.llm/plugin.yaml` | Deterministic resolution + hash locks + policy enforcement. |
| SRV-04 | Gateway claim-level JSON enforcement + repair loop | exists | `autocapture/gateway/service.py` | `autocapture/gateway/service.py`<br>`autocapture/answer/claim_validation.py` | Uses `ClaimValidator` with retries when enabled. |
| SRV-05 | Stage-level decode strategy routing + concurrency limits | partial | `autocapture/gateway/service.py`, plugins | `autocapture/gateway/service.py`<br>`autocapture/plugins/builtin/autocapture.builtin.decode/plugin.yaml`<br>`autocapture/config.py` | Strategy routing + plugin-resolved decode backends exist; still requires external OpenAI-compatible servers for actual decoding. |
| RAG-01 | Retrieval stack: lexical + vector + sparse + late + RRF fusion | exists | `autocapture/memory/*` | `autocapture/memory/retrieval.py`<br>`autocapture/indexing/lexical_index.py`<br>`autocapture/indexing/vector_index.py`<br>`autocapture/indexing/spans_v2.py` | Deterministic fusion and tiered retrieval are implemented. |
| RAG-02 | Qdrant backend + vector plugin | exists | `autocapture/indexing/*` + plugin | `autocapture/indexing/vector_index.py`<br>`autocapture/plugins/builtin/autocapture.builtin.vector/plugin.yaml`<br>`autocapture/plugins/builtin/factories.py` | Qdrant can be disabled via config; plugin wraps backend. |
| RAG-03 | Graph adapters (GraphRAG / HyperGraphRAG / Hyper-RAG) + service endpoints | exists | `autocapture/graph/*`, `autocapture/memory/graph_adapters.py`, plugins | `autocapture/graph/app.py`<br>`autocapture/graph/service.py`<br>`autocapture/memory/graph_adapters.py`<br>`autocapture/plugins/builtin/autocapture.builtin.graph/plugin.yaml` | Graph adapters are plugin-resolved HTTP clients; external worker CLIs still required when `graph_service.require_workers=true`. |
| RAG-04 | Prompt-injection scanning + redaction in context packs | exists | `autocapture/memory/*` | `autocapture/memory/prompt_injection.py`<br>`autocapture/memory/context_pack.py` | Redaction is applied before evidence is serialized. |
| CTX-01 | Context pack evidence schema + spans + line maps | exists | `autocapture/memory/*`, `autocapture/agents/answer_graph.py` | `autocapture/memory/context_pack.py`<br>`autocapture/agents/answer_graph.py` | Line maps persisted for citations; evidence spans include bbox metadata. |
| CTX-02 | Claim-level citations schema + parser/renderer | exists | `autocapture/answer/*` | `autocapture/answer/claims.py`<br>`autocapture/answer/claim_validation.py` | Claims JSON v2 is enforced via validators. |
| CTX-03 | Citation integrity checks (ledger + media presence) | exists | `autocapture/answer/integrity.py` | `autocapture/answer/integrity.py`<br>`autocapture/storage/ledger.py` | Integrity check verifies spans and provenance ledger entries. |
| CTX-04 | Entailment gate (heuristic + LLM judge) with policy actions | exists | `autocapture/answer/*`, `autocapture/agents/answer_graph.py` | `autocapture/answer/entailment.py`<br>`autocapture/agents/answer_graph.py` | Supports block/regenerate/abstain/expand policies. |
| OPS-01 | OpenTelemetry spans + safe attribute handling | exists | `autocapture/observability/*` | `autocapture/observability/otel.py`<br>`autocapture/agents/answer_graph.py`<br>`autocapture/gateway/service.py` | OTel initialized in API; spans used across major stages. |
| OPS-02 | Prometheus metrics + /metrics endpoints | exists | `autocapture/observability/*`, services | `autocapture/observability/metrics.py`<br>`autocapture/api/server.py`<br>`autocapture/gateway/app.py`<br>`autocapture/graph/app.py` | Metrics endpoints are exposed per service. |
| OPS-03 | Observability config + Prometheus scrape config | exists | `autocapture/config.py`, `infra/prometheus.yml` | `autocapture/config.py`<br>`infra/prometheus.yml` | Ports and scrape targets are configurable. |
| TRN-01 | Decode backends (swift/lookahead/medusa) via external servers | partial | `autocapture/gateway/service.py`, plugins + scripts | `autocapture/gateway/service.py`<br>`autocapture/plugins/builtin/autocapture.builtin.decode/plugin.yaml`<br>`scripts/run_vllm.sh`<br>`scripts/run_vllm_gpu_a.sh` | Proxy plugins are available; runtime still requires external servers (vLLM) for actual decoding. |
| TRN-02 | LoRA adapter allowlist + gateway enforcement | partial | `autocapture/config.py`, `autocapture/gateway/service.py` | `autocapture/config.py`<br>`autocapture/gateway/service.py` | Runtime gating exists; no training pipeline or adapter registry beyond config. |
| TRN-03 | Training pipelines (LoRA/QLoRA/DPO) | partial | `autocapture/training/*`, plugins | `autocapture/training/pipelines.py`<br>`autocapture/plugins/builtin/autocapture.builtin.training/plugin.yaml`<br>`autocapture/main.py` | CLI + plugin scaffolds exist; command-based execution supported via settings.json; no bundled trainer. |
