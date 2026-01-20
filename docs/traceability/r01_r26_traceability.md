# SPEC-260117 R01-R26 traceability (plan-to-repo map)

This is a concrete mapping from R01-R26 to repo components, enforcement points, tests, and config. It is additive-only and respects frozen surfaces in `autocapture/stability/frozen_manifest.json`.

## Frozen surfaces (must not edit without manifest update)
- `autocapture/memory/context_pack.py`
- `docs/CONTEXT_PACK.md`

## Traceability matrix

| Rxx | Requirement focus | Enforcement points (paths) | Tests/gates (paths) | Config knobs |
| --- | --- | --- | --- | --- |
| R01 | Request trace root + IDs | `autocapture/api/server.py` (middleware + response), `autocapture/observability/otel.py` | New tests under `tests/test_otel_tracing.py` (planned) | `features.enable_otel` |
| R02 | OpenInference-style span attrs | `autocapture/observability/otel.py`, `autocapture/agents/answer_graph.py`, `autocapture/memory/retrieval.py` | New unit tests for span attributes (planned) | `features.enable_otel` |
| R03 | Local Phoenix setup | Add compose/runbook under `infra/` + `docs/Observability.md` | Doc-only validation | New config doc entries |
| R04 | Optional OTLP exporter / collector fan-out | Extend `autocapture/observability/otel.py` | New tests for exporter config (planned) | New observability config fields |
| R05 | Trace propagation + request_id | `autocapture/api/server.py` (request handling) | New tests for response envelope (planned) | None (additive) |
| R06 | Telemetry sanitizer / payload capture modes | New module under `autocapture/observability/telemetry.py` and integration into `otel_span` usage | New tests for redaction (planned) | New telemetry config section |
| R07 | Provider registry abstraction | New module `autocapture/llm/registry.py` (planned) | New provider contract tests in `tests/providers/` | New provider config section |
| R08 | Structured outputs policy | `autocapture/agents/answer_graph.py` (stage calls), provider adapters in `autocapture/llm/providers.py` | New tests for schema handling | New per-stage structured output config |
| R09 | vLLM guided decoding probe | Extend `autocapture/llm/providers.py` (openai_compatible) | New stubbed contract tests | Provider-specific config |
| R10 | TGI provider | New provider in `autocapture/llm/providers.py` | New stubbed provider tests | Provider registry config |
| R11 | TEI embedding provider | Extend `autocapture/embeddings/service.py` or new provider module | New embedding contract tests | Embedder config |
| R12 | Provider health endpoint | `autocapture/api/server.py` or `autocapture/gateway/app.py` | New tests for `/health/providers` | New health config (optional) |
| R13 | Promptfoo eval harness | Add under `evals/promptfoo/` (planned) | CI job in `.github/workflows/ci.yml` | Promptfoo config files |
| R14 | Redteam suite (nightly) | `.github/workflows/ci.yml` scheduled job | Nightly CI artifact | Promptfoo config |
| R15 | Injection + PII + schema eval gates | `tools/*_gate.py` extensions or new tools | Unit tests + CI checks | New gate config |
| R16 | OWASP mapping file | `config/security_controls.yml` (planned) | New gate script to validate mapping | New config file |
| R17 | Security gate coverage check | New tool under `tools/security_controls_gate.py` | CI step in `.github/workflows/ci.yml` | None |
| R18 | Prompt injection heuristics | New module `autocapture/security/injection.py` and integration into answer flow | New unit tests | New policy config |
| R19 | Tool allowlist policy | `autocapture/plugins/policy.py` and new policy enforcement in LLM calls | New unit tests | New policy config |
| R20 | Prompt injection enforcement | `autocapture/policy/envelope.py`, `autocapture/agents/answer_graph.py` | `tests/test_policy_envelope.py` | `policy.*` |
| R21 | PII text redaction for cloud | `autocapture/security/redaction.py` (planned) + provider request builders | New unit tests | New privacy config |
| R22 | Image redaction modes | `autocapture/vision/` modules, storage pipeline | New tests | New privacy config |
| R23 | Semantic caching (policy-aware) | New cache module under `autocapture/cache/` + integration at LLM call boundary | New unit tests | New cache config |
| R24 | Cache privacy + pruning | `autocapture/cache/` + maintenance hook | New unit tests | Cache TTL / max entries config |
| R25 | Budgets + DoS controls | `autocapture/runtime_budgets.py` and call sites in `autocapture/agents/answer_graph.py` | New unit tests | `config/defaults/budgets.json` + new knobs |
| R26 | Trace-to-evidence linkage | Add hash computation in `autocapture/answer/provenance.py` + response fields in `autocapture/api/server.py` | New tests for response envelope + verifier | Additive response fields |

## Existing components to preserve (no behavior regression)
- Stage routing and cloud/offline gating: `autocapture/model_ops/router.py`, `autocapture/plugins/policy.py`, `autocapture/gateway/service.py`, `autocapture/memory/router.py`.
- RulesVerifier behavior: `autocapture/memory/verification.py` and `autocapture/answer/claim_validation.py`.
- Extractive fallback: `autocapture/agents/answer_graph.py` + `autocapture/memory/compression.py`.
- Context pack format: `autocapture/memory/context_pack.py` (frozen) + `docs/CONTEXT_PACK.md` (frozen).
- Prometheus metrics: `autocapture/observability/metrics.py` and `/metrics` endpoints.

## Notes on additive surfaces
- Response envelope additions (e.g., `trace_id`, `context_pack_hash`) should be added in `autocapture/api/server.py` and kept optional to preserve backward compatibility.
- Any schema changes touching `autocapture/memory/context_pack.py` require a frozen manifest update and test gate updates.
