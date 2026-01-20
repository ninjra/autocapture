# Config Reference (SPEC-1 Additions)

## Gateway
- `gateway.enabled`: enable the LLM Gateway (default: true).
- `gateway.bind_host` / `gateway.port`: bind address and port.
- `gateway.require_api_key`: require API key for public `/v1/*` endpoints.
- `gateway.api_key`: API key for public endpoints.
- `gateway.internal_token`: token for internal stage calls (`X-Internal-Token`).
- `gateway.gpu_max_concurrency`: global GPU concurrency cap for gateway routing.

## Graph Service
- `graph_service.enabled`: enable graph worker service (default: true).
- `graph_service.require_workers`: require external CLI workers (default: true).
- `graph_service.graphrag_cli`, `graph_service.hypergraphrag_cli`, `graph_service.hyperrag_cli`:
  CLI wrappers invoked by the graph service.
- `graph_service.worker_timeout_s`: CLI timeout for index/query jobs.

## Retrieval
- `retrieval.v2_enabled`, `retrieval.use_spans_v2`: enable v2 retrieval paths.
- `retrieval.sparse_enabled`, `retrieval.late_enabled`: enable sparse + late-interaction stages.
- `retrieval.fusion_enabled`, `retrieval.rrf_enabled`: enable fusion + RRF.
- `retrieval.traces_enabled`: persist retrieval traces.
- `retrieval.graph_adapters.*`: HTTP graph adapters (defaults to Graph service).

## Citation Validation
- `verification.citation_validator.allow_legacy_evidence_ids`: allow evidence_ids-only claims.
- `verification.citation_validator.max_line_span`: max lines per citation span.

## Features
- `features.enable_otel`: enable OpenTelemetry spans/metrics (default: true).

## Model Registry
- `model_registry.enabled`: enable stage-based routing registry.
- `model_registry.providers.*.max_concurrency`: per-provider concurrency cap.
- `model_registry.stages.*.requirements.claims_schema`: set to `claims_json_v1` for claim-level outputs.
- `model_registry.stages.*.requirements.require_json`: enforce JSON-only responses.
- `model_registry.models.*.lmcache_enabled`: enable LMCache hints for that model.
