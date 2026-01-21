# Config Reference (SPEC-SINGLE-MACHINE Additions)

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

## Memory Service
- `memory_service.enabled`: enable the Memory Service (default: false).
- `memory_service.bind_host` / `memory_service.port`: bind address and port.
- `memory_service.database_url`: optional DB URL override for the Memory Service (defaults to `${data_dir}/memory_service.db`).
- `memory_service.default_namespace`: namespace used when none supplied by clients.
- `memory_service.enable_ingest` / `enable_query` / `enable_feedback`: endpoint toggles.
- `memory_service.embedder.provider`: `stub` or `local` (default: stub).
- `memory_service.reranker.provider`: `disabled` or `stub`.
- `memory_service.retrieval.*`: `topk_*`, budgets, type priority, and per-type caps.
- `memory_service.ranking.*`: stable ranking weights and recency half-life.
- `memory_service.policy.allowed_audiences`: allowlist for query and ingest.
- `memory_service.policy.sensitivity_order`: ordered list for sensitivity checks.

## Retrieval
- `retrieval.v2_enabled`, `retrieval.use_spans_v2`: enable v2 retrieval paths.
- `retrieval.sparse_enabled`, `retrieval.late_enabled`: enable sparse + late-interaction stages.
- `retrieval.fusion_enabled`, `retrieval.rrf_enabled`: enable fusion + RRF.
- `retrieval.traces_enabled`: persist retrieval traces.
- `retrieval.graph_adapters.*`: HTTP graph adapters (defaults to Graph service).

## Routing (Backends)
- `routing.vector_backend`: vector backend plugin id (default: `local` for SQLite).
- `routing.spans_v2_backend`: spans_v2 backend plugin id (default: `local` for SQLite).
- `routing.table_extractor`: table extractor plugin id (default: `disabled`).

## Table Extractor
- `table_extractor.enabled`: enable the table extraction pipeline (default: false).
- `table_extractor.allow_cloud`: allow cloud-backed table extraction when enabled (default: false).

## Citation Validation
- `verification.citation_validator.allow_legacy_evidence_ids`: allow evidence_ids-only claims.
- `verification.citation_validator.max_line_span`: max lines per citation span.

## Features
- `features.enable_otel`: enable OpenTelemetry spans/metrics (default: true).
- `features.enable_memory_service_write_hook`: enable Memory Service ingest hook.
- `features.enable_memory_service_read_hook`: enable Memory Service read hook.

## Paths & Storage
- `paths.base_dir`: optional base directory; derives capture/worker/memory defaults.
- `paths.data_dir` / `paths.staging_dir`: overrides for capture storage paths.
- `paths.worker_dir`: overrides worker data dir.
- `paths.memory_dir`: overrides memory store root.
- `paths.tracking_db_path`: overrides host event SQLite file.
- `paths.database_path`: overrides SQLite metadata DB file path.

## Tracking
- `tracking.raw_event_stream_enabled`: persist raw keyboard/mouse events (default: true).
- `tracking.raw_event_flush_interval_ms`: raw input flush cadence in ms (default: 500).
- `tracking.raw_event_batch_size`: buffer size before flush (default: 500).
- `tracking.raw_event_retention_days`: raw input retention window (default: 60).

## UI (Search Popup)
- `ui.search_popup.focus_steal_on_show`: focus on open (default: true).
- `ui.search_popup.focus_return_on_submit`: return focus after submit (default: true).
- `ui.search_popup.fade_when_inactive`: fade when inactive (default: true).
- `ui.search_popup.active_opacity` / `inactive_opacity`: opacity levels.
- `ui.search_popup.pin_default`: keep popup interactive by default (default: false).

## Model Registry
- `model_registry.enabled`: enable stage-based routing registry.
- `model_registry.providers.*.max_concurrency`: per-provider concurrency cap.
- `model_registry.stages.*.requirements.claims_schema`: set to `claims_json_v1` for claim-level outputs.
- `model_registry.stages.*.requirements.require_json`: enforce JSON-only responses.
- `model_registry.models.*.lmcache_enabled`: enable LMCache hints for that model.

## Policy (PolicyEnvelope)
- `policy.enforce_prompt_injection`: enable prompt-injection enforcement (default: false).
- `policy.prompt_injection_warn_threshold`: risk score threshold for warnings.
- `policy.prompt_injection_block_threshold`: risk score threshold for blocking LLM calls.
- `policy.structured_output_mode`: structured output mode (default: `none`).
- `policy.max_context_chars`: optional hard cap on context pack size.
- `policy.max_evidence_items`: optional hard cap on evidence items.

## Cache
- `cache.enabled`: enable semantic cache (default: false).
- `cache.path`: sqlite cache path override (default: null).
- `cache.max_entries`: max cache entries (default: 10000).
- `cache.ttl_s`: TTL in seconds (default: 86400).
- `cache.prune_interval_s`: prune cadence in seconds (default: 3600).
- `cache.redact_on_cloud`: redact payloads on cloud-enabled calls.

## Observability Telemetry
- `observability.telemetry.capture_payloads`: payload capture mode (none|redacted|full).
- `observability.telemetry.exporter`: exporter type (none|otlp).
- `observability.telemetry.otlp_endpoint`: OTLP endpoint for exporter.
- `observability.telemetry.otlp_protocol`: protocol (http/protobuf).
- `observability.telemetry.allow_cloud_export`: allow telemetry export to non-loopback endpoints.
- `observability.telemetry.max_attr_len`: max attribute length for spans.

## Security
- `security.secure_mode`: fail closed on checksum mismatches or unknown native extensions.
