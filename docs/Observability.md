# Observability

Phase 0 includes OpenTelemetry instrumentation with a strict attribute allowlist. Instrumentation is enabled by default.

## Enable/disable OpenTelemetry
- `features.enable_otel` defaults to `true`. Set it to `false` to disable tracing/metrics.
- If OpenTelemetry libraries are not installed, instrumentation remains a no-op.

## Spans
Emitted spans include:
- capture_frame
- store_media
- extract_ocr
- retrieval.lexical
- retrieval.vector
- retrieval.graph
- vector_upsert
- answergraph.stage
- llm.call
- gateway.upstream
- verification.citations
- verification.entailment

## Metrics
When enabled, the following metrics are emitted:
- Stage latency histograms
- Error counters (best-effort)
- Queue depth gauge (where a queue exists)
- Gateway request counters/latency (`gateway_requests_total`, `gateway_latency_ms`)
- Graph worker request counters/latency (`graph_requests_total`, `graph_latency_ms`)
- Verification failures (`verification_failures_total`)

## Metrics endpoints
- Core app metrics: `observability.prometheus_port`
- Gateway metrics: `GET /metrics` on the gateway port
- Graph worker metrics: `GET /metrics` on the graph worker port

See `config/prometheus.yml` for a WSL2-friendly Prometheus scrape example.

## Performance snapshots
Autocapture writes lightweight performance snapshots to disk for tuning and refactors.

- Runtime + API snapshots are written to `data/perf/runtime.jsonl` and `data/perf/api.jsonl`.
- The local dashboard surfaces runtime perf (CPU, queues, p50/p95 latency) in Maintenance.
- The Maintenance tab also includes a perf log viewer (recent JSONL entries).
- Performance mode can be set to `auto`, `foreground` (max capture), or `idle` (low impact).
  The override file lives at `data/state/profile_override.json` and is picked up automatically.

## Attribute Allowlist
Only these attribute keys are permitted:
- stage_name
- success
- provider_id
- model_id
- error_type
- count
- mode
- retrieval_mode
- adapter
- verdict
- attempt
- citation_count
- claim_count
- queue_depth

## Forbidden Attributes
The following must never appear in telemetry attributes:
- OCR text
- window titles
- URLs
- user queries
- secrets or tokens
- file paths
