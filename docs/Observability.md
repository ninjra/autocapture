# Observability

Phase 0 adds optional OpenTelemetry instrumentation with a strict attribute allowlist. Instrumentation is disabled by default.

## Enable OpenTelemetry
- Set `features.enable_otel` to `true` in your configuration.
- If OpenTelemetry libraries are not installed, instrumentation remains a no-op.

## Spans
Emitted spans include:
- capture_frame
- store_media
- extract_ocr
- index_lexical
- vector_search
- vector_upsert
- answer_generate
- gateway.upstream
- verify_citations
- verify_entailment

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

## Attribute Allowlist
Only these attribute keys are permitted:
- stage_name
- success
- provider_id
- error_type
- count
- mode
- queue_depth

## Forbidden Attributes
The following must never appear in telemetry attributes:
- OCR text
- window titles
- URLs
- user queries
- secrets or tokens
- file paths
