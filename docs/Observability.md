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

## Metrics
When enabled, the following metrics are emitted:
- Stage latency histograms
- Error counters (best-effort)
- Queue depth gauge (where a queue exists)

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
