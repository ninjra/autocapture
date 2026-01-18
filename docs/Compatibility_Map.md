# Compatibility Map

This map summarizes Phase 0 changes and why they were made, without altering the overall architecture.

## Capture
- Added a shared FrameRecord builder and monotonic timestamps to standardize new writes.
- Ensured hashing happens after privacy masking.

## Extraction
- Persisted canonical OCR spans with offsets and bbox coordinates.
- Added normalized text for indexing while preserving raw text for citations.

## Indexing and Retrieval
- Introduced thresholding and explicit no-evidence handling.
- Added retention-aware index pruning hooks and an integrity scan utility.
- Added a deterministic retrieval eval gate.

## Answer and Infra
- Added provider routing capabilities and safe selection logging.
- Added optional OpenTelemetry instrumentation with a strict attribute allowlist.
- Added secret redaction and an environment-backed secret store.
