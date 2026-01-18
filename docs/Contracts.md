# Phase 0 Contracts

This document defines the Phase 0 contracts used across capture, extraction, retrieval, and answer flows. These contracts are stable, minimal, and designed for safe iteration.

## FrameRecord v1
Required fields for new writes:
- frame_id (UUID/ULID)
- event_id (optional grouping identifier)
- created_at_utc (wall-clock UTC datetime)
- monotonic_ts (monotonic clock float; ordering and latency math only)
- monitor_id
- monitor_bounds (x, y, width, height)
- app_name (nullable)
- window_title (nullable)
- active_window (nullable; keep key present)
- image_path or blob reference (nullable; keep key present)
- privacy_flags: { excluded, masked_regions_applied, cloud_allowed }
- frame_hash (content hash after masking)
- phash (optional)

Clock rule:
- Never compute latency deltas using wall-clock time. Use monotonic_ts only.

## OCRSpan (canonical citation unit)
Each span must include:
- span_id (stable per extraction output)
- start_offset, end_offset (offsets into concatenated raw OCR text)
- bbox_px (x0, y0, x1, y1 in pixels, referencing the stored masked frame)
- conf (0..1)
- text (raw immutable)
- engine (string)
- frame_id and frame_hash provenance

Immutability rule:
- Raw OCR text and spans are immutable once stored. Use normalized text only for indexing.

## RetrievalResult v1
Each retrieved item must include:
- event_id and frame_id
- snippet text and snippet_offset (when applicable)
- bbox pointer (OCR span bbox) or non_citable=true
- score breakdown fields (lexical, dense, sparse, late_interaction, rerank) even if null
- dedupe_group_id (nullable)

Non-citable rule:
- If a result cannot be mapped to a citation unit, it must be flagged non_citable.
