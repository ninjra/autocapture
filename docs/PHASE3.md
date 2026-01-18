# Phase 3 Implementation (SPEC-4)

This document maps the Phase 3 requirements from the provided SPEC-4 prompt to concrete modules,
flags, data paths, and tests in this repo. Optional/recommended items are treated as required but
remain behind feature flags for safe rollout.

## Pillars
- **Privacy-first defaults:** local-only by default; cloud requires explicit per-stage opt-in.
- **Determinism:** stable IDs, sorted-key JSON serialization, deterministic parse failures.
- **Performance:** responsive in ACTIVE_INTERACTIVE, aggressive IDLE_DRAIN, hard pause on fullscreen.
- **Citations:** evidence payloads include span-level region data for overlays.

## Requirement Mapping (M1–M13, S1–S3, C1–C2)

| Req | Description | Implementation | Flags | Tests |
| --- | --- | --- | --- | --- |
| M1 | FULLSCREEN_HARD_PAUSE (pause capture + workers + release GPU; auto-resume) | `autocapture/runtime_governor.py`, `autocapture/runtime.py`, `autocapture/worker/supervisor.py`, `autocapture/gpu_lease.py` | `runtime.auto_pause.*` | `tests/test_runtime_governor.py`, `tests/test_worker_supervisor_qos.py` |
| M2 | ACTIVE_INTERACTIVE QoS (bounded concurrency, low CPU priority, lightweight capture) | `autocapture/runtime_governor.py`, `autocapture/worker/supervisor.py`, `autocapture/runtime_qos.py` | `runtime.qos.*` | `tests/test_worker_supervisor_qos.py` |
| M3 | IDLE_DRAIN (aggressive backlog drain) | `autocapture/runtime_governor.py`, `autocapture/worker/supervisor.py` | `runtime.qos.profile_idle` | `tests/test_worker_supervisor_qos.py` |
| M4 | UI grounding extraction | `autocapture/vision/ui_grounding.py`, `autocapture/worker/event_worker.py` | `vision_extract.ui_grounding.*` | `tests/test_ui_grounding_schema.py` |
| M5 | Strict JSON grounding + parse-failure capture | `autocapture/vision/extractors.py`, `autocapture/vision/ui_grounding.py` | `vision_extract.*` | `tests/test_ui_grounding_schema.py` |
| M6 | Late-interaction (ColBERT-style), stage-2 + optional stage-1 | `autocapture/embeddings/late.py`, `autocapture/memory/retrieval.py`, `autocapture/indexing/spans_v2.py` | `retrieval.late_*` | `tests/test_retrieval_late_stage1.py` |
| M7 | Learned sparse retrieval (SPLADE-like) | `autocapture/embeddings/sparse.py`, `autocapture/indexing/spans_v2.py`, `autocapture/memory/retrieval.py` | `retrieval.sparse_*` | `tests/test_retrieval_hybrid.py` |
| M8 | Multi-query retrieval + fusion (RRF) | `autocapture/memory/retrieval.py` | `retrieval.fusion_enabled`, `retrieval.multi_query_enabled`, `retrieval.rrf_enabled` | `tests/test_retrieval_hybrid.py` |
| M9 | Retrieval-aware generation (R²AG signals) | `autocapture/agents/answer_graph.py`, `autocapture/memory/context_pack.py` | `retrieval.*` | `tests/test_answer_citations.py` |
| M10 | Speculative RAG (draft/verify + deep path) | `autocapture/agents/answer_graph.py` | `retrieval.speculative_*`, `model_stages.draft_generate.enabled` | `tests/test_answer_speculative.py` |
| M11 | OCR → layout → markdown + PP-Structure (flagged) | `autocapture/vision/layout.py`, `autocapture/vision/paddle_layout.py`, `autocapture/worker/event_worker.py` | `ocr.layout_enabled`, `ocr.paddle_ppstructure_*` | `tests/test_ocr_layout.py`, `tests/test_paddle_layout.py` |
| M12 | Modern reranker (mode-aware batching/device) | `autocapture/memory/reranker.py`, `autocapture/memory/retrieval.py` | `reranker.*`, `runtime.qos.*` | `tests/test_retrieval_hybrid.py` |
| M13 | Template hardening + provenance | `autocapture/security/template_lint.py`, `autocapture/memory/prompts.py` | `templates.*` | `tests/test_template_lint.py`, `tests/test_promptops_validation.py` |
| S1–S3 | Metrics + traces + rollout flags | `autocapture/observability/metrics.py`, `autocapture/memory/retrieval.py`, `autocapture/config.py` | `retrieval.traces_enabled`, `features.*` | `tests/test_runtime_governor.py` |
| C1 | UI overlay citations (region highlights) | `autocapture/vision/citation_overlay.py`, `autocapture/api/server.py`, `autocapture/ui/web/*` | `ui.overlay_citations_enabled` | `tests/test_citation_overlay.py` |
| C2 | Multi-monitor + HDR metadata (flagged) | `autocapture/capture/orchestrator.py`, `autocapture/worker/event_worker.py`, `autocapture/vision/hdr.py` | `capture.multi_monitor_enabled`, `capture.hdr_enabled` | (unit coverage via metadata in tags) |

## Runtime Governor & Modes
- **Modes:** `FULLSCREEN_HARD_PAUSE`, `ACTIVE_INTERACTIVE`, `IDLE_DRAIN`.
- **Fullscreen pause:** `RuntimeGovernor` detects fullscreen, pauses capture and worker loops, and
  triggers GPU lease release via registered hooks.
- **QoS:** `RuntimeQosProfile` controls worker counts, batch sizes, and CPU priority per mode.

## GPU Lifecycle
- Global GPU lease hooks in `autocapture/gpu_lease.py`.
- OCR, embeddings, reranker, and Ollama providers register release hooks and unload on fullscreen.
- PP-Structure extractor drops its engine on GPU release events.

## Extraction & Grounding
- Deterministic OCR layout is produced via `vision/layout.py`, persisted to `events.tags`.
- Optional PP-Structure backend produces layout blocks/markdown behind `ocr.paddle_ppstructure_*`.
  Requires local model directory to avoid downloads.
- UI grounding is available via VLM or heuristic backends and stored in `events.tags.ui_elements`.

## Retrieval & RAG
- **Spans v2** collection supports dense + sparse + late vectors.
- **Sparse encoder** uses local hash fallback if no model is available.
- **Late stage-1** retrieval runs only on narrow windows (configurable).
- **Multi-query + RRF** fusion and deterministic rewrite generation are available behind flags.
- **Retrieval traces** are persisted with deterministic JSON serialization.

## AnswerGraph & Citations
- Context packs include retrieval signals and span-level metadata.
- Speculative draft/verify path is enabled behind `retrieval.speculative_*`.
- Region-level citations are exposed in `used_context_pack.evidence[*].meta.spans`.

## Template Hardening
- Jinja constructs are parsed and disallowed if unsafe.
- Templates are linted on load; provenance hashes are logged when enabled.

## UI Overlay Citations
- `/api/citations/overlay` renders a deterministic overlay for span bboxes.
- Web UI exposes per-citation overlay buttons when enabled.

## Multi-monitor + HDR Metadata
- Event tags now include `capture_meta` with frame size, monitor bounds, DPI scale, and HDR tags.
- HDR hook is deterministic; tone mapping is a no-op unless HDR is detected.

## Config Flags (Summary)
- `runtime.auto_pause.*`, `runtime.qos.*`
- `ocr.layout_enabled`, `ocr.paddle_ppstructure_*`
- `retrieval.v2_enabled`, `retrieval.fusion_enabled`, `retrieval.multi_query_enabled`,
  `retrieval.rrf_enabled`, `retrieval.sparse_enabled`, `retrieval.late_enabled`,
  `retrieval.late_stage1_*`, `retrieval.traces_enabled`, `retrieval.speculative_*`
- `templates.*`
- `ui.overlay_citations_enabled`
- `capture.multi_monitor_enabled`, `capture.hdr_enabled`

## Tests & CI Notes
- CPU-only CI is supported via deterministic fallbacks.
- Optional GPU/Windows tests should be marked and enabled via env vars if added later.

## Troubleshooting
- PP-Structure requires a local model dir; set `ocr.paddle_ppstructure_model_dir`.
- Overlay endpoint returns 403 unless `ui.overlay_citations_enabled` is true.
