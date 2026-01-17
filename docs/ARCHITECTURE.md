# Autocapture Architecture

## Overview
Autocapture is a local-first Personal Activity Memory Engine. It captures activity, stores evidence locally, and answers questions using context packs with verifiable citations.

## Pipelines

### Capture
- `autocapture/capture/` handles screenshot capture, metadata, and OCR staging.
- Events are persisted in `events` with screen-text transcripts and spans.
 - `autocapture/runtime_governor.py` provides runtime modes and policies that
   coordinate capture, workers, and GPU lifecycle:
   - `FULLSCREEN_HARD_PAUSE`: pause capture and background workers; release GPU.
   - `ACTIVE_INTERACTIVE`: capture continues with throttled background work.
   - `IDLE_DRAIN`: ramp concurrency to drain OCR/embedding/vision backlogs.

### Vision Extraction
- `autocapture/vision/` runs VLM-first extraction with full-screen tiling.
- Structured layout output is stored under `EventRecord.tags["vision_extract"]` and the
  consolidated transcript is stored in `EventRecord.ocr_text` for retrieval compatibility.
- OCR layout heuristics store `layout_blocks` and `layout_md` in `EventRecord.tags`.
- UI grounding (optional) stores structured `ui_elements` in `EventRecord.tags` and
  is enabled only by config/runtime QoS.

### Agents + Enrichment
- `autocapture/agents/` orchestrates enrichment, vision captioning, and nightly highlights.
- Agent jobs are stored in `agent_jobs`/`agent_results` with leasing + retries.
- Enrichment and vision outputs are augmentation-only, stored in tags + synthetic indexes.
- `autocapture/enrichment/scheduler.py` scans recent events to enqueue missing
  vision_extract/enrichment/thread_summary jobs and tracks backlog + at-risk metrics.
- SQL/code artifacts are extracted deterministically and indexed for lexical + vector search.

### Retrieval + Context Pack
- `autocapture/memory/retrieval.py` fetches event evidence from SQLite.
- Qdrant retrieval supports dense, sparse, and late-interaction vectors via `spans_v2`
  when enabled in config.
- Retrieval fusion (RRF) and optional late reranking provide higher recall under
  guarded, deterministic rewrite policies.
- `autocapture/memory/context_pack.py` formats JSON + canonical text context packs.
  - Daily highlight snippets are attached under `aggregates`.
  - Evidence payloads include event_id, timestamps, screenshot metadata, and retrieval
    signals for citations and verifier enforcement.
- `autocapture/format/tron.py` adds TRON encode/decode for compact structured payloads.
- `autocapture/memory/threads.py` stores deterministic activity threads and thread summaries
  for broad/time-window retrieval.

### Providers + Routing
- `autocapture/memory/router.py` selects providers per layer.
- Layers: OCR, embeddings, retrieval, reranking, compression, verification, LLM.
- `autocapture/model_ops/router.py` selects LLMs per stage (query_refine, draft_generate,
  final_answer, tool_transform) with per-stage cloud opt-ins.
- Vision extractors use `vision_extract` backend configuration with explicit cloud-image
  gating.
- `autocapture/llm/governor.py` enforces adaptive concurrency and foreground priority across
  interactive queries + background agents.

### Time Intent
- `autocapture/memory/time_intent.py` deterministically parses time expressions for
  time-range retrieval and timeline answers (time-only queries return short timelines).

### Research Scout
- `autocapture/research/scout.py` discovers recent models/papers, caches results, and
  writes ranked proposal reports.

### Privacy + Entity Resolution
- `autocapture/memory/entities.py` manages stable pseudonyms and alias mapping.
- Tokens are derived from HMAC with a local secret key.
- `token_vault` stores reversible mappings for emails/domains/paths (encrypted at rest).

## Data Model
The local SQLite DB contains:
- `events` (activity + OCR evidence)
- `runtime_state` (singleton runtime mode + pause metadata)
- `retrieval_traces` (optional retrieval fusion traces)
- `threads`, `thread_events`, `thread_summaries` (activity threads + summaries)
- `entities`, `entity_aliases` (alias graph)
- `daily_aggregates` (time-series metrics)
- `agent_jobs`, `agent_results` (agent queue + outputs)
- `event_enrichments` (most recent enrichment pointer per event)
- `daily_highlights` (nightly summaries)
- `token_vault` (encrypted reversible tokens)
- `prompt_library`, `prompt_ops_runs`
- `events.tags` can include `layout_blocks`, `layout_md`, and `ui_elements`
  for layout and UI-grounding evidence.

## UI
- Local FastAPI server serves `/` from `autocapture/ui/web/`.
- UI includes Chat, Search, Highlights, and Settings surfaces.

## Configuration
- `autocapture/config.py` defines configuration for mode, routing, privacy, retention, and PromptOps.
- `config/example.yml` is the reference configuration.

## Template Safety
- Prompt templates are treated as code and validated with a lint gate to block
  unsafe patterns before runtime; CI runs the lint gate.
- Production runs load templates only from trusted repo paths.

## Repo Memory
- `.memory/` stores append-only critical memory (constraints + ADRs), enforced by `.tools/memory_guard.py`.
