# Autocapture Architecture

## Overview
Autocapture is a local-first Personal Activity Memory Engine. It captures activity, stores evidence locally, and answers questions using context packs with verifiable citations.

## Pipelines

### Capture
- `autocapture/capture/` handles screenshot capture, metadata, and OCR staging.
- Events are persisted in `events` with screen-text transcripts and spans.

### Vision Extraction
- `autocapture/vision/` runs VLM-first extraction with full-screen tiling.
- Structured layout output is stored under `EventRecord.tags["vision_extract"]` and the
  consolidated transcript is stored in `EventRecord.ocr_text` for retrieval compatibility.

### Agents + Enrichment
- `autocapture/agents/` orchestrates enrichment, vision captioning, and nightly highlights.
- Agent jobs are stored in `agent_jobs`/`agent_results` with leasing + retries.
- Enrichment and vision outputs are augmentation-only, stored in tags + synthetic indexes.
- `autocapture/enrichment/scheduler.py` scans recent events to enqueue missing
  vision_extract/enrichment/thread_summary jobs and tracks backlog + at-risk metrics.
- SQL/code artifacts are extracted deterministically and indexed for lexical + vector search.

### Retrieval + Context Pack
- `autocapture/memory/retrieval.py` fetches event evidence from SQLite.
- `autocapture/memory/context_pack.py` formats JSON + canonical text context packs.
  - Daily highlight snippets are attached under `aggregates`.
  - Evidence payloads include event_id, timestamps, and screenshot metadata for citations.
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
- `threads`, `thread_events`, `thread_summaries` (activity threads + summaries)
- `entities`, `entity_aliases` (alias graph)
- `daily_aggregates` (time-series metrics)
- `agent_jobs`, `agent_results` (agent queue + outputs)
- `event_enrichments` (most recent enrichment pointer per event)
- `daily_highlights` (nightly summaries)
- `token_vault` (encrypted reversible tokens)
- `prompt_library`, `prompt_ops_runs`

## UI
- Local FastAPI server serves `/` from `autocapture/ui/web/`.
- UI includes Chat, Search, Highlights, and Settings surfaces.

## Configuration
- `autocapture/config.py` defines configuration for mode, routing, privacy, retention, and PromptOps.
- `config/example.yml` is the reference configuration.

## Repo Memory
- `.memory/` stores append-only critical memory (constraints + ADRs), enforced by `.tools/memory_guard.py`.
