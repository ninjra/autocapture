# Autocapture Architecture

## Overview
Autocapture is a local-first Personal Activity Memory Engine. It captures activity, stores evidence locally, and answers questions using context packs with verifiable citations.

## Pipelines

### Capture
- `autocapture/capture/` handles screenshot capture, metadata, and OCR staging.
- Events are persisted in `events` with OCR text and spans.

### Agents + Enrichment
- `autocapture/agents/` orchestrates enrichment, vision captioning, and nightly highlights.
- Agent jobs are stored in `agent_jobs`/`agent_results` with leasing + retries.
- Enrichment and vision outputs are augmentation-only, stored in tags + synthetic indexes.

### Retrieval + Context Pack
- `autocapture/memory/retrieval.py` fetches event evidence from SQLite.
- `autocapture/memory/context_pack.py` formats JSON + canonical text context packs.
  - Daily highlight snippets are attached under `aggregates`.

### Providers + Routing
- `autocapture/memory/router.py` selects providers per layer.
- Layers: OCR, embeddings, retrieval, reranking, compression, verification, LLM.

### Privacy + Entity Resolution
- `autocapture/memory/entities.py` manages stable pseudonyms and alias mapping.
- Tokens are derived from HMAC with a local secret key.
- `token_vault` stores reversible mappings for emails/domains/paths (encrypted at rest).

## Data Model
The local SQLite DB contains:
- `events` (activity + OCR evidence)
- `entities`, `entity_aliases` (alias graph)
- `daily_aggregates` (time-series metrics)
- `agent_jobs`, `agent_results` (agent queue + outputs)
- `event_enrichments` (latest enrichment pointer per event)
- `daily_highlights` (nightly summaries)
- `token_vault` (encrypted reversible tokens)
- `prompt_library`, `prompt_ops_runs`

## UI
- Local FastAPI server serves `/` from `autocapture/ui/web/`.
- UI includes Chat, Search, Highlights, and Settings surfaces.

## Configuration
- `autocapture/config.py` defines configuration for mode, routing, privacy, retention, and PromptOps.
- `config/example.yml` is the reference configuration.
