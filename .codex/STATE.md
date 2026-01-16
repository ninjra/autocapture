# Codex State

Objective: Phase 5 - WSL bridge ingest/storage endpoints + dashboard widget.

Checklist:
- [x] Ensure ingest endpoint supports bridge token, privacy filter, idempotency
- [x] Add storage endpoint with cached size + TTL response
- [x] Update dashboard with Storage & Retention card
- [x] Add/adjust tests for ingest + storage
- [x] Fix python-multipart dependency and failing tests (scrypt params, paging config)
- [x] Run gates (ruff/black/pytest/doctor/freeze verify)
- [x] Update SESSION_LOG and PR_CHECKLIST
- [ ] Commit Phase 5 changes

Notes:
- pytest now passes; doctor still fails on `/home/ninjra/AppData` permission.

Phase 2+ audit (2026-01-15):
- Already present: retrieval reranker + tests; API paging with page/page_size; key export/import with scrypt+AES-GCM; PromptOps pipeline + basic Jinja2 guards; offline/cloud guard plumbing; frozen surface manifest.
- Missing/needs work: VLM-first extractor + tiling + structured tags; TRON encode/decode + output formats; time intent parsing + time-only retrieval; stage-based model routing + draft/final flow; research scout + proposals; critical memory subsystem + CI guard; docs/config/test updates.

Phase 2+ update (2026-01-16):
- Implemented VLM-first extraction with tiling + structured tags, TRON encode/decode + output formats, deterministic time intent parsing, and stage-based routing with optional drafts.
- Added research scout CLI with caching + proposal log, plus docs for VLM/TRON/time/stages and research watchlist.
- Added DPAPI portability test and tightened PromptOps sandbox checks.
- Pending: optional GitHub Actions scheduler for research scout; full test run after poetry install.
