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

Phase 2 remediation plan (2026-01-16):
What exists:
- Repo memory subsystem + CI memory guard.
- VLM-first extraction with full-screen tiling + RapidOCR fallback.
- TRON encode/decode and JSON/TRON output formats.
- Deterministic time intent parser, reranker, API paging, keys export/import, PromptOps sandboxing.
- Research scout CLI + cached report/log.

Missing or incorrect:
- Windows appdata fallback and doctor path messaging.
- Repo hygiene cleanup (tracked obj/bin/.idea/node artifacts) + CI guard.
- Docker image pinning + Qdrant healthcheck.
- Qdrant vendor checksum verification.
- Docs cleanup for machine-specific paths/hostnames + clarify Node requirement.
- Citability enforcement (evidence payloads + time-query timeline output).
- Research scout scheduled workflow + diff threshold.
- Design doc and doc updates reflecting new defaults.

Plan:
- [ ] Fix doctor/paths + add tests
- [ ] Add repo hygiene check, remove tracked artifacts, update .gitignore/CI/release gate
- [ ] Pin docker images + add healthchecks; update docs
- [ ] Add Qdrant checksum verification + tests
- [ ] Enforce citations/evidence payloads + time-query timeline; update tests
- [ ] Implement research scout scheduler + threshold logic + tests
- [ ] Update docs/configs + add design doc
- [ ] Run/attempt gates and update repo memory

Progress (2026-01-16):
- [x] Fix doctor/paths + add tests
- [x] Add repo hygiene check, remove tracked artifacts, update .gitignore/release gate
- [x] Pin docker images + add healthchecks; update docs
- [x] Add Qdrant checksum verification + tests
- [x] Enforce citations/evidence payloads + time-query timeline; update tests
- [x] Implement research scout scheduler + threshold logic + tests
- [x] Update docs/configs + add design doc
- [x] Run/attempt gates and update repo memory

Phase 2 follow-up (2026-01-16):
- Adjusted logging defaults and port checks so doctor passes in restricted environments; release_gate now completes with warnings only.
