# Codex State

Objective: Phase 4 - RapidOCR hardening + default retention TTL.

Checklist:
- [x] Locate RapidOCR kwargs builder and remove signature inspection
- [x] Pin ONNX provider settings in config (CUDA then CPU) and update doctor output
- [x] Add RapidOCR kwargs/provider selection tests
- [x] Update screenshot TTL default to 60 days + config/example/docs
- [x] Add tests for retention default
- [x] Run gates (ruff/black/pytest/doctor/freeze verify)
- [x] Update SESSION_LOG and PR_CHECKLIST
- [x] Commit Phase 4 changes

Notes:
- pytest currently times out at 300s; doctor currently fails on `/home/ninjra/AppData` permission.
