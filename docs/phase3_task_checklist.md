# Phase 3 Task Checklist (Spec Pack)

Machine-checkable checklist derived from the Phase 3 task pack (Tasks 1–10).
Each task includes required flags + defaults, modules/files, wiring points, tests,
docs/CI changes, and acceptance criteria.

## Task 1 — RuntimeGovernor (fullscreen hard pause + QoS)
- [ ] Flags/defaults
  - [ ] runtime.auto_pause.enabled (default: false if uncertain; safe-off)
  - [ ] runtime.auto_pause.poll_hz (default: conservative)
  - [ ] runtime.auto_pause.fullscreen_hard_pause_enabled (default: false unless safe-on)
  - [ ] runtime.auto_pause.thresholds.active_input_window_s (default: conservative)
  - [ ] runtime.auto_pause.thresholds.idle_after_s (default: conservative)
  - [ ] runtime.qos.enabled (default: false unless preserves prior behavior)
  - [ ] runtime.qos.FULLSCREEN_HARD_PAUSE.sleep_ms
  - [ ] runtime.qos.FULLSCREEN_HARD_PAUSE.max_batch
  - [ ] runtime.qos.FULLSCREEN_HARD_PAUSE.max_concurrency
  - [ ] runtime.qos.FULLSCREEN_HARD_PAUSE.gpu_policy
  - [ ] runtime.qos.ACTIVE_INTERACTIVE.sleep_ms
  - [ ] runtime.qos.ACTIVE_INTERACTIVE.max_batch
  - [ ] runtime.qos.ACTIVE_INTERACTIVE.max_concurrency
  - [ ] runtime.qos.ACTIVE_INTERACTIVE.gpu_policy
  - [ ] runtime.qos.IDLE_DRAIN.sleep_ms
  - [ ] runtime.qos.IDLE_DRAIN.max_batch
  - [ ] runtime.qos.IDLE_DRAIN.max_concurrency
  - [ ] runtime.qos.IDLE_DRAIN.gpu_policy
- [ ] Modules/files
  - [ ] autocapture/runtime/governor.py (RuntimeMode enum, RuntimeGovernor)
  - [ ] autocapture/runtime/__init__.py exports
- [ ] Wiring points
  - [ ] Single shared RuntimeGovernor instance (app/supervisor init)
  - [ ] Detectors injectable; safe fallbacks
  - [ ] snapshot() returns immutable struct
  - [ ] allow_workers() returns false when FULLSCREEN_HARD_PAUSE + auto_pause enabled
  - [ ] qos_budget() returns per-mode budgets w/ safe defaults when qos disabled
- [ ] Tests
  - [ ] tests/test_runtime_governor.py (fullscreen → FULLSCREEN_HARD_PAUSE)
  - [ ] tests/test_runtime_governor.py (recent input → ACTIVE_INTERACTIVE)
  - [ ] tests/test_runtime_governor.py (idle → IDLE_DRAIN)
  - [ ] tests/test_runtime_governor.py (reason stable, since_ts monotonic)
- [ ] Docs/CI
  - [ ] docs/runtime.md updated (modes + transitions)
  - [ ] config/example.yml updated with runtime flags
- [ ] Acceptance criteria
  - [ ] Mode transitions deterministic with stable reason strings
  - [ ] allow_workers/qos_budget behave as specified

## Task 2 — Hard pause all workers when FULLSCREEN_HARD_PAUSE
- [ ] Wiring points
  - [ ] Worker supervisor uses shared RuntimeGovernor
  - [ ] Each worker loop checks allow_workers() before any lease acquisition
  - [ ] No DB writes or lease heartbeat/renewal during hard pause
  - [ ] Sleeps use governor.qos_budget().sleep_ms
- [ ] Tests
  - [ ] tests/test_fullscreen_hard_pause_integration.py
    - [ ] Forces fullscreen mode
    - [ ] Runs worker loop for bounded window
    - [ ] Asserts zero DB writes + no lease advancement
    - [ ] SKIP if external deps required
- [ ] Acceptance criteria
  - [ ] Hard pause fully suppresses worker work + leases

## Task 3 — GPU release on fullscreen
- [ ] Flags/defaults
  - [ ] runtime.auto_pause.release_gpu (default: safe-on if already used)
  - [ ] llm.ollama_keep_alive (default: unset/None)
- [ ] Modules/files
  - [ ] autocapture/runtime/gpu_lease.py (register_release_hook + release_all + logging)
  - [ ] OCR engine .close() idempotent, safe on CPU
  - [ ] Reranker .close() + register hook
  - [ ] Ollama client keep_alive + unload_models hook
- [ ] Wiring points
  - [ ] Governor calls gpu_lease.release_all() on transition into fullscreen
  - [ ] Hooks called once per transition; exceptions logged
  - [ ] hard_offline honored (no network calls)
- [ ] Tests
  - [ ] tests/test_gpu_release_hooks.py (hook calls once per transition)
  - [ ] Optional NVML VRAM drop check (SKIP if unavailable)
- [ ] Acceptance criteria
  - [ ] Fullscreen pause triggers GPU release hooks, idempotent and safe

## Task 4 — OCR → layout → markdown (deterministic)
- [ ] Modules/files
  - [ ] autocapture/ocr/layout.py (deterministic layout blocks + markdown)
  - [ ] autocapture/worker/ocr_worker.py stores layout_blocks + layout_md tags
  - [ ] Retrieval payload includes layout_md text when present
- [ ] Determinism
  - [ ] Stable sort by y, then x, then stable id/index
  - [ ] Explicit tie-breakers for identical coords
  - [ ] Stable rounding/normalization of numeric values
- [ ] Tests
  - [ ] tests/test_ocr_layout_md.py (golden markdown + block ordering)
  - [ ] tests/test_ocr_layout_md.py (tie-case deterministic ordering)
- [ ] Acceptance criteria
  - [ ] Empty spans → "" and [] deterministically stored

## Task 5 — Qdrant schema v2 (named vectors)
- [ ] Flags/defaults
  - [ ] retrieval.use_spans_v2 (default: false)
- [ ] Modules/files
  - [ ] autocapture/indexing/qdrant_schema.py (v2 collection + named vectors)
  - [ ] autocapture/indexing/vector_index.py + worker/embed_worker.py
- [ ] Wiring points
  - [ ] Create v2 collection if missing (no destructive migration)
  - [ ] Dense + sparse + late named vectors in upserts + queries
  - [ ] Deterministic upsert ordering
  - [ ] hard_offline blocks network if Qdrant remote
- [ ] Tests
  - [ ] tests/test_qdrant_named_vectors_smoke.py
    - [ ] Integration if Qdrant reachable
    - [ ] Contract/mocked shape if unavailable (SKIP with reason)
- [ ] Acceptance criteria
  - [ ] v2 path works behind flag; v1 unaffected

## Task 6 — Fusion retrieval (multi-query + RRF) + late rerank
- [ ] Modules/files
  - [ ] autocapture/retrieval/fusion.py (RRF + deterministic ties)
  - [ ] autocapture/retrieval/query_rewrite.py (gated rewrites + offline)
  - [ ] autocapture/retrieval/retrieval_service.py (fusion pipeline)
- [ ] Wiring points
  - [ ] Multi-query rewrites gated by confidence/flags
  - [ ] RRF tie-breakers: fused_score → best_rank → stable id
  - [ ] Late rerank respects governor QoS + gpu_policy
  - [ ] FULLSCREEN_HARD_PAUSE disables rerank
- [ ] Tests
  - [ ] tests/test_rrf_fusion.py (RRF correctness + tie-breakers)
  - [ ] tests/test_rrf_fusion.py (rewrite gating)
  - [ ] tests/test_rrf_fusion.py (late rerank ordering)
- [ ] Acceptance criteria
  - [ ] Deterministic fused ordering

## Task 7 — Retrieval-aware generation (R²AG) + speculative early exit
- [ ] Modules/files
  - [ ] autocapture/memory/context_pack.py (retrieval signals)
  - [ ] autocapture/memory/answer_graph.py (speculative verify + deep pass)
- [ ] Wiring points
  - [ ] Retrieval signals include scores, ranks, gaps, matched spans, per-engine provenance
  - [ ] Verifier always run for speculative + deep paths
  - [ ] Early exit only if high-confidence + verified
- [ ] Tests
  - [ ] tests/test_speculative_answering.py (verifier enforced)
  - [ ] tests/test_speculative_answering.py (early exit only on confidence)
  - [ ] tests/test_speculative_answering.py (low-confidence forces deep)
- [ ] Acceptance criteria
  - [ ] Verifier cannot be bypassed

## Task 8 — UI Grounding extractor
- [ ] Flags/defaults
  - [ ] vision_extract.ui_grounding.enabled (default: false)
  - [ ] UI grounding scheduling default: IDLE_DRAIN only
- [ ] Modules/files
  - [ ] autocapture/vision/ui_grounding.py (schema validation + normalization)
  - [ ] autocapture/worker/vision_extract_worker.py (store ui_elements)
- [ ] Wiring points
  - [ ] Strict schema rejects malformed/out-of-range
  - [ ] Deterministic normalization: sort + rounding + stable IDs
  - [ ] hard_offline: no remote calls; local-only if available
- [ ] Tests
  - [ ] tests/test_ui_grounding_schema.py (accept valid)
  - [ ] tests/test_ui_grounding_schema.py (reject invalid)
  - [ ] tests/test_ui_grounding_schema.py (deterministic normalization)
- [ ] Acceptance criteria
  - [ ] UI grounding produces deterministic ui_elements tags

## Task 9 — Template security hardening gates
- [ ] Modules/files
  - [ ] autocapture/promptops/validate.py (validator CLI)
  - [ ] .github/workflows/*.yml (CI gate step)
- [ ] Wiring points
  - [ ] Validator rejects unsafe template constructs
  - [ ] Actionable errors w/ line/field
  - [ ] CI runs validator on PRs
- [ ] Tests
  - [ ] tests/test_prompt_template_security.py (blocked payload)
  - [ ] tests/test_prompt_template_security.py (known-safe pass)
- [ ] Acceptance criteria
  - [ ] CI gate prevents unsafe template merges

## Task 10 — Documentation + operator playbook
- [ ] Docs
  - [ ] runtime modes + transitions
  - [ ] fullscreen hard pause behavior + overrides
  - [ ] QoS tuning + retention budgets
  - [ ] troubleshooting: “Why is capture paused?”
- [ ] Config
  - [ ] config/example.yml updated with all new flags + defaults
  - [ ] hard_offline notes included
- [ ] Acceptance criteria
  - [ ] Operator docs reflect new flags + behavior
