# SPEC-SINGLE-MACHINE audit (2026-01-19)

## Baseline
- Clean tree before changes: `git status -sb` → `## main...origin/main`, `git diff --stat` empty.
- Baseline HEAD: `1c945996eaec23422644fb681d651df448a65369`.
- Branch created: `spec-single-machine-plugin-system`.

## Entry points + config load
- CLI entrypoint: `autocapture/main.py` (argparse, `main()`), script hook in `pyproject.toml` → `autocapture = "autocapture.main:main"`.
- Config load: `autocapture/config.py` → `load_config()` reads YAML, applies legacy key mapping, then `apply_settings_overrides()`.
- Settings overrides path: `Path(config.capture.data_dir) / "settings.json"` in `apply_settings_overrides()`.

## settings.json read/write + UI wiring
- Persistence helpers: `autocapture/settings_store.py` (`read_settings`, `write_settings`, `update_settings`).
- API settings endpoints: `autocapture/api/server.py` → `POST /api/settings` (merge + write) and `GET /api/settings` (returns defaults if missing).
- Export includes settings: `autocapture/export.py` writes `settings.json` into bundle and updates backup timestamp.
- UI settings save/load: `autocapture/ui/web/app.js` reads `/api/settings` and posts routing + llm settings; routing uses DOM ids `routingOcr`, `routingEmbedding`, `routingRetrieval`, `routingCompressor`, `routingVerifier`.

## Router switching points + guardrails (current strings)
- StageRouter (LLM): `autocapture/model_ops/router.py`
  - Providers: `ollama`, `openai`, `openai_compatible` (also uses `config.routing.llm`, `config.llm.provider`, per-stage provider override).
  - Cloud gating: `_guard_cloud()` enforces `output.allow_tron_compression`, `model_stages.<stage>.allow_cloud`, `offline=false`, `privacy.cloud_enabled=true`.
- ProviderRouter (pipeline routing): `autocapture/memory/router.py`
  - LLM routing logic for `openai`, `openai_compatible`, `ollama` (uses env keys if config missing).
  - NOTE: only `openai` is gated by `privacy.cloud_enabled` + `offline`; `openai_compatible` has no cloud/offline guard here, and no `output.allow_tron_compression` check.
  - Embedding/reranker/ocr routing uses `config.routing.embedding|reranker|ocr` strings; no retrieval/compressor/verifier routing found in runtime code.
- ScreenExtractorRouter (vision): `autocapture/vision/extractors.py`
  - Engines: `vlm`, `qwen-vl`, `deepseek-ocr`, `deepseek`, `rapidocr`, `rapidocr-onnxruntime`, `disabled/off`.
  - Fallback: `vision_extract.fallback_engine`.
  - Cloud image gating: `_cloud_images_allowed()` requires non-local endpoint + `allow_cloud`, `offline=false`, `privacy.cloud_enabled=true`, `privacy.allow_cloud_images=true`.
- UI grounding vision uses same cloud image gating: `autocapture/vision/ui_grounding.py`.
- Additional gating: `autocapture/capture/frame_record.py` computes `cloud_allowed = privacy.cloud_enabled && allow_cloud_images && !offline`.
- OCR worker gating: `autocapture/worker/supervisor.py` disables OCR workers when routing or engine is disabled.

## Routing defaults & back-compat strings
- Defaults in `ProviderRoutingConfig` (`autocapture/config.py`):
  - `capture=local`, `ocr=local`, `embedding=local`, `retrieval=local`, `reranker=disabled`, `compressor=extractive`, `verifier=rules`, `llm=ollama`.
- Stage config: `ModelStageConfig` includes `provider`, `allow_cloud`, `model`, `base_url`, `api_key` (used by StageRouter).
- NOTE: `routing.retrieval`, `routing.compressor`, `routing.verifier` are present in config/UI but not referenced in runtime routing logic (rg hits only in UI/settings).

## Existing implementations to wrap as built-in plugins
- LLM providers: `autocapture/llm/providers.py` (`OllamaProvider`, `OpenAIProvider`, `OpenAICompatibleProvider`).
- Vision/OCR: `autocapture/vision/extractors.py` (RapidOCR, VLM, DeepSeek OCR) + `autocapture/vision/rapidocr.py`.
- Embeddings: `autocapture/embeddings/service.py` + `autocapture/embeddings/sparse.py`, `autocapture/embeddings/late.py`.
- Retrieval: `autocapture/memory/retrieval.py` + `autocapture/memory/store.py`.
- Reranker: `autocapture/memory/reranker.py`.
- Compressor: `autocapture/memory/compression.py` (extractive).
- Verifier: `autocapture/memory/verification.py` (RulesVerifier).
- Vector backend: `autocapture/indexing/vector_index.py` (VectorBackend protocol + Qdrant); sidecar in `autocapture/qdrant/sidecar.py`.
- Research: CLI in `autocapture/main.py` → `autocapture/research/scout.py`.
- UI surfaces: `autocapture/ui/web/index.html`, `autocapture/ui/web/app.js`.

## API server + health checks
- FastAPI app construction: `autocapture/api/server.py` (`app = FastAPI(...)`).
- Health checks: `autocapture/api/server.py` includes embedder/Qdrant checks; `autocapture/doctor.py` has OCR/embeddings/Qdrant checks.

## UI routing dropdowns (hard-coded)
- Static options in `autocapture/ui/web/index.html`:
  - Model select: Local (Ollama), Cloud (OpenAI).
  - Routing defaults: OCR/local, Embeddings/local, Retrieval/local, Compressor/extractive+abstractive, Verifier/rules.
- Cloud/local heuristic: `autocapture/ui/web/app.js` uses `routing.llm.startsWith('openai')` to set model selection.

## Packaging / bundling
- `pyproject.toml` includes package data for `autocapture/prompts/**/*.yaml`, `autocapture/ui/web/**`, `autocapture/bench/fixtures/**`.
- `pyinstaller.spec` bundles `autocapture/ui/web`, `autocapture/prompts/derived/*.yaml`, `autocapture.yml`, `alembic`, and vendor `ffmpeg`/`qdrant` trees.
- No plugin manifest/assets paths are currently bundled (would need to add for SPEC-SINGLE-MACHINE).

## SPEC-SINGLE-MACHINE implementation updates (2026-01-19)
- Plugin core: `autocapture/plugins/` (catalog, manifest models, registry, policy gate, manager, hashing, settings).
- Built-in manifests: `autocapture/plugins/builtin/**/plugin.yaml` (LLM, vision/OCR, embedder, retrieval, reranker, compressor, verifier, vector, prompts, research).
- API management surfaces: `autocapture/api/server.py` (`/api/plugins/*`) + asset route `/plugins/{plugin_id}/assets/*`.
- CLI commands: `autocapture/main.py` → `autocapture plugins list|enable|disable|lock|doctor`.
- UI routing + plugins tab: `autocapture/ui/web/index.html`, `autocapture/ui/web/app.js`, `autocapture/ui/web/styles.css`.
- Packaging: `pyproject.toml` + `pyinstaller.spec` now include `autocapture/plugins/builtin/**`.
