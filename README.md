# Autocapture

A local-first desktop recall app for Windows 11 that runs as a single binary: tray UI + local web dashboard + background worker. Docker is optional for advanced, remote deployments.

## Features

- **Tray + local web UI** with a lightweight search popup and dashboard.
- **VLM-first screen understanding** using DXCam (with MSS fallback), full-screen tiling, and a RapidOCR fallback.
- **Embedded search** with SQLite-backed vector + spans_v2 indexes by default. Qdrant remains optional via routing overrides.
- **Private by default** with local storage and optional cloud LLM fallback.
- **Time-aware Q&A** with deterministic time parsing, citations, and optional TRON/JSON outputs.
- **Stage-routed LLM pipeline** with query refinement, draft generation, and final answer stages (local-first defaults).
- **LLM gateway** (OpenAI-compatible proxy) with stage policy fallback and claim-level validation.
- **Graph adapters** for graph-style retrieval workers (GraphRAG / HyperGraphRAG / Hyper-RAG).
- **Operational UX facade**: `/api/state` snapshot, schema-driven settings preview/apply, doctor diagnostics, audit views, and safe delete flows.
- **Optional DiffusionVL local server** (`tools/diffusionvl_server.py`) for OpenAI-compatible VLM hosting.
- **Deterministic memory store CLI** (SQLite + FTS5) with snapshots and citations.

## Phase 3 Highlights (SPEC-4)

- **Runtime governor + fullscreen hard pause** with GPU lease release and auto-resume.
- **Hybrid retrieval v2**: multi-query + RRF fusion, sparse retrieval, and late-interaction reranking (flagged).
- **Region-level citations**: span bboxes included in context packs and optional overlay rendering.
- **PP-Structure layout** (optional): `ocr.paddle_ppstructure_*` for local PaddleOCR layouts.

See `docs/PHASE3.md` for the full requirement mapping and rollout flags.
Plugin system details are in `docs/plugins.md`.

## Repository Layout

```text
autocapture/
  capture/            # Input hooks, screen capture, duplicate detection
  storage/            # Local storage, retention policies, encryption utilities
  config.py           # YAML configuration loader and validation models
  encryption.py       # AES-GCM helpers for local artifacts
  logging_utils.py    # Structured logging configuration
  main.py             # CLI entry point (run/api/tray)
  __main__.py         # Primary CLI entry point
config/
  example.yml         # Reference configuration with detailed tuning knobs
pyproject.toml        # Project dependencies and tooling configuration
```

## Quickstart

1. Create a virtual environment:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1  # PowerShell (use activate.bat for cmd.exe)
   python -m pip install --upgrade pip poetry
   ```
2. Install dependencies:
   ```powershell
   poetry install --with dev
   ```
   For the full Windows app (tray UI + capture + OCR + embeddings):
   ```powershell
   poetry install --with dev --extras "ui windows ocr ocr-gpu embed-fast"
   # Optional extras: embed-st
   ```
3. Run Autocapture:
   ```powershell
   poetry run autocapture
   ```
   On first run, Autocapture writes a default config to `%LOCALAPPDATA%/Autocapture/autocapture.yml`
   on Windows and stores data under `%LOCALAPPDATA%/Autocapture/data` unless overridden.
   Video recording is enabled by default, so ensure FFmpeg is available (bundle it
   via `tools/vendor_windows_binaries.py` or set `ffmpeg.explicit_path`).

   For a full GPU + security run (recommended for representative testing):
   ```powershell
   poetry run autocapture setup --profile full --apply
   poetry run autocapture doctor --verbose
   poetry run autocapture app
   ```

### Windows release bundling

Windows release builds can bundle Qdrant + FFmpeg so local mode runs without Docker when Qdrant
is explicitly selected via routing. Use the vendor bootstrap before packaging:

```powershell
python tools/vendor_windows_binaries.py
```

### Development & Testing

```powershell
poetry install --with dev
poetry run ruff check .
poetry run black --check .
poetry run pytest -q
```

Node.js is not required for the core workflows; the UI assets are bundled in the repo.

### Spec and Schema Validation

- Canonical spec: `BLUEPRINT.md`
- Validate: `python tools/validate_blueprint.py BLUEPRINT.md`
- Grounded SQL artifacts: `docs/schemas/*.sql`

PowerShell helper:
```powershell
.\dev.ps1 check
.\dev.ps1 smoke
```

### WSL2 local services

See `docs/runbook-single-machine.md` for the full SPEC-SINGLE-MACHINE runbook. Quick summary:

```bash
docker compose -f infra/compose.yaml up -d
scripts/run_graph_worker.sh
scripts/run_gateway.sh
scripts/run_api.sh
scripts/run_vllm_gpu_a.sh <model-name>
scripts/run_vllm_gpu_b.sh <model-name>
scripts/run_vllm_cpu.sh <model-name>
```

Configure endpoints in `autocapture.yml`:
- `gateway.*`
- `graph_service.*`
- `model_registry.*`
- `retrieval.graph_adapters.*`

### AI assistance (model preference)

- Default to the highest reasoning/thinking model available for AI-assisted changes
  and reviews.
- If your account supports a "Pro Extended" tier/mode, prefer it for non-trivial work.
- When cost/latency requires a faster model, note that choice in the PR description and
  rerun any critical reasoning steps on the highest tier when possible.
- Keep model selection local (tool settings or environment variables); do not commit
  secrets or personal overrides.

### Daily Use

- `Ctrl+Shift+Space` opens the search popup.
- Open the dashboard link from the tray menu.
- Pause/resume capture from the tray menu (hotkey is reserved for search).

## Canonical pipeline

- `autocapture run` starts the CaptureOrchestrator (captures → `CaptureRecord`) and the
  EventIngestWorker (`CaptureRecord` → `EventRecord` via OCR).
- The API server retrieves from `EventRecord` and builds ContextPack v1 responses.

Legacy/experimental modules exist (for example `autocapture/capture/service.py` and
`autocapture/worker/worker_main.py`) and are not the canonical path used by `main.py` run/api.

## PromptOps

PromptOps automates prompt refreshes and evals.

```powershell
poetry run autocapture promptops run
poetry run autocapture promptops status
poetry run autocapture promptops list
```

## Troubleshooting

- If PyCharm does not show newly added modules, follow the Git integration checklist in
  [`docs/troubleshooting.md`](docs/troubleshooting.md).

## Observability

- Prometheus metrics are exposed on the configured `observability.prometheus_port`.
- Import the Grafana dashboard from `docs/dashboard.json`:
  1. Grafana → Dashboards → Import.
  2. Upload `docs/dashboard.json`.
  3. Select your Prometheus data source.

## Health checks

- `GET /healthz/deep` performs a deep dependency check (DB + optional Qdrant/embedding).
  It returns HTTP 503 when any non-skipped check fails.
- `GET /api/state` returns the canonical state snapshot used by the web UI + CLI.
- `GET /api/doctor` returns structured diagnostic checks (same payload as `autocapture doctor --json`).

## UX CLI

```powershell
poetry run autocapture status --json
poetry run autocapture settings schema --json
poetry run autocapture settings preview --file settings.json --tier guided
poetry run autocapture settings apply --file settings.json --preview-id <token> --confirm
poetry run autocapture delete preview range --start-utc 2026-01-01T00:00:00Z --end-utc 2026-01-02T00:00:00Z
poetry run autocapture audit list --json
```

## Export / backup

Create a portable backup for the last 90 days of events (default):

```powershell
poetry run autocapture export --out "%LOCALAPPDATA%/Autocapture/export.zip"
```

Use `--no-zip` to write a folder instead of a zip archive, and `--days N` to change the
date window.

## Research scout

Generate a local model/paper discovery report:

```powershell
poetry run autocapture research scout --out "docs/research/scout_report.json"
```

Scheduled runs are available via `.github/workflows/research-scout.yml`, which
opens a PR only when the ranked list changes beyond the configured threshold.

## Repo memory

This repo maintains an append-only critical memory in `.memory/`. Validate it with:

```powershell
python .tools/memory_guard.py --check
```

## License

MIT
