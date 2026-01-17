# Autocapture

A local-first desktop recall app for Windows 11 that runs as a single binary: tray UI + local web dashboard + background worker. Docker is optional for advanced, remote deployments.

## Features

- **Tray + local web UI** with a lightweight search popup and dashboard.
- **VLM-first screen understanding** using DXCam (with MSS fallback), full-screen tiling, and a RapidOCR fallback.
- **Embedded search** with Qdrant-backed vector indexes and fast embeddings (falls back to lexical-only retrieval if Qdrant is unavailable). Windows release builds bundle a local Qdrant sidecar.
- **Private by default** with local storage and optional cloud LLM fallback.
- **Time-aware Q&A** with deterministic time parsing, citations, and optional TRON/JSON outputs.
- **Stage-routed LLM pipeline** with query refinement, draft generation, and final answer stages (local-first defaults).
- **Optional DiffusionVL local server** (`tools/diffusionvl_server.py`) for OpenAI-compatible VLM hosting.

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

### Windows release bundling

Windows release builds bundle Qdrant + FFmpeg so local mode runs without Docker. Use the vendor
bootstrap before packaging:

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

PowerShell helper:
```powershell
.\dev.ps1 check
.\dev.ps1 smoke
```

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
