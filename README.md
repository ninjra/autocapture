# Autocapture

A local-first desktop recall app for Windows 11 that runs as a single binary: tray UI + local web dashboard + background worker.

## Features

- **Tray + local web UI** with a lightweight search popup and dashboard.
- **Local capture + OCR** using DXCam (with MSS fallback) and on-device OCR.
- **Embedded search** with Qdrant-backed vector indexes and fast embeddings (falls back to lexical-only retrieval if Qdrant is unavailable).
- **Private by default** with local storage and optional cloud LLM fallback.

## Repository Layout

```text
autocapture/
  capture/            # Input hooks, screen capture, duplicate detection
  storage/            # Local storage, retention policies, encryption utilities
  config.py           # YAML configuration loader and validation models
  encryption.py       # AES-GCM helpers for local artifacts
  logging_utils.py    # Structured logging configuration
  main.py             # Legacy shim entry point
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
   python -m pip install --upgrade pip
   ```
2. Install dependencies:
   ```powershell
   python -m pip install -e .
   ```
3. Run Autocapture:
   ```powershell
   python -m autocapture
   ```
   On first run, Autocapture creates `%LOCALAPPDATA%/Autocapture/config.yml` and opens a folder picker to choose your data directory.

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

## Troubleshooting

- If PyCharm does not show newly added modules, follow the Git integration checklist in
  [`docs/troubleshooting.md`](docs/troubleshooting.md).

## License

MIT
