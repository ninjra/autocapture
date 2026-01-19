# Autocapture Runtime

## Overview
The runtime system provides deterministic controls for:
- GPU selection (auto/on/off) and CUDA visibility
- Foreground vs idle tuning
- File-based pause latch with atomic reason writes
- Deterministic benchmark harness
- Windows fullscreen watchdog (optional)

## Shared runtime directory (Windows + WSL)
Default runtime dir is repo-local `.runtime/`.

Recommended shared setup:
- Windows: `C:\autocapture_runtime`
- WSL: `/mnt/c/autocapture_runtime`

Path normalization:
- In WSL/Linux, `AUTOCAPTURE_RUNTIME_DIR=C:\autocapture_runtime` is mapped to `/mnt/c/autocapture_runtime`.
- In Windows, `AUTOCAPTURE_RUNTIME_DIR=/mnt/c/autocapture_runtime` is mapped to `C:\autocapture_runtime`.

## Environment variables
Core:
- `AUTOCAPTURE_RUNTIME_DIR` (default: repo `.runtime/`)
- `AUTOCAPTURE_GPU_MODE` (auto|on|off, default: auto)
- `AUTOCAPTURE_PROFILE` (foreground|idle, default: foreground)
- `AUTOCAPTURE_PAUSE_LATCH` (default: `${RUNTIME_DIR}/pause.flag`)
- `AUTOCAPTURE_PAUSE_REASON` (default: `${RUNTIME_DIR}/pause_reason.json`)
- `AUTOCAPTURE_BENCH_DIR` (default: `artifacts/bench`)
- `AUTOCAPTURE_LOG_DIR` (default: `artifacts/logs`)

Profile tuning (both prefixed and unprefixed forms are supported):
- `AUTOCAPTURE_FOREGROUND_MAX_WORKERS` / `FOREGROUND_MAX_WORKERS`
- `AUTOCAPTURE_FOREGROUND_BATCH_SIZE` / `FOREGROUND_BATCH_SIZE`
- `AUTOCAPTURE_FOREGROUND_POLL_INTERVAL_MS` / `FOREGROUND_POLL_INTERVAL_MS`
- `AUTOCAPTURE_FOREGROUND_MAX_QUEUE_DEPTH` / `FOREGROUND_MAX_QUEUE_DEPTH`
- `AUTOCAPTURE_FOREGROUND_MAX_CPU_PCT_HINT` / `FOREGROUND_MAX_CPU_PCT_HINT`
- `AUTOCAPTURE_IDLE_MAX_WORKERS` / `IDLE_MAX_WORKERS`
- `AUTOCAPTURE_IDLE_BATCH_SIZE` / `IDLE_BATCH_SIZE`
- `AUTOCAPTURE_IDLE_POLL_INTERVAL_MS` / `IDLE_POLL_INTERVAL_MS`
- `AUTOCAPTURE_IDLE_MAX_QUEUE_DEPTH` / `IDLE_MAX_QUEUE_DEPTH`
- `AUTOCAPTURE_IDLE_MAX_CPU_PCT_HINT` / `IDLE_MAX_CPU_PCT_HINT`

## Make targets
- `make check`
- `make check-gpu`
- `make dev-fast`
- `make dev-idle`
- `make bench MODE=both`

Equivalent one-liners:
- `AUTOCAPTURE_GPU_MODE=off AUTOCAPTURE_PROFILE=foreground poetry run pytest -q -m "not gpu"`
- `AUTOCAPTURE_GPU_MODE=auto AUTOCAPTURE_PROFILE=foreground poetry run pytest -q -m gpu`
- `AUTOCAPTURE_PROFILE=foreground poetry run python -m autocapture.bench.run --mode both`

## Pause latch behavior
- `pause.flag` exists when paused.
- `pause_reason.json` is written atomically and includes:
  - `reason`, `source`, `ts_ms` and optional metadata keys.
- The latch is created after the reason file write to avoid partial reads.

Manual pause/resume:
- Pause: create `pause.flag` and optionally write `pause_reason.json`.
- Resume: delete `pause.flag`. Delete `pause_reason.json` only if you own the source.

## Windows watchdog
See `tools/windows_watchdog/README.md`.

Quick steps (Windows):
1) Optional: set `AUTOCAPTURE_RUNTIME_DIR=C:\autocapture_runtime`.
2) Run `tools\windows_watchdog\install_startup.bat`.
3) Run `tools\windows_watchdog\run_watchdog.bat` to test.

Verify from WSL:
- `ls /mnt/c/autocapture_runtime/pause.flag`
- `cat /mnt/c/autocapture_runtime/pause_reason.json`

## Troubleshooting
- GPU missing: set `AUTOCAPTURE_GPU_MODE=off` or `auto`.
- Latch stuck: remove `pause.flag`; verify `pause_reason.json` source before deleting.
- Watchdog issues: inspect `watchdog.log` and ensure `pywin32` installed.
- Fullscreen detection: tune `WATCHDOG_TOL_PX` for your monitor layout.
