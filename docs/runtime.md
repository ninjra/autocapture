# Runtime Gates

This doc describes runtime switches, pause latch semantics, and local gates.

## Runtime governor modes
The runtime governor selects one of three modes based on fullscreen detection and input idle time:
- **FULLSCREEN_HARD_PAUSE:** capture and workers are paused; GPU release hooks fire once per transition.
- **ACTIVE_INTERACTIVE:** interactive defaults; lightweight background work only.
- **IDLE_DRAIN:** aggressive backlog drain when input is idle.

Mode transitions are logged with `mode`, `reason`, and `since_ts`. The latest state is also persisted
in the runtime state record (see `autocapture/runtime_governor.py`).

Relevant flags:
- `runtime.auto_pause.enabled`: enable fullscreen auto-pause.
- `runtime.auto_pause.fullscreen_hard_pause_enabled`: allow FULLSCREEN_HARD_PAUSE on fullscreen.
- `runtime.auto_pause.mode`: `hard` (workers + capture) or `soft` (capture only).
- `runtime.auto_pause.release_gpu`: fire GPU release hooks on fullscreen transitions.
- `runtime.qos.enabled`: enable QoS profiles for worker counts/batches.

## QoS budgets
QoS profiles live under `runtime.qos` and control worker counts, batch sizes, and optional budgets:
- `sleep_ms`: minimum sleep for paused loops.
- `max_batch`: max batch size hint.
- `max_concurrency`: max concurrency hint.
- `gpu_policy`: `allow_gpu|prefer_cpu|disallow_gpu|release_on_pause`.

Profiles are evaluated per mode, and the governor exposes `qos_budget()` for fast checks.

## Runtime env vars
- `GPU_MODE`: `auto` (default), `on`, or `off`.
- `PROFILE`: `foreground` or `idle` (optional override; when unset, runtime auto-selects).
- `AUTOCAPTURE_RUNTIME_DIR`: shared runtime directory for pause latch files.
- `AUTOCAPTURE_REDACT_WINDOW_TITLES`: `1` (default) to redact titles in pause metadata.
- `AUTOCAPTURE_CUDA_DEVICE_INDEX`: preferred CUDA device index (default `0`).
- `AUTOCAPTURE_CUDA_VISIBLE_DEVICES`: override `CUDA_VISIBLE_DEVICES`.

## Pause latch
Pause is controlled by files in the runtime dir:
- `pause.flag`: presence means paused.
- `pause_reason.json`: minimal metadata (reason, source, timestamp).

Workers block on pause and resume when the latch is removed. File updates are atomic.

## Profiles
Profiles adjust worker concurrency and batch sizing through `runtime.qos`:
- Foreground: interactive defaults (smaller worker counts, lower batch sizes).
- Idle: throughput defaults (larger worker counts, larger batch sizes).

Set `PROFILE=idle` to force idle profile; otherwise the runtime governor chooses based on input.

## Fullscreen hard pause behavior
When fullscreen hard pause is enabled:
- capture ticks are skipped
- worker loops do not acquire or renew leases
- GPU release hooks are invoked on transition into fullscreen

To override: set `runtime.auto_pause.fullscreen_hard_pause_enabled=false` (or `mode=soft`) in config.

## Local gates
```
make check
make check-gpu
make bench
make bench-cpu
make bench-gpu
```
Bench outputs are written to `artifacts/bench/` by default.

## Windows watchdog
See `tools/win_watchdog/README.md` for setup. Ensure `AUTOCAPTURE_RUNTIME_DIR` is mapped:
- Windows: `C:\autocapture_runtime`
- WSL: `/mnt/c/autocapture_runtime`

## Troubleshooting
- CUDA unavailable with `GPU_MODE=on`: install GPU backend or set `GPU_MODE=off`.
- Pause latch not seen in WSL: ensure runtime dir is shared under `/mnt/c`.
