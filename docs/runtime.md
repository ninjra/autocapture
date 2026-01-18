# Runtime Gates

This doc describes runtime switches, pause latch semantics, and local gates.

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
