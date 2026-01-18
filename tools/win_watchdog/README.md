# Windows Fullscreen Watchdog

This watchdog runs on Windows and toggles the shared pause latch when a fullscreen app is detected.

## Runtime dir mapping
Set a shared runtime directory so Windows and WSL see the same files:

- Windows:
  - `setx AUTOCAPTURE_RUNTIME_DIR C:\autocapture_runtime`
- WSL:
  - `export AUTOCAPTURE_RUNTIME_DIR=/mnt/c/autocapture_runtime`

## Run
```
python tools/win_watchdog/watchdog.py --poll-hz 8
```

Flags:
- `--poll-hz`: polling rate (default 8 Hz)
- `--tolerance`: pixel tolerance for fullscreen detection (default 2)
- `--runtime-dir`: override runtime dir
- `--manual-flag`: name of manual pause latch file (default `manual_pause.flag`)
- `--redact-titles`: force window title redaction

Manual pause:
- Create `manual_pause.flag` in the runtime dir to force pause regardless of fullscreen state.

By default, window titles are redacted. Set `AUTOCAPTURE_REDACT_WINDOW_TITLES=0` to allow titles.
