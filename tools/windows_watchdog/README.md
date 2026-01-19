# Windows Watchdog

This package runs a small fullscreen watchdog on Windows and toggles the Autocapture pause latch.

## Requirements
- Windows 11
- Python installed and on PATH

## Runtime directory
- Default: `C:\autocapture_runtime`
- Override with `AUTOCAPTURE_RUNTIME_DIR`.

## Run manually
1) Double-click `run_watchdog.bat`.
2) The watchdog writes:
   - `pause.flag`
   - `pause_reason.json`
   - `watchdog.log`

## Install on startup (no admin)
- Run `install_startup.bat` to create a Startup shortcut.
- Run `uninstall_startup.bat` to remove it.
- Keep the folder in a stable location. If moved, re-run install.

## Privacy
- Window titles are redacted by default.
- To include titles, set `WATCHDOG_INCLUDE_TITLES=1` before launching.

## Verify from WSL
- `ls /mnt/c/autocapture_runtime/pause.flag`
- `cat /mnt/c/autocapture_runtime/pause_reason.json`

## Troubleshooting
- `pywin32` install errors: ensure Python is on PATH and rerun `run_watchdog.bat`.
- Permissions: pick a runtime dir you can write to.
- Multiple monitors: adjust `WATCHDOG_TOL_PX` if fullscreen detection is flaky.
- Logging: check `watchdog.log` in the runtime directory.
