# Troubleshooting

## Why is capture paused?

When capture appears paused, check these common causes:

1. **Fullscreen hard pause.**
   - Verify `runtime.auto_pause.enabled=true`.
   - If a fullscreen app is active, the governor enters `FULLSCREEN_HARD_PAUSE`.
   - Inspect the latest runtime state (`pause_reason`, `since_ts`) from logs or the runtime state
     record persisted by `RuntimeGovernor`.

2. **Manual pause latch.**
   - If `pause.flag` exists in the runtime directory, capture/worker loops block until it is removed.
   - See `docs/runtime.md` for pause latch details.

3. **Privacy/overlay rules.**
   - `privacy.paused=true` or `privacy.snooze_until_utc` can stop capture.
   - Exclusion filters (`privacy.exclude_processes`, `privacy.exclude_window_title_regex`) can
     suppress captures for specific apps/windows.

If you need capture to continue during fullscreen apps, set
`runtime.auto_pause.fullscreen_hard_pause_enabled=false` (or `runtime.auto_pause.mode=soft`).

## WSL path and GPU checks

- If `autocapture doctor` reports `windows_path_on_posix`, update paths to POSIX
  (e.g. `/mnt/c/Autocapture`) or set `paths.base_dir` in `autocapture.yml`.
- If GPU checks fail under WSL, verify `nvidia-smi` works in WSL2 and ensure the
  CUDA-enabled packages are installed. Set `AUTOCAPTURE_GPU_MODE=off` to force CPU.
- For a full Windows GPU + security configuration, run
  `poetry run autocapture setup --profile full --apply` and re-run the doctor.

## PyCharm does not show new Git files

When PyCharm is open while you add files on disk (for example by pulling from
Git or copying directories manually), it can miss the updates if the project
roots or VCS mapping are not configured correctly. Use the following checklist
from the workstation to ensure PyCharm can see all tracked files:

1. **Confirm the Git repository is recognised.**
   - In PyCharm, open *File → Settings → Version Control*.
   - Ensure the project root (e.g. `<repo-root>`)
     appears under *Directory* with `Git` listed in the *VCS* column.
   - If it is missing, click the `+` icon, select the repository directory, and
     choose `Git` as the VCS type.

2. **Refresh the file system state.**
   - Press `Ctrl+Alt+Y` (or use *File → Synchronize*) to force a rescan of the
     project tree. This prompts PyCharm to pick up files created outside the IDE.

3. **Rescan Git roots.**
   - Open *VCS → Git → Remap VCS Root* and make sure the local repository path is
     selected. Click *OK* to re-index the Git metadata.

4. **Check `.gitignore` rules.**
   - Confirm the missing files are not matched by `.gitignore`. In the built-in
     terminal run `git check-ignore -v PATH/TO/FILE` to see whether Git is
     ignoring the path.

5. **Verify Git status from the terminal.**
   - In the project terminal run `git status`. If the files show up there but
     not in PyCharm, use *File → Invalidate Caches / Restart…* and choose
     *Invalidate and Restart*.

6. **Ensure the Git executable is available.**
   - Still in *Settings → Version Control → Git*, click *Test* next to the
     *Path to Git executable*. Point it at the correct `git.exe` if the test
     fails.

Following these steps should make the new modules (for example
`autocapture/ocr/pipeline.py`) visible to PyCharm so you can commit and sync
changes normally.

## Two Python processes appear when running Autocapture

If you notice both your virtual environment Python and the system-wide
`python.exe` running `-m autocapture.main`, it usually means a service or
scheduled task is launching the orchestrator (or OCR worker) with the generic
`python` command from `PATH`. Ensure every automation entry points to the
virtual environment interpreter directly:

```powershell
$venv = "<repo-root>\\.venv\\Scripts\\python.exe"
nssm install Autocapture $venv "-m" "autocapture.main" --config "<repo-root>/autocapture.yml"
```

You can find the correct path programmatically while inside the virtual
environment:

```powershell
python - <<'PY'
import sys
print(sys.executable)
PY
```

Update Task Scheduler definitions, PowerShell scripts, or helper batch files to
use that path so all child processes inherit the venv’s packages and settings.

Starting with this release the orchestrator also refuses to start under an
unexpected interpreter and prints a message similar to::

    Autocapture refused to start under unexpected interpreter.
    Expected: <repo-root>\.venv\Scripts\python.exe
    Current:  %LOCALAPPDATA%\Programs\Python\Python311\python.exe

If you see that output, adjust the service configuration and restart it so the
preferred virtual environment interpreter is used.

### Capture the source of unexpected spawns

When the system interpreter still appears after applying the checks above, turn
on spawn debugging to capture a stack trace for every new subprocess or
``multiprocessing`` worker. Set the following environment variables for the
service (or in your shell) before launching Autocapture:

```powershell
$env:AUTOCAPTURE_DEBUG_SPAWN = "1"
$env:AUTOCAPTURE_DEBUG_SPAWN_LOG = "%LOCALAPPDATA%/Autocapture/logs/spawn-debug.log"  # optional
python -m autocapture.main --config autocapture.yml
```

Each process writes its PID, interpreter path, and a traceback to the specified
log file whenever something calls :func:`subprocess.Popen`, :func:`subprocess.run`,
or the Windows ``multiprocessing`` spawn helpers. Inspect the most recent entries in
that log to identify which code path is requesting a new Python interpreter and
adjust the configuration accordingly.

## Overlay tracker

- The overlay UI requires the tray app (`autocapture app`) so the Qt event loop is running.
- If hotkeys do not fire, check for conflicts and update `overlay_tracker.hotkeys` in config.
- If the overlay hides during fullscreen apps, set `overlay_tracker.ui.auto_hide_fullscreen=false`.
