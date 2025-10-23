# Troubleshooting

## PyCharm does not show new Git files

When PyCharm is open while you add files on disk (for example by pulling from
Git or copying directories manually), it can miss the updates if the project
roots or VCS mapping are not configured correctly. Use the following checklist
from the workstation to ensure PyCharm can see all tracked files:

1. **Confirm the Git repository is recognised.**
   - In PyCharm, open *File → Settings → Version Control*.
   - Ensure the project root (e.g. `D:\pycharm\autoscreenshjot+ocr\autocapture`)
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
$venv = "C:\\Path\\To\\repo\\.venv\\Scripts\\python.exe"
nssm install Autocapture $venv "-m" "autocapture.main" --config C:/Path/To/autocapture.yml
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
    Expected: D:\path\to\repo\.venv\Scripts\python.exe
    Current:  C:\Users\you\AppData\Local\Programs\Python\Python311\python.exe

If you see that output, adjust the service configuration and restart it so the
preferred virtual environment interpreter is used.
