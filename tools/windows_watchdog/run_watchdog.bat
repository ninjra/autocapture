@echo off
setlocal
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%" || exit /b 1

if not exist ".venv\Scripts\python.exe" (
  echo Creating watchdog virtual environment...
  python -m venv .venv
  if errorlevel 1 (
    echo Failed to create venv. Ensure Python is on PATH.
    exit /b 1
  )
)

call ".venv\Scripts\activate.bat"
if errorlevel 1 (
  echo Failed to activate venv.
  exit /b 1
)

python -m pip install -r requirements.txt
if errorlevel 1 (
  echo Failed to install requirements.
  exit /b 1
)

python watchdog.py
