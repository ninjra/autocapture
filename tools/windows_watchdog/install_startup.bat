@echo off
setlocal
set "SCRIPT_DIR=%~dp0"
set "STARTUP=%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup"
set "LINK=%STARTUP%\AutocaptureWatchdog.lnk"
set "TARGET=%SCRIPT_DIR%run_watchdog.bat"

powershell -NoProfile -Command "$WshShell = New-Object -ComObject WScript.Shell; $Shortcut = $WshShell.CreateShortcut('%LINK%'); $Shortcut.TargetPath = '%TARGET%'; $Shortcut.WorkingDirectory = '%SCRIPT_DIR%'; $Shortcut.Save()"
if errorlevel 1 (
  echo Failed to create startup shortcut.
  exit /b 1
)

echo Installed startup shortcut: %LINK%
