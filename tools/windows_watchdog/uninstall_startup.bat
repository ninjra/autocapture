@echo off
setlocal
set "STARTUP=%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup"
set "LINK=%STARTUP%\AutocaptureWatchdog.lnk"

if exist "%LINK%" (
  del "%LINK%"
  echo Removed startup shortcut: %LINK%
) else (
  echo Startup shortcut not found: %LINK%
)
