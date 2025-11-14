@echo off
REM --- Open elevated Command Prompt (requests UAC) in the current folder ---
REM Place this .bat in any folder and double-click it.
:: Try 'net session' which requires admin; if it succeeds we're already elevated.
net session >nul 2>&1
if %errorlevel%==0 (
  cd /d "%~dp0"
  start "" cmd /k "cd /d "%~dp0""
  exit /b
)
:: Not elevated — re-launch an elevated cmd that starts in this folder
powershell -Command "Start-Process cmd -ArgumentList '/k cd /d \"%~dp0\"' -Verb RunAs"
