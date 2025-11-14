@echo off
REM --- Open Command Prompt in the current folder (no admin) ---
REM Place this .bat in any folder and double-click it.
cd /d "%~dp0"
start "" cmd /k "cd /d "%~dp0""
