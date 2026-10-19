@echo off
rem Diagnose why AI answers / translation / OCR are not working. Installs missing packages first.
cd /d "%~dp0"
set PYTHONPATH=%~dp0src
echo Step 1/2: installing missing packages (first time takes 1-3 minutes, please wait)...
.venv\Scripts\pip install -r requirements.txt
echo.
echo Step 2/2: checking AI engines...
.venv\Scripts\python scripts\check_ai.py
pause
