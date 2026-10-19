@echo off
rem IndicLearn Studio launcher (Windows). Creates a venv on first run and keeps its packages up to date.
cd /d "%~dp0"
if not exist .venv ( py -3 -m venv .venv && .venv\Scripts\python -m pip install -U pip )
.venv\Scripts\pip install -q -r requirements.txt
set PYTHONPATH=%~dp0src
start "" http://127.0.0.1:8000
.venv\Scripts\python -m indiclearn.studio
