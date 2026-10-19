@echo off
rem One-time offline setup: local LLM (Ollama + Gemma 4), speech, translation and search models.
rem Install Ollama first from https://ollama.com/download and start it.
cd /d "%~dp0"
if not exist .venv ( py -3 -m venv .venv && .venv\Scripts\python -m pip install -U pip && .venv\Scripts\pip install -r requirements.txt )
.venv\Scripts\pip install -r requirements-models.txt huggingface_hub
.venv\Scripts\python scripts\setup_offline.py %*
pause
