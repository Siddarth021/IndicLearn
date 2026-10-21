@echo off
rem One-time FULLY OFFLINE setup for IndicLearn (RTX 3050 6 GB / 16 GB RAM laptop).
rem   1. Ollama  : install from https://ollama.com/download first (free), then this script pulls gemma4:e4b
rem   2. GPU     : switches PyTorch to the CUDA build so the RTX card is used
rem   3. Telugu  : our fine-tuned translator + M2/M4 Telugu models (python training\kaggle_run.py fetch --kind indic)
rem   4. .env    : INDICLEARN_OFFLINE=true, INDICLEARN_LLM_MODEL=gemma4:e4b
cd /d "%~dp0"
if not exist .venv ( py -3 -m venv .venv && .venv\Scripts\python -m pip install -U pip && .venv\Scripts\pip install -r requirements.txt )
.venv\Scripts\pip install -r requirements-models.txt huggingface_hub kaggle
.venv\Scripts\python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || .venv\Scripts\pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu121
if not exist models\trained\te_style_translator\config.json ( set KAGGLE_CONFIG_DIR=%~dp0secrets & .venv\Scripts\python training\kaggle_run.py fetch --kind indic )
.venv\Scripts\python scripts\setup_offline.py %*
.venv\Scripts\python scripts\set_env.py INDICLEARN_OFFLINE=true INDICLEARN_LLM_MODEL=gemma4:e4b
pause
