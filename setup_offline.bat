@echo off
rem One-time FULLY OFFLINE setup for IndicLearn (RTX 3050 6 GB / 16 GB RAM laptop).
rem   1. Ollama  : install from https://ollama.com/download first (free); this script pulls gemma4:e2b (e4b does not fit a 6 GB GPU)
rem   2. GPU     : switches PyTorch to the CUDA build so the RTX card is used
rem   3. Telugu  : our fine-tuned translator + Telugu M2/M4 models (from your Kaggle run)
rem   4. .env    : INDICLEARN_OFFLINE=true, INDICLEARN_LLM_MODEL=gemma4:e2b
cd /d "%~dp0"

rem C: is nearly full on this laptop, so every big temporary file goes to the project drive instead.
if not exist "%~dp0.tmp" mkdir "%~dp0.tmp"
set "TEMP=%~dp0.tmp"
set "TMP=%~dp0.tmp"
set "PIP_CACHE_DIR=%~dp0.tmp\pip-cache"
set "HF_HUB_CACHE=%~dp0.tmp\hf"
set "KAGGLE_CONFIG_DIR=%~dp0secrets"

if not exist .venv ( py -3 -m venv .venv && .venv\Scripts\python -m pip install -U pip && .venv\Scripts\pip install -r requirements.txt )
.venv\Scripts\pip install -r requirements-models.txt huggingface_hub "kaggle<2"

rem GPU build of PyTorch (2.4 GB). Retries, because the download can drop on a slow link.
for /l %%i in (1,1,6) do (
  .venv\Scripts\python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || .venv\Scripts\pip install --default-timeout=300 --retries 30 --upgrade "torch>=2.6" --index-url https://download.pytorch.org/whl/cu124
)

rem Telugu models (the style translator is 2.2 GB)
if not exist models\trained\te_style_translator\config.json (
  .venv\Scripts\python training\kaggle_run.py fetch --kind indic
  if not exist models\trained\te_style_translator\config.json echo. & echo Telugu translator NOT installed yet: download the notebook output in the browser, then run scripts\install_telugu_models.py ^<zip^>
)

.venv\Scripts\python scripts\setup_offline.py --small %*
.venv\Scripts\python scripts\set_env.py INDICLEARN_OFFLINE=true INDICLEARN_LLM_MODEL=gemma4:e2b
echo.
echo If anything above says "No space left on device", free space on C: (Settings - Storage - Temporary files) and run this again.
pause
