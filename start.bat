@echo off
REM NIFTY engine: one-click launcher for Windows (double-click). Paper trading only.
REM First run: installs Python packages, downloads free market data, trains the models (10-20 min).
setlocal
cd /d %~dp0
echo === NIFTY engine ===

where py >nul 2>nul && (set "PY=py -3") || (set "PY=python")
%PY% -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul
if errorlevel 1 (
  echo Python 3.10+ is needed: install it from https://www.python.org/downloads/ ^(tick "Add python.exe to PATH"^).
  pause & exit /b 1
)
if not exist .venv ( %PY% -m venv .venv )
call .venv\Scripts\activate.bat
if not exist .venv\.installed (
  echo Installing Python packages ^(first run^) ...
  python -m pip install -q --upgrade pip
  python -m pip install -q -r requirements.txt && echo ok> .venv\.installed
)
if not exist .env ( copy .env.example .env >nul & echo Created .env: add your UPSTOX_ANALYTICS_TOKEN there for live option prices ^(optional^). )

where ollama >nul 2>nul
if errorlevel 1 (
  echo Ollama is not installed: news uses a keyword lexicon and the LLM agent is off. Install from https://ollama.com
) else (
  curl -s http://localhost:11434/api/tags >nul 2>nul || ( start "" /b ollama serve & timeout /t 3 >nul )
  for /f %%m in ('python -c "from nifty_engine.config import EngineConfig as E; print(E.load().llm_model)"') do set "MODEL=%%m"
  ollama list | findstr /b /c:"%MODEL% " >nul || ( echo Downloading the local LLM %MODEL% ^(one time^) ... & ollama pull %MODEL% )
)

if not exist store\models\champion.json (
  echo First run: downloading daily market data and training the daily model ...
  python -m nifty_engine update-data
  python -m nifty_engine train
)
if not exist store\intraday\nifty_5m.csv ( python -m nifty_engine refresh-intraday )
if not exist store\models\intraday\champion.json (
  echo Training the intraday movement model ^(about 2 minutes^) ...
  python -m nifty_engine intraday-train
)
echo Starting the engine. The dashboard opens at http://localhost:8010  ^(close this window to stop^).
python -m nifty_engine app
pause
