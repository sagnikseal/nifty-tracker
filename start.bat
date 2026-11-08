@echo off
REM NIFTY engine: one-click launcher for Windows (double-click). Paper trading only.
REM First run: installs Python packages, downloads free market data, trains the models (10-20 min).
REM Every run: offers a few seconds to customise settings (see docs/FEATURES.md section 1.1).
setlocal enabledelayedexpansion
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

REM --- Settings: press 'c' within a few seconds to customise; Enter/timeout keeps your last choices
REM (or the defaults, on a first run). Full list of what each choice affects: docs/FEATURES.md.
set "CHOICES_FILE=.venv\.engine_choices.bat"
set "LLM_CHOICE=" & set "LLM_MODEL=" & set "GROUPS_CHOICE=base" & set "MODELRULES_AUTO=1" & set "AGENT_AUTO=1" & set "PORT=8010"
if exist "%CHOICES_FILE%" call "%CHOICES_FILE%"

set "CUSTOMIZED=0"
echo.
choice /c CN /t 6 /d N /n /m "Press C within 6 seconds to choose settings (LLM model, training features, auto-trade, port), or N/timeout to keep your current settings: "
if errorlevel 2 goto after_menu
set "CUSTOMIZED=1"

echo.
echo --- Local LLM model (news reading + the trading agent) ---
echo   1^) qwen3.5:9b   (default; ~6.6 GB, fits a 16 GB Mac)
echo   2^) llama3.1:8b  (older, reliable structured output)
echo   3^) gemma4:12b   (stronger reasoning, slower)
echo   4^) qwen3:8b
echo   5^) type a different Ollama model name
echo   6^) none -- disable the LLM agent and news reading (keyword lexicon only)
set "DFLT_LLM=%LLM_CHOICE%"
if "%DFLT_LLM%"=="" set "DFLT_LLM=1"
set /p "LLMPICK=Choice [current: %DFLT_LLM%]: "
if "%LLMPICK%"=="" set "LLMPICK=%DFLT_LLM%"
if "%LLMPICK%"=="1" (set "LLM_CHOICE=1" & set "LLM_MODEL=qwen3.5:9b")
if "%LLMPICK%"=="2" (set "LLM_CHOICE=2" & set "LLM_MODEL=llama3.1:8b")
if "%LLMPICK%"=="3" (set "LLM_CHOICE=3" & set "LLM_MODEL=gemma4:12b")
if "%LLMPICK%"=="4" (set "LLM_CHOICE=4" & set "LLM_MODEL=qwen3:8b")
if "%LLMPICK%"=="6" (set "LLM_CHOICE=6" & set "LLM_MODEL=")
if "%LLMPICK%"=="5" (
  set "LLM_CHOICE=5"
  set /p "LLM_MODEL=Ollama model name (e.g. mistral:7b): "
)

echo.
echo --- Intraday model training features (docs/FEATURES.md section 4) ---
echo   1^) base   (37 features: price/VIX only -- trains in a few minutes, default)
echo   2^) all    (189 features: slower, most tested with no benefit so far)
echo   3^) custom (type a comma list -- see docs/FEATURES.md section 4 for names)
set /p "GPICK=Choice [current: %GROUPS_CHOICE%]: "
if "%GPICK%"=="" set "GPICK=%GROUPS_CHOICE%"
if "%GPICK%"=="1" set "GROUPS_CHOICE=base"
if /i "%GPICK%"=="base" set "GROUPS_CHOICE=base"
if "%GPICK%"=="2" set "GROUPS_CHOICE=all"
if /i "%GPICK%"=="all" set "GROUPS_CHOICE=all"
if "%GPICK%"=="3" (
  set /p "GROUPS_CHOICE=Feature groups (comma-separated): "
) else (
  if not "%GPICK%"=="1" if not "%GPICK%"=="2" if /i not "%GPICK%"=="base" if /i not "%GPICK%"=="all" set "GROUPS_CHOICE=%GPICK%"
)

echo.
set "DFLT_MR=Y"
if "%MODELRULES_AUTO%"=="0" set "DFLT_MR=N"
set /p "MR=Auto-trade the 'Model + your rules' book (paper)? [Y/n, current: %DFLT_MR%]: "
if not "%MR%"=="" (
  if /i "%MR:~0,1%"=="n" (set "MODELRULES_AUTO=0") else (set "MODELRULES_AUTO=1")
)

if not "%LLM_MODEL%"=="" (
  set "DFLT_AR=Y"
  if "%AGENT_AUTO%"=="0" set "DFLT_AR=N"
  set /p "AR=Auto-trade the 'LLM agent + your rules' book (paper)? [Y/n, current: !DFLT_AR!]: "
  if not "!AR!"=="" (
    if /i "!AR:~0,1!"=="n" (set "AGENT_AUTO=0") else (set "AGENT_AUTO=1")
  )
) else (
  set "AGENT_AUTO=0"
)

set /p "PPORT=Dashboard port [current: %PORT%]: "
if not "%PPORT%"=="" set "PORT=%PPORT%"

> "%CHOICES_FILE%" (
  echo set "LLM_CHOICE=%LLM_CHOICE%"
  echo set "LLM_MODEL=%LLM_MODEL%"
  echo set "GROUPS_CHOICE=%GROUPS_CHOICE%"
  echo set "MODELRULES_AUTO=%MODELRULES_AUTO%"
  echo set "AGENT_AUTO=%AGENT_AUTO%"
  echo set "PORT=%PORT%"
)

:after_menu
if "%CUSTOMIZED%"=="1" (
  if "%MODELRULES_AUTO%"=="0" (set "NE_MODEL_RULES_AUTO=false") else (set "NE_MODEL_RULES_AUTO=true")
  if not "%LLM_MODEL%"=="" (
    set "NE_LLM_MODEL=%LLM_MODEL%"
    set "NE_AGENT_ENABLED=true"
    if "%AGENT_AUTO%"=="0" (set "NE_AGENT_AUTO=false") else (set "NE_AGENT_AUTO=true")
  ) else (
    set "NE_AGENT_ENABLED=false"
  )
)

if /i "%NE_AGENT_ENABLED%"=="false" (
  echo LLM agent disabled by your choice: news is scored with the keyword lexicon, no agent book.
) else (
  where ollama >nul 2>nul
  if errorlevel 1 (
    echo Ollama is not installed: news uses a keyword lexicon and the LLM agent is off. Install from https://ollama.com
  ) else (
    curl -s http://localhost:11434/api/tags >nul 2>nul || ( start "" /b ollama serve & timeout /t 3 >nul )
    for /f %%m in ('python -c "from nifty_engine.config import EngineConfig as E; print(E.load().llm_model)"') do set "MODEL=%%m"
    ollama list | findstr /b /c:"%MODEL% " >nul || ( echo Downloading the local LLM %MODEL% ^(one time^) ... & ollama pull %MODEL% )
  )
)

if not exist store\models\champion.json (
  echo First run: downloading daily market data and training the daily model ...
  python -m nifty_engine update-data
  python -m nifty_engine train
)
if not exist store\intraday\nifty_5m.csv ( python -m nifty_engine refresh-intraday )

set "NEED_TRAIN=0"
if not exist store\models\intraday\champion.json set "NEED_TRAIN=1"
if not "%GROUPS_CHOICE%"=="base" set "NEED_TRAIN=1"
if "%NEED_TRAIN%"=="1" (
  if "%GROUPS_CHOICE%"=="base" (
    echo Training the intraday movement model ^(a few minutes^) ...
    python -m nifty_engine intraday-train
  ) else (
    echo Training the intraday movement model ^(feature groups: %GROUPS_CHOICE%^) ...
    python -m nifty_engine intraday-train --groups %GROUPS_CHOICE%
  )
)

echo Starting the engine. The dashboard opens at http://localhost:%PORT%  ^(close this window to stop^).
python -m nifty_engine app --port %PORT%
pause
