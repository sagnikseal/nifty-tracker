#!/usr/bin/env bash
# NIFTY engine: one-click launcher for macOS (double-click in Finder) and Linux (./start.sh).
# Paper trading only: nothing here can place a real order.
# First run: installs Python packages, downloads free market data, trains the models (10-20 min).
# Every run: starts the local LLM (Ollama) if installed, then the engine + dashboard in your browser.
set -e
cd "$(dirname "$0")"
echo "=== NIFTY engine ==="

PY=""
for c in python3.13 python3.12 python3.11 python3.10 python3; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then PY="$c"; break; fi
done
if [ -z "$PY" ]; then
  echo "Python 3.10+ is needed. macOS: install from https://www.python.org/downloads/ (or: brew install python)."
  read -r -p "Press Enter to close." _; exit 1
fi

[ -d .venv ] || "$PY" -m venv .venv
. .venv/bin/activate
STAMP=.venv/.requirements.sha
SUM=$(cat requirements.txt | shasum | cut -d' ' -f1)
if [ ! -f "$STAMP" ] || [ "$(cat "$STAMP")" != "$SUM" ]; then
  echo "Installing Python packages (first run or requirements changed) ..."
  python -m pip install -q --upgrade pip
  python -m pip install -q -r requirements.txt
  echo "$SUM" > "$STAMP"
fi
[ -f .env ] || { cp .env.example .env; echo "Created .env: add your UPSTOX_ANALYTICS_TOKEN there for live option prices (optional)."; }

MODEL=$(python -c "from nifty_engine.config import EngineConfig as E; print(E.load().llm_model)")
if command -v ollama >/dev/null 2>&1; then
  if ! curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo "Starting Ollama ..."; (ollama serve >/dev/null 2>&1 &); sleep 3
  fi
  if ! ollama list 2>/dev/null | awk '{print $1}' | grep -qx "$MODEL"; then
    echo "Downloading the local LLM $MODEL (one time, several GB) ..."
    ollama pull "$MODEL" || echo "Could not pull $MODEL; the engine will use the keyword lexicon for news."
  fi
else
  echo "Ollama is not installed: news is scored with a keyword lexicon and the LLM agent is off."
  echo "For the local LLM: install from https://ollama.com (or: brew install ollama), then run this again."
fi

if [ ! -f store/models/champion.json ]; then
  echo "First run: downloading daily market data and training the daily model ..."
  python -m nifty_engine update-data
  python -m nifty_engine train || echo "Daily model training failed; see the messages above."
fi
if [ ! -f store/intraday/nifty_5m.csv ]; then
  if grep -q '^UPSTOX_ANALYTICS_TOKEN=..' .env 2>/dev/null; then
    echo "Downloading intraday history from Upstox (since 2022) ..."; python -m nifty_engine fetch-intraday || true
  else
    python -m nifty_engine refresh-intraday || true
  fi
fi
if [ ! -f store/models/intraday/champion.json ]; then
  echo "Training the intraday movement model (about 2 minutes) ..."
  python -m nifty_engine intraday-train || echo "Intraday model needs more history (Upstox token + fetch-intraday)."
fi

echo "Starting the engine. The dashboard opens at http://localhost:8010  (close this window to stop)."
exec python -m nifty_engine app
