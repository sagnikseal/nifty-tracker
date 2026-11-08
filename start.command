#!/usr/bin/env bash
# NIFTY engine: one-click launcher for macOS (double-click in Finder) and Linux (./start.sh).
# Paper trading only: nothing here can place a real order.
# First run: installs Python packages, downloads free market data, trains the models (10-20 min).
# Every run: offers a few seconds to customise settings (§1.1 in docs/FEATURES.md), starts the
# local LLM (Ollama) if enabled, then the engine + dashboard in your browser.
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

# --- Settings menu: press 'c' within a few seconds to customise; Enter/timeout keeps your last
# choices (or the defaults, on a first run). Full list of what each choice affects: docs/FEATURES.md.
CHOICES_FILE=.venv/.engine_choices
LLM_CHOICE="" ; LLM_MODEL="" ; GROUPS_CHOICE="base" ; MODELRULES_AUTO="1" ; AGENT_AUTO="1" ; PORT="8010"
[ -f "$CHOICES_FILE" ] && . "$CHOICES_FILE"

CUSTOMIZED=0
echo ""
read -r -t 6 -p "Press 'c' + Enter within 6s to choose settings (LLM model, training features, auto-trade, port) -- Enter or a timeout keeps your current settings: " ANSWER || true
echo ""
if [ "$ANSWER" = "c" ] || [ "$ANSWER" = "C" ]; then
  CUSTOMIZED=1
  echo "--- Local LLM model (news reading + the trading agent) ---"
  echo "  1) qwen3.5:9b   (default; ~6.6 GB, fits a 16 GB Mac)"
  echo "  2) llama3.1:8b  (older, reliable structured output)"
  echo "  3) gemma4:12b   (stronger reasoning, slower)"
  echo "  4) qwen3:8b"
  echo "  5) type a different Ollama model name"
  echo "  6) none -- disable the LLM agent and news reading (keyword lexicon only)"
  read -r -p "Choice [current: ${LLM_CHOICE:-1}]: " c
  c=${c:-${LLM_CHOICE:-1}}
  case "$c" in
    1) LLM_CHOICE=1; LLM_MODEL="qwen3.5:9b" ;;
    2) LLM_CHOICE=2; LLM_MODEL="llama3.1:8b" ;;
    3) LLM_CHOICE=3; LLM_MODEL="gemma4:12b" ;;
    4) LLM_CHOICE=4; LLM_MODEL="qwen3:8b" ;;
    5) read -r -p "Ollama model name (e.g. mistral:7b): " LLM_MODEL; LLM_CHOICE=5 ;;
    6) LLM_CHOICE=6; LLM_MODEL="" ;;
    *) LLM_CHOICE=1; LLM_MODEL="qwen3.5:9b" ;;
  esac

  echo ""
  echo "--- Intraday model training features (docs/FEATURES.md section 4) ---"
  echo "  1) base      (37 features: price/VIX only -- trains in a few minutes, default)"
  echo "  2) all       (189 features: + technicals, Bank Nifty, futures, option chain, global, macro --"
  echo "               slower, and most of these groups tested with no benefit so far)"
  echo "  3) custom    (type a comma list, e.g. tech,chain -- see docs/FEATURES.md section 4 for names)"
  read -r -p "Choice [current: ${GROUPS_CHOICE}]: " g
  g=${g:-$GROUPS_CHOICE}
  case "$g" in
    1|base) GROUPS_CHOICE="base" ;;
    2|all) GROUPS_CHOICE="all" ;;
    3) read -r -p "Feature groups (comma-separated): " GROUPS_CHOICE ;;
    *) GROUPS_CHOICE="$g" ;;
  esac

  echo ""
  dflt_mr=$([ "$MODELRULES_AUTO" = "0" ] && echo N || echo Y)
  read -r -p "Auto-trade the 'Model + your rules' book (paper)? [Y/n, current: ${dflt_mr}]: " mr
  [ -n "$mr" ] && case "$mr" in [nN]*) MODELRULES_AUTO=0 ;; *) MODELRULES_AUTO=1 ;; esac

  if [ -n "$LLM_MODEL" ]; then
    dflt_ar=$([ "$AGENT_AUTO" = "0" ] && echo N || echo Y)
    read -r -p "Auto-trade the 'LLM agent + your rules' book (paper)? [Y/n, current: ${dflt_ar}]: " ar
    [ -n "$ar" ] && case "$ar" in [nN]*) AGENT_AUTO=0 ;; *) AGENT_AUTO=1 ;; esac
  else
    AGENT_AUTO=0
  fi

  read -r -p "Dashboard port [current: ${PORT}]: " p
  PORT=${p:-$PORT}

  {
    echo "LLM_CHOICE=$LLM_CHOICE"
    echo "LLM_MODEL=\"$LLM_MODEL\""
    echo "GROUPS_CHOICE=\"$GROUPS_CHOICE\""
    echo "MODELRULES_AUTO=$MODELRULES_AUTO"
    echo "AGENT_AUTO=$AGENT_AUTO"
    echo "PORT=$PORT"
  } > "$CHOICES_FILE"
fi

if [ "$CUSTOMIZED" = "1" ]; then
  export NE_MODEL_RULES_AUTO=$([ "$MODELRULES_AUTO" = "1" ] && echo true || echo false)
  if [ -n "$LLM_MODEL" ]; then
    export NE_LLM_MODEL="$LLM_MODEL"
    export NE_AGENT_ENABLED=true
    export NE_AGENT_AUTO=$([ "$AGENT_AUTO" = "1" ] && echo true || echo false)
  else
    export NE_AGENT_ENABLED=false
  fi
fi

if [ "${NE_AGENT_ENABLED:-true}" = "false" ]; then
  echo "LLM agent disabled by your choice: news is scored with the keyword lexicon, no agent book."
elif command -v ollama >/dev/null 2>&1; then
  MODEL=$(python -c "from nifty_engine.config import EngineConfig as E; print(E.load().llm_model)")
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
NEED_TRAIN=0
[ -f store/models/intraday/champion.json ] || NEED_TRAIN=1
[ "$GROUPS_CHOICE" != "base" ] && NEED_TRAIN=1
if [ "$NEED_TRAIN" = "1" ]; then
  if [ "$GROUPS_CHOICE" = "base" ] || [ -z "$GROUPS_CHOICE" ]; then
    GROUPS_FLAG=()
    NOTE=""
  else
    GROUPS_FLAG=(--groups "$GROUPS_CHOICE")
    NOTE=" (feature groups: $GROUPS_CHOICE)"
  fi
  echo "Training the intraday movement model${NOTE} ..."
  python -m nifty_engine intraday-train "${GROUPS_FLAG[@]}" || echo "Intraday model needs more history (Upstox token + fetch-intraday)."
fi

echo "Starting the engine. The dashboard opens at http://localhost:$PORT  (close this window to stop)."
exec python -m nifty_engine app --port "$PORT"
