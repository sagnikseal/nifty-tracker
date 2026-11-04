#!/usr/bin/env sh
# macOS/Linux launcher (no Docker needed). First run installs deps, downloads data and trains.
cd "$(dirname "$0")"
[ -d .venv ] || python3 -m venv .venv
. .venv/bin/activate
pip install -q -r requirements-engine.txt
if command -v ollama >/dev/null 2>&1; then
  curl -s http://localhost:11434/api/tags >/dev/null 2>&1 || (ollama serve >/dev/null 2>&1 &)
else
  echo "Ollama not found: news uses the keyword lexicon. For the local LLM: brew install ollama && ollama pull llama3.1:8b"
fi
[ -f store/models/champion.json ] || python -m nifty_engine all
exec python -m nifty_engine run
