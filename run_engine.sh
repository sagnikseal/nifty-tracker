#!/usr/bin/env sh
# macOS/Linux launcher (no Docker needed).
cd "$(dirname "$0")"
[ -d .venv ] || python3 -m venv .venv
. .venv/bin/activate
pip install -q -r requirements-engine.txt
[ -f store/models/champion.json ] || python -m nifty_engine all
exec python -m nifty_engine run
