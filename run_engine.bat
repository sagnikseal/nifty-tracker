@echo off
REM Windows one-click launcher (no Docker needed). First run installs deps, downloads data, trains.
cd /d %~dp0
if not exist .venv ( python -m venv .venv )
call .venv\Scripts\activate
pip install -q -r requirements-engine.txt
if not exist store\models\champion.json ( python -m nifty_engine all )
python -m nifty_engine run
