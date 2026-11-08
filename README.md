# NIFTY engine

A local, free, paper-trading engine for NIFTY 50 options. It has:

- a live dashboard;
- a 5-minute movement model;
- LLM news reading with learned market reactions;
- your ATM option rules;
- a local-LLM trading agent;
- tests that report losses as readily as wins.

**Start:** double-click `start.command` (Mac) or `start.bat` (Windows). The dashboard opens at http://localhost:8010.

**Guide:** [`docs/GUIDE.md`](docs/GUIDE.md), covering setup, rules, model and LLM options, what the tests found, and news sources.

**Paper only:** nothing in this repository can place a real order.

## Layout

| Path | What |
|---|---|
| `start.command`, `start.bat`, `start.sh` | one-click launchers (install, first-time training, start) |
| `nifty_engine/live/` | runtime: prices every 5 s, bars and predictions, agent, news, scheduled jobs |
| `docs/MODELS.md`, `docs/DATA_COLLECTION.md` | model zoo; GitHub Actions data collector |
| `nifty_engine/web/` | dashboard (no build step, works offline) and the glossary behind its tooltips |
| `nifty_engine/intraday_model.py` | movement model, walk-forward evaluation |
| `nifty_engine/strategies/atm_rules.py` | your ATM rules (shared by the replay and live trading) |
| `nifty_engine/paper/atm_replay.py` | replay on real 1-minute expired-option prices |
| `nifty_engine/semantic/` | local-LLM news scoring, news-impact model, pre-open brief |
| `nifty_engine/agent/` | expert prompts, trading agent with guardrails, LLM benchmark |
| `nifty_engine/data/news_sources.py` | fast news sources, health tracking, `sources-check` |
| `nifty_engine/model.py`, `features.py`, `zoo.py` | daily (09:20) model and model zoo |
| `nifty_engine/research/` | pre-registered experiments, deflated Sharpe, PBO |
| `config/engine.json` | all settings (rules under `atm_*`) |
| `docs/API.md` | dashboard ↔ engine API |
| `docs/RESEARCH.md` | research harness: pre-registered experiments and strategy books |

## Tests

```bash
python -m unittest discover -s tests
```

There are 99 tests. They cover look-ahead protection for every feature, the exact behaviour of the ATM rules, conservative replay fills, news parsing and polling, the news-impact study, agent guardrails, the model zoo, the GitHub collector, and the runtime and API end to end.
