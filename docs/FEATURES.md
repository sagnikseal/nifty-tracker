# Every option in the code (reference)

This is the full catalogue of what exists: every CLI command, every setting, every model, every
feature group, every book. For *why* things work this way and what the results so far mean, read
[`GUIDE.md`](GUIDE.md). For modelling detail, [`MODELS.md`](MODELS.md) and [`RESEARCH.md`](RESEARCH.md).
For the dashboard ↔ engine API, [`API.md`](API.md). For the GitHub data collector, [`DATA_COLLECTION.md`](DATA_COLLECTION.md).

Any setting below can be overridden for one run as an environment variable `NE_<SETTING_UPPERCASE>`
(e.g. `NE_ATM_STOP_INR=3000`), or permanently in `config/engine.json`.

---

## 1. Launching

| Script | Platform | What it does |
|---|---|---|
| `start.command` | macOS (double-click), Linux | installs on first run, offers a settings menu (§1.1), trains the models if missing, starts the app |
| `start.bat` | Windows (double-click) | same, Windows syntax |
| `start.sh` | Linux | thin wrapper that calls `start.command` |

### 1.1 The settings menu

Each launch gives you a few seconds to type `c` and customise; otherwise it starts with your last
choices (or the defaults, on a first run). Your answers are remembered for next time.

| Prompt | Choices | Affects |
|---|---|---|
| Local LLM model | `qwen3.5:9b` (default) / `llama3.1:8b` / `gemma4:12b` / `qwen3:8b` / type your own / `none` (disables the agent) | `llm_model`; see §6 |
| Intraday training features | `base` (37 features, a few minutes, default) / `all` (189 features, slower) / custom comma list | `intraday-train --groups`; see §4 |
| Auto-trade "Model + your rules"? | Y/n | `model_rules_auto` |
| Auto-trade "LLM agent + your rules"? | Y/n (only asked if the LLM is enabled) | `agent_auto` |
| Dashboard port | number, default `8010` | `--port` |

If you pick `all` or a custom feature list, the intraday model retrains that run even if one already
exists (it has to, the feature set changed); picking `base` reuses an existing model, same as not
customising at all.

### 1.2 Starting it yourself

```bash
python -m nifty_engine app                       # everything + open the dashboard
python -m nifty_engine run    --port 8080         # everything, no browser
python -m nifty_engine serve  --no-browser         # dashboard + reports only, no live engine
```
Flags: `--port <n>` (default 8010), `--host <addr>` (keep `127.0.0.1`: this machine only).

---

## 2. Every CLI command

```
python -m nifty_engine <command> [flags]
```

**Everyday**

| Command | What it does | Flags |
|---|---|---|
| `app` | start everything + open the dashboard | `--port`, `--host`, `--no-browser` |
| `run` | same as `app`, no browser | same |
| `serve` | dashboard + reports only, no live engine | `--port`, `--host`, `--no-browser` |

**Models and tests** (all walk-forward, out-of-sample)

| Command | What it does | Flags |
|---|---|---|
| `intraday-train` | movement model: P(up in 30 min), P(up by 15:15), every 5 min → `INTRADAY_REPORT.md` | `--groups all\|<comma list>` (§4), `--members cat,xgb` (§7), `--quick` |
| `intraday-zoo` | full stacked zoo (run on the Mac) → `INTRADAY_ZOO_REPORT.md` | `--quick`, `--seq <archs>` or `--seq ''`, `--tsfm <models>`, `--members` |
| `intraday-ablation` | which feature groups help, pre-registered → `INTRADAY_ABLATION_REPORT.md` | `--quick`, `--members` |
| `atm-replay` | your ATM rules on real 1-minute option prices → `ATM_REPLAY_REPORT.md` | — |
| `news-impact` | how NIFTY moved after each LLM-read headline → `NEWS_IMPACT_REPORT.md` | — |
| `llm-bench` | compare local LLMs on labelled headlines + agent scenarios → `LLM_BENCH_REPORT.md` | `--models <comma list>` |
| `sources-check` | test every news source from this machine → `SOURCES_REPORT.md` | — |
| `data-report` | what data exists / is missing / how to get it → `DATA_REPORT.md` | — |
| `research` | every pre-registered daily-model/book experiment → `RESEARCH_REPORT.md` | `--only P1,S4`, `--quick` |
| `train` / `backtest` | daily (09:20 open→close) model: walk-forward train / backtest | `--capital`, `--threshold` |
| `zoo` | daily model zoo (§7) | `--fresh`, `--promote`, `--members`, `--no-deep` |
| `variants` | daily options book: stop/threshold grid → `VARIANTS.md` | — |
| `replay-real` | daily options book vs real expired-option prices | `--since` |

**Data**

| Command | What it does | Flags |
|---|---|---|
| `update-data` | daily NIFTY/VIX/global series (free, Yahoo) | — |
| `fetch-intraday` | 5-min NIFTY + VIX since 2022 (Upstox) | — |
| `fetch-history` | Bank Nifty, 10 heavyweights, continuous NIFTY futures (Upstox) | — |
| `refresh-intraday` | top up the above since the last run (Upstox, else Yahoo) | — |
| `fetch-expired-options` | real prices of expired weekly options (Upstox Plus) | `--since`, `--mode itm\|wings\|full`, `--max-expiries` |
| `build-option-features` | historical option-chain features from expired-option candles | — |
| `record` | record today's option chain + depth, all day (Upstox) | — |
| `chain-poll` | **one** option-chain snapshot, then exit (for a 15-min cron, not daily use) | — |
| `news-capture` | collect headlines now | — |
| `news-score` | score unscored headlines with the LLM | — |
| `news-backfill` | ~3 months of history from GDELT, then scores it | — |
| `brief` | write today's pre-open LLM brief | — |
| `collect-news` | legacy lexicon-only daily collector (superseded by `news-capture`) | — |
| `books` | status of the (off-by-default) research strategy books | — |
| `llm-check` | is the local LLM reachable? prints a test score | — |
| `signal` | print today's daily-model signal | — |

---

## 3. `config/engine.json` — every setting

**Paper account, options structure**

| Setting | Default | Meaning |
|---|---|---|
| `capital` | 100000 | paper capital (₹) for the daily options book |
| `books` | `"options"` | `"options"`, `"futures"` or both, comma-separated |
| `option_style` | `"itm_buy"` | or `"debit_spread"` |
| `spread_width_pts` | 150 | short-leg distance for debit spreads |
| `lot_size` | 65 | NIFTY lot size |
| `max_lots` | 1 | |

**Risk (daily options book)**

| Setting | Default | Meaning |
|---|---|---|
| `risk_per_trade_pct` | 10.0 | loss budget at the disaster stop, % of equity |
| `max_premium_pct` | 35.0 | max premium outlay per trade, % of equity |
| `option_stop_pct` | 60.0 | disaster-stop only (tight stops lost money in replay) |
| `option_target_pct` | 70.0 | take profit at this % rise in premium |
| `daily_loss_limit_pct` | 6.0 | defined but **not wired into any exit path** — don't rely on it |
| `max_drawdown_kill_pct` | 30.0 | stop paper trading after this account drawdown |
| `paper_trade_without_edge` | true | allow paper trading an unproven model (flagged in the UI) |
| `max_trades_per_day` | 1 | |

**Iron condor research book**

| Setting | Default | Meaning |
|---|---|---|
| `condor_k` | 1.25 | short strikes at k × VIX-implied daily σ from spot |
| `condor_wing` | 200 | long wings this many points beyond the shorts |
| `condor_stop_mult` | 1.5 | exit at this multiple of the credit received |
| `condor_p_threshold` | 0.80 | trade only when the range model says P(stays inside) ≥ this |
| `condor_max_loss_pct` | 15.0 | max defined loss per trade, % of equity |

**Daily-model signal**

| Setting | Default | Meaning |
|---|---|---|
| `target` | `"oc"` | `"oc"` open→close or `"cc"` close→next close |
| `prob_threshold` | 0.56 | trade only if p ≥ this (long) or ≤ 1 − this (short) |
| `deadzone_bps` | 5.0 | training labels: moves smaller than this are dropped as "no edge" |
| `retrain_every_days` | 21 | walk-forward refit cadence |
| `use_zoo` | true | weekly retrain also runs the daily zoo, may deploy best-3 |
| `min_train_days` | 750 | |
| `history_start` | `"2008-01-01"` | |
| `feature_groups` | `"base"` | daily-model groups (§5); promote one at a time after it passes research |

**Costs, timing, data source**

| Setting | Default | Meaning |
|---|---|---|
| `brokerage_per_order` | 20.0 | ₹ per order |
| `option_slippage_pts` | 1.0 | per leg per side |
| `futures_slippage_pts` | 1.0 | |
| `entry_time` / `exit_time` | 09:20 / 15:15 | daily options book |
| `broker` | `"auto"` | `"auto"`, `"upstox"` or `"yahoo"` — signals never depend on a broker |
| `poll_seconds` | 30 | legacy scheduler poll (superseded by the live runtime's own threads) |
| `record_live` | true | record option chain / depth during market hours (needs an Upstox token) |

**Research strategy books** (off by default — see §8.2)

| Setting | Default |
|---|---|
| `strategy_books` | `""` (empty) |
| `book_capital` | 100000 (₹ per research book) |

**Local LLM**

| Setting | Default | Meaning |
|---|---|---|
| `llm_model` | `"qwen3.5:9b"` | see §6 |
| `llm_url` | `http://localhost:11434` | |
| `llm_knowledge_cutoff` | `"2026-03-10"` | headlines on/before this use the keyword lexicon in every test |

**Your ATM rules** — see §9 for the full table.

**Runtime**

| Setting | Default | Meaning |
|---|---|---|
| `price_poll_seconds` | 5.0 | how often open positions are priced (stop/target checked each time) |
| `news_poll_seconds` | 60 | base interval for fast news sources |
| `agent_enabled` | true | the LLM trading agent exists at all |
| `agent_auto` | true | the agent's book trades automatically (paper) |
| `model_rules_auto` | true | the model+rules book trades automatically (paper) |

---

## 4. Intraday feature groups (`INTRADAY_GROUPS`)

The 37 "base" price/VIX features are always on; these are extra, added via `--groups` on
`intraday-train` or `intraday-zoo`. `--groups all` turns on every group below.

| Group | Column prefix | Adds | Needs |
|---|---|---|---|
| `tech` | `t5_` | EMA/MACD/RSI/ADX/Supertrend/Bollinger/stochastic/CCI/ROC/variance ratio on 5-min bars | — |
| `tech_mtf` | `m15_`, `m60_`, `mtf_` | the same, on 15- and 60-minute bars | — |
| `candles` | `cdl_`, `ha_` | candlestick patterns, Heikin-Ashi runs | — |
| `structure` | `sw_`, `don`, `round` | swing highs/lows, Donchian breakouts, round-number distance | — |
| `banknifty` | `bn_` | Bank Nifty leadership vs NIFTY | `banknifty_5m.csv` (`fetch-history`) |
| `breadth` | `br_` | heavyweight-stock breadth | heavyweight `*_5m.csv` (`fetch-history`) |
| `futures` | `fut_` | basis, VWAP, volume surge, OI regime | `niftyfut_5m.csv` (`fetch-history`, Upstox Plus) |
| `chain` | `oc_` | PCR, OI flow, ATM IV, skew, walls, implied move | `store/options_hist_1m` (`fetch-expired-options`) |
| `global_intraday` | `gl_` | US futures / USDINR / crude / gold moves during the day | hourly series (`update-data`) |
| `daily_ctx` | `dc_` | yesterday's global/volatility context | `update-data` |
| `events` | `evt_` | scheduled-event flags for the day | `config/events.json` |
| `news` | `nw_` | news tone/shock, learned impact (built automatically once headlines exist) | `news-capture` / `news-backfill` |
| `tsfm` | `tsfm_` | foundation-model forecasts as features (`intraday-zoo --tsfm`, §7.3) | the matching package installed |

Ablation result as of 2026-11-08: none of `tech`/`tech_mtf`/`candles`/`structure`/`banknifty`/`breadth`/`daily_ctx`/`events`/`futures`/`global_intraday` improved the model over the base 37 features (see `INTRADAY_ABLATION_REPORT.md`); `chain` initially looked like it did, but that was a data leak, fixed (see `GUIDE.md` §11). `news` and `tsfm` haven't been tested yet.

---

## 5. Daily-model feature groups (`FEATURE_GROUPS`)

Used by `train` / `backtest` / `research` via `config/engine.json`'s `feature_groups`.

| Group | Adds | Tested as |
|---|---|---|
| `base` | returns, volatility, global markets (strict timing), VIX regime, calendar | P0 (benchmark) |
| `asia_extra` | KOSPI / Taiwan / Shanghai opening gaps | P1 |
| `adr` | ADR and INDA premium vs the NSE close | P2 |
| `smc` | daily SMC structure (BOS/CHoCH, order blocks, FVGs, sweeps) | P3 |
| `structure` | CPR / PDH / PDL / weekly levels | P4 |
| `regime` | HAR volatility forecast, VRP, efficiency ratio | P5 |
| `events` | scheduled-event flags | P6 |
| `overnight_futures` | ES/NQ/crude/BTC/USDINR move, 15:30→09:00 IST | P7 |
| `options_hist` | previous-close option-chain features | P8 |
| `news` | pre-open news tone | P9 |

Only promote a group into the live `feature_groups` setting after it passes its experiment on a full (non-`--quick`) `research` run.

---

## 6. Local LLM models

Any Ollama model works; compare what's actually installed on your machine with `llm-bench`:

| Model | Size (4-bit) | Notes |
|---|---|---|
| `qwen3.5:9b` (default) | ~6.6 GB | current pick for a 16 GB Mac |
| `llama3.1:8b` | ~5 GB | older, reliable structured JSON |
| `gemma4:12b` | ~8 GB | stronger reasoning, slower |
| `qwen3:8b` | ~5 GB | widely-supported fallback |
| anything else you've pulled | — | type the Ollama tag |

Set `llm_knowledge_cutoff` to the model's training cutoff (or release date if unknown): headlines on or before it are scored with the keyword lexicon in every test, never trusted to the model's memory.

---

## 7. Model members

### 7.1 Intraday movement model (default, `intraday-train`)
`cat` (CatBoost), `xgb` (XGBoost) — gradient-boosted trees, combined by a **soft majority vote**: each month, only members whose trailing out-of-sample log-loss beats the base rate get a vote; their probabilities are averaged. No qualified member → the base rate (no trade). Logistic regression and LightGBM were removed after testing showed no skill.

### 7.2 Intraday zoo (`intraday-zoo`, run on the Mac)
- **Tabular:** `cat`, `xgb` (same as above, on the full feature set)
- **Sequence** (last 48 five-minute bars): `tcn`, `gru`, `lstm`, `patchtst`, `cnn` — pick via `--seq`
- Combined the same way: a soft majority vote, never a learned stacker (one was tried on the daily model and did worse than the plain average)

### 7.3 Time-series foundation-model features (`--tsfm`, optional, needs separate install)
| Name | Package |
|---|---|
| `chronos-bolt-small`, `chronos-bolt-base` | `pip install "chronos-forecasting>=1.5"` |
| `chronos-2` | `pip install "chronos-forecasting>=2.0"` |
| `timesfm-2.5` | `pip install "timesfm[torch] @ git+https://github.com/google-research/timesfm.git"` |
| `kronos-small`, `kronos-base` | clone `github.com/shiyu-coder/Kronos`, `pip install -r` its requirements |

These add forecast-derived *features* (median/quantile return forecasts), they aren't used as the final model on their own.

### 7.4 Daily model zoo (`zoo`, `train`'s own champion/challenger)
`logit`, `hgb`, `rf`, `et`, `svm` (scikit-learn); `xgb`, `lgbm`, `cat` (boosters); `lstm` (+ `tabm`, `tabpfn` if those packages are installed). Combined as `mean_all` (plain average) or `best3` (average of the 3 members with the best trailing log-loss, re-picked each block) — **`best3` is what's deployed**. MLP, CNN and Transformer were removed after testing showed no skill; a learned stacker was removed because it underperformed the plain average.

---

## 8. Paper-trading books

### 8.1 Always on, in the dashboard
| Book | Decides by | Auto-trade setting |
|---|---|---|
| Model + your rules | the movement model's probability + your ATM rules (§9) | `model_rules_auto` |
| LLM agent + your rules | the local LLM, same ATM rules, with guardrails | `agent_auto` |
| Manual (you) | the dashboard's Buy/Exit buttons | — |

### 8.2 Research strategy books (off by default; set `strategy_books` to enable)
| Book | When | Trade | Exit |
|---|---|---|---|
| `directional_itm` | 09:20, p ≥ 0.56 or ≤ 0.44 | 1 ITM call/put | 60% stop, 70% target, 15:15 |
| `debit_spread` | same + VIX in the top 40% of its year | ATM / ATM ± 150 | 50% stop, 80% target, 15:15 |
| `credit_spread` | same direction, VRP > 0 | sell the 0.75σ OTM spread, 150 wide | 1.5× credit stop, 60% capture, 15:15 |
| `iron_condor` | 09:20, VRP > 0, range model ≥ 0.80, no event | shorts at 1.25σ, 200-pt wings | 1.5× credit stop, 15:15 |
| `iron_fly_0dte` | 09:45 on expiry day | short ATM straddle, 1σ wings | 1× credit stop, 40% capture, 15:15 |
| `event_straddle` | 15:10 before RBI/FOMC/Budget | long ATM straddle | 10:30 next session |
| `turn_of_month` | 15:10, 2nd-last session of the month | ITM call | 15:10, 3rd session next month |
| `smc_intraday` | 5-min bars, 09:30–13:30 | sweep → BOS/CHoCH → FVG/OB retrace | structure stop, 2R target, 15:15 |
| `llm_analyst` | 09:20, LLM brief p ≥ 0.58 or ≤ 0.42 | 1 ITM call/put | 60% stop, 70% target, 15:15 |

---

## 9. Your ATM rules (the trade logic the engine actually runs with)

| Setting | Default | Meaning |
|---|---|---|
| `atm_entry_threshold` | 0.60 | enter a call at P(up by 15:15) ≥ this, a put at ≤ 1 − this |
| `atm_entry_start` / `atm_entry_end` | 09:20 / 13:30 | entry window |
| `atm_exit_time` | 15:15 | hard time exit |
| `atm_target_inr` | 5000 | take-profit, ₹ |
| `atm_extend_prob` | 0.80 | at target, hold `atm_extend_minutes` more if P(continues) > this |
| `atm_extend_minutes` | 30 | |
| `atm_stop_inr` | 2000 | hard stop, ₹ |
| `atm_against_bars` | 2 | exit after this many consecutive 5-min closes against the trade |
| `atm_against_scope` | `"in_profit"` | or `"always"` / `"after_target"` |
| `atm_lots` | 1 | |
| `atm_max_trades_per_day` | 1 | |
| `atm_news_exit_bps` | 8.0 | exit on a headline with learned impact ≥ this against the trade (only once the news-impact model passes its test) |

---

## 10. Research experiments (`config/experiments.json`, pre-registered)

| ID | Tests | | ID | Tests |
|---|---|---|---|---|
| P0 | base daily model (benchmark) | | S4 | intraday iron condor |
| P1 | + Asia opening gaps | | S5 | expiry-day iron fly, real prices only |
| P2 | + ADR/INDA premium | | S6 | long straddle into RBI/FOMC/Budget |
| P3 | + daily SMC structure | | S7 / S7b | turn-of-month, options / futures |
| P4 | + CPR/PDH/PDL/weekly levels | | S8 / S8b | SMC setups, options / index R-multiples |
| P5 | + HAR vol, VRP, efficiency ratio | | M1 | meta-labelling filter across books |
| P6 | + scheduled-event flags | | F1 | LLM pre-open brief vs the base model |
| P7 | + overnight futures/crude/BTC/USDINR | | F2 | live option-chain/depth recordings |
| P8 | + previous-close option chain | | S1 | ITM option buy on confident days |
| P9 | + pre-open news | | S2 | debit spread when IV is high |
| | | | S3 | credit spread when VRP > 0 |

Run with `python -m nifty_engine research`, or `--only P2,S4` for just those.

---

## 11. ATM replay variants (`atm-replay` / `intraday-ablation`'s sibling, `ATM_REPLAY_REPORT.md`)

Pre-registered, all reported: entry threshold × `against_scope` grid (`thr0.55/0.60/0.65` × `in_profit/always/after_target`), plus `thr0.60_no_extension`, `thr0.60_time_exit_only`, and the daily-model-entry set (`daily_in_profit`, `daily_always`, `daily_after_target`, `daily_stop2000_only`, `daily_time_exit_only`).

---

## 12. News sources

Free, polled on their own schedule, health-tracked in the dashboard: RBI, SEBI, the US Federal Reserve, NSE announcements/results/board-meetings/circulars, Economic Times, LiveMint, CNBC-TV18, BusinessLine, Moneycontrol, Business Standard, NDTV Profit, Financial Express, Investing.com, Google News (India + global macro), MarketWatch, CNBC World, plus the Upstox News API when a token is set. Add your own (e.g. an RSSHub Telegram feed) in `config/news_sources.json`. See `SOURCES_REPORT.md` for which actually answer from your network.

---

## 13. Dashboard reports (Reports tab)

| Key | File |
|---|---|
| `intraday` | `INTRADAY_REPORT.md` |
| `atm_replay` | `ATM_REPLAY_REPORT.md` |
| `news_impact` | `NEWS_IMPACT_REPORT.md` |
| `research` | `RESEARCH_REPORT.md` |
| `data` | `DATA_REPORT.md` |
| `llm_bench` | `LLM_BENCH_REPORT.md` |
| `sources` | `SOURCES_REPORT.md` |
| `backtest` | `BACKTEST_REPORT.md` |

Tabs: Live, Books & trades, Reports, Sources & alerts, Glossary (163 terms, hover any jargon anywhere in the UI).

---

## 14. Environment variables (`.env`, copy from `.env.example`)

| Variable | Purpose |
|---|---|
| `UPSTOX_ANALYTICS_TOKEN` | read-only Upstox token (recommended — real prices, chain recording) |
| `UPSTOX_ACCESS_TOKEN` / `UPSTOX_API_KEY` / `UPSTOX_PIN` / `UPSTOX_TOTP_SECRET` | alternative daily-refresh login |
| `NE_LLM_MODEL`, `NE_LLM_URL`, `NE_LLM_KNOWLEDGE_CUTOFF`, `NE_LLM_THINK` | local LLM overrides |
| `NE_<ANY_CONFIG_FIELD>` | overrides that field in `config/engine.json` for this run (e.g. `NE_ATM_STOP_INR=3000`) |

GitHub Actions collector secrets/variables (`DATA_COLLECTION.md`): `UPSTOX_ANALYTICS_TOKEN` (secret), `DATA_REPO_TOKEN` (secret), `DATA_REPO` (variable), `COLLECT_ENABLED` (variable, `true` to turn it on).
