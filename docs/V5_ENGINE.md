# NIFTY engine v5: setup, daily use and how to test every idea

v5 adds four things on top of the v4 walk-forward engine:

1. **More data**, all free:
   - your Upstox Analytics Token (option chain, depth, OI, news, expired contracts with Plus)
   - related markets: overnight US futures, Asia opening gaps, the ADRs of NIFTY heavyweights, the INDA ETF
   - scheduled events
2. **Market analysis**:
   - SMC structure (BOS/CHoCH, order blocks, FVGs, liquidity sweeps, equal highs/lows)
   - classic indicators (VWAP, ORB, CPR, Supertrend, ADX)
   - option-chain microstructure (PCR, OI walls, max pain, GEX and zero-gamma, skew, implied move, build-up)
   - a volatility regime forecast (HAR-RV)
3. **A local LLM** (Ollama, on your Mac) that reads news and the day's numbers like an analyst. It produces structured, timestamped features and a pre-open brief.
4. **Nine strategy books, paper-traded side by side**, plus a **research harness**. The harness replays every book on real expired-option prices and tests every new feature group against the base model on the same days. The rules are fixed in advance, and losses are reported too.

Nothing costs money to run. The engine never places orders: the Analytics Token is read-only.

---

## 1. One-time setup on the Mac

```bash
git pull
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-engine.txt            # add requirements-zoo.txt for the full model zoo

# Local LLM (free). The 8B model needs about 5 GB of RAM.
brew install ollama
ollama serve &                                    # or open the Ollama app
ollama pull llama3.1:8b
python -m nifty_engine llm-check                  # should print available: true and a scored test headline

cp .env.example .env                              # then set UPSTOX_ANALYTICS_TOKEN=...
```

**Choosing a local model.** Any Ollama instruct model works. Set `llm_model` in `config/engine.json`, and set `llm_knowledge_cutoff` to that model's training cutoff. Headlines first seen on or before the cutoff are scored with the keyword lexicon in every feature and test, so the LLM can't "remember" what happened next. `llama3.1:8b` (cutoff December 2023) is the default because it handles structured JSON output reliably. A larger or newer model may read news better. Compare them with experiment P9 after a few months of data.

## 2. First data download (once; the long steps can run overnight)

| Step | Command | Time | What it gives |
|---|---|---|---|
| Daily + hourly global data | `python -m nifty_engine update-data` | 2-5 min | 28 daily series since 2007; ~2 years of hourly ES/NQ/crude/gold/BTC/USDINR |
| 5-minute NIFTY + VIX since 2022 | `python -m nifty_engine fetch-intraday` | 5-10 min | needed by every strategy book |
| Train the champion | `python -m nifty_engine train` | 5-15 min | walk-forward model and calibration |
| Real expired-option prices | `python -m nifty_engine fetch-expired-options --mode full --since 2024-10-01` | **hours** (Upstox rate limits; resumable) | ATM ± 800 points for every weekly expiry; real-price replays |
| Historical option-chain features | `python -m nifty_engine build-option-features` | minutes | previous-close PCR, walls, IV, skew, GEX per session |
| ~3 months of news | `python -m nifty_engine news-backfill` | 20-60 min (GDELT + LLM scoring) | first news history after the LLM cutoff |

## 3. Test everything

```bash
python -m unittest discover -s tests -v           # 38 tests, including leakage tests for every new feature
python -m nifty_engine research                   # every experiment; report at store/reports/RESEARCH_REPORT.md
python -m nifty_engine research --only S4,S8      # just some of them
```

Experiments are pre-registered in [`config/experiments.json`](../config/experiments.json). The thresholds were fixed before any test ran. If you change a rule after seeing results, that rule no longer tells you anything.

| ID | What is tested | Pass rule (summary) |
|---|---|---|
| P0 | base daily model | reported (benchmark) |
| P1-P6 | asia_extra, adr, smc, structure, regime, events feature groups, each vs base on identical days | lower log-loss, Diebold-Mariano p < 0.10, ≥ 250 days |
| P7-P9 | overnight_futures, options_hist, news (short history), stacked on the base probability | p < 0.10 on ≥ 120 out-of-sample days |
| S1-S8 | the strategy books, on real option prices where downloaded, VIX-modeled otherwise | net/trade > 0, t ≥ 2, deflated Sharpe ≥ 0.95 across all trials, ≥ 50% real prices |
| S7b, S8b | turn-of-month on futures since 2007; SMC setups in R-multiples on the index | reported |
| M1 | meta-labelling filter across all books | better net/trade while keeping ≥ 70% of trades |
| F1, F2 | the LLM brief vs the model; live chain/depth recordings | need 60 recorded sessions |

**Promoting a feature group.** If, say, P2 passes on your full run (not `--quick`), set `"feature_groups": "base,adr"` in `config/engine.json`, then run `python -m nifty_engine train`. The live signal always uses exactly the groups its model was trained with, so changing the config can't silently mismatch the model. Promote one group at a time and re-run research after each. Every group you test is another trial: with 6 groups, a p-value just under 0.10 can be luck.

**Enabling a book for paper trading.** The books in `strategy_books` all paper-trade from day one, so they collect live evidence. Each book has its own ₹1L paper equity. Treat a book as tradable with real money only once it passes in research **and** holds up in at least 3 months of live paper trading.

## 4. Daily operation

```bash
python -m nifty_engine run        # scheduler + dashboard at http://localhost:8010/paper
```

| IST | What happens |
|---|---|
| 08:40 | refresh data; capture and score news (RSS, Upstox News API, GDELT) with the local LLM |
| 09:10-15:31 | **recorder**: full option chain every 30 s (REST) and 30-level depth for futures + ATM ± 10 strikes, plus heavyweight stocks for breadth (WebSocket, Upstox Plus) → `store/live/<date>/` |
| 09:20 | v4 signal and ITM trade; range model; the local LLM writes the pre-open brief |
| 09:20-15:25 | v5 books at their own times; SMC checks every closed 5-minute bar from 09:30 to 13:30; stops and targets |
| every 30 min | news capture |
| 15:45 | EOD: equity per book, outcomes, data refresh |
| Fri 16:15 | retrain (champion/challenger) |

**Recordings are the most valuable data you will own.** Option-chain and depth history can't be downloaded later, and experiment F2 needs 60 sessions of it. Leave `run` (or `python -m nifty_engine record`) going on market days.

Dashboard endpoints: `/api/v5/books`, `/api/v5/brief`, `/api/v5/chain`, `/api/v5/research`.

## 5. The strategy books (rules fixed in `nifty_engine/strategies/books.py`)

| Book | When | Trade | Exit |
|---|---|---|---|
| directional_itm | 09:20, p ≥ 0.56 or ≤ 0.44 | 1 ITM call/put | 60% stop, 70% target, 15:15 |
| debit_spread | same, and India VIX in the top 40% of its year | ATM / ATM ± 150 | 50% stop, 80% target, 15:15 |
| credit_spread | same direction, VRP > 0 | sell the 0.75σ OTM spread on the faded side, 150 wide | 1.5 × credit stop, 60% capture, 15:15 |
| iron_condor | 09:20, VRP > 0, range model ≥ 0.80, no event | shorts at 1.25σ, 200-pt wings | 1.5 × credit stop, 15:15 |
| iron_fly_0dte | 09:45 on expiry day, VRP > 0, gap < 1σ, no event | short ATM straddle, 1σ wings | 1 × credit stop, 40% capture, 15:15 |
| event_straddle | 15:10 before an RBI/FOMC/Budget reaction session | long ATM straddle | 10:30 next session |
| turn_of_month | 15:10 on the 2nd-last session of the month | ITM call that outlives the hold | 15:10 on the 3rd session of the next month |
| smc_intraday | 5-min bars 09:30-13:30 | sweep → BOS/CHoCH → retrace into FVG/OB; ITM option | structure stop, 2R target, 15:15 |
| llm_analyst | 09:20, LLM brief p ≥ 0.58 or ≤ 0.42 | 1 ITM call/put | 60% stop, 70% target, 15:15 |

In live paper trading, `directional_itm` is the existing v4 `options` book (`books` in `config/engine.json`), so it isn't listed again in `strategy_books`. Research tests it as S1.

VRP = India VIX's implied daily volatility minus the HAR-RV forecast. When it is positive, options are pricing more movement than the volatility model expects.

## 6. Other things that may improve accuracy, and where they stand

| Idea | Status |
|---|---|
| Overnight US futures as a GIFT Nifty proxy | built; P7 (history grows daily) |
| ADR / INDA premium over the NSE close | built; P2 |
| More Asian opens (KOSPI, Taiwan, Shanghai) | built; P1 |
| Option-chain state at the previous close and at 09:20 | built; P8 (history), F2 (live) |
| News read by a local LLM, with look-ahead protection | built; P9, F1 |
| Volatility regime decides buy vs sell strategies | built (VRP filter in the books); P5 |
| Meta-labelling: a second model decides whether to take each trade | built; M1 |
| Fractional-Kelly sizing on each book's own win/loss profile | built (`meta/stacker.py: kelly_lots`); used after a book passes |
| Deflated Sharpe and PBO, so many trials don't fool us | built; in every strategy result |
| Pre-open breadth of the 50 stocks, futures basis, OI build-up | recorded live (depth/ltpc feed); testable after 60 sessions |
| FII/DII flows and participant-wise OI | loader exists (`load_participant_oi`); NSE forbids scraping, so save the CSVs into `store/nse/participant_oi/` by hand |
| GIFT Nifty itself | not on any free feed or on Upstox; ES/NQ proxy instead |
| Triple-barrier labels (predict "target before stop", not up/down) | next step: needs the intraday labels from the recordings |
| Conformal prediction for calibrated "no-trade" zones | next step |

## 7. Limits to keep in mind

- **Modeled vs real prices.** Replays fall back to VIX-modeled premiums wherever real expired prices weren't downloaded. Real put premiums carry a skew the model misses. Every trade records its `price_source`, and a strategy only passes if at least half its trades are on real prices.
- **0DTE history.** Upstox may not serve expiry-day bars for expired weeks. In that case S5 shows INSUFFICIENT_DATA until the recorder has collected expiry days.
- **Upstox Plus WebSocket limits.** Upstox's own pages disagree on the socket count (3 or 5). The recorder uses 2 sockets, with at most 50 instruments on the `full_d30` one.
- **Going live with real money** would need a full-access token (the Analytics Token can't place orders) and SEBI's 2025 retail-algo framework (broker-registered algo, static IP for order APIs). It would also need separate kill-switches. None of that is built, on purpose.
