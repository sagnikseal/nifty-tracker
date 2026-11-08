# NIFTY engine: live dashboard, your ATM rules, a local-LLM agent, and honest tests

The engine is one app you start with a double-click. It has:

1. **A movement model** that predicts NIFTY every 5 minutes: P(up over the next 30 minutes) and P(up by 15:15).
2. **A news-impact model**. It measures how NIFTY actually moved after each headline the LLM read, and learns reactions from that.
3. **Your ATM rules** as a paper-trading book, plus a **local-LLM trading agent** using the same rules, plus a **manual** book.
4. **A professional live dashboard**: live price, predictions, positions, the LLM's reading of each headline, the option chain, source health, reports, and hover explanations for every trading term.
5. **Faster news** from official sources (RBI, SEBI, NSE, the Fed) and major media, polled every 30-60 seconds. Each source's health is tracked.

Everything is paper trading. No code path can place a real order.

---

## 1. Start it

| Machine | One click |
|---|---|
| Mac | double-click `start.command` (first time: right-click → Open) |
| Windows | double-click `start.bat` |
| Linux | `./start.sh` |

The first run does the setup and takes 10-20 minutes:

- creates a Python environment and installs packages;
- downloads free market data;
- trains the daily and intraday models;
- downloads the local LLM, if Ollama is installed.

After that, start-up takes a few seconds and the dashboard opens at **http://localhost:8010**. Keep the window open while you want it running. It is safe to leave it running all week.

**Recommended:** put your Upstox **Analytics Token** in `.env` as `UPSTOX_ANALYTICS_TOKEN=`. It gives real-time NIFTY and VIX, real option bid/ask prices for paper fills, and option-chain recording. Without it, prices come from Yahoo (about 1 minute delayed) and option prices are modelled from India VIX. The dashboard says which source each number came from.

**Local LLM:** install Ollama from https://ollama.com. The launcher starts it and pulls the configured model.

### Can it be a single standalone app?

Not sensibly. The engine runs a Python model stack and a local LLM that must stay running in the background, so it can't be a pure web page.

What you get instead is the next best thing: a **one-click launcher** that starts the engine in a terminal window and opens the dashboard in your browser. The dashboard talks to the engine over a local connection (`localhost` only, nothing exposed to the network).

---

## 2. What happens during the day (IST)

| Time | What |
|---|---|
| 08:30 | refresh daily, hourly and intraday data |
| every 30-60 s | poll the news sources. New headlines are stamped with the time this machine first saw them, then scored by the LLM |
| 09:10 | option-chain and depth recorder starts (needs the Upstox token) |
| 09:20 | daily model probability; the LLM writes the pre-open brief |
| every 5 s | NIFTY, VIX and open-position prices. Stop, target and extension rules are checked on every update |
| every 5-min close | movement-model predictions, the "2 closes against" rule, model-book entries, and the agent's decision |
| 15:15 | time exit |
| 15:45 | end of day: equity, intraday data stored |
| Fri 16:15 | retrain the daily model, the movement model and the news-impact model |

Each of these runs in its own thread. A slow news source or a slow LLM can never delay a stop-loss.

---

## 3. Your ATM rules (book `model_rules`; the agent book uses the same exits)

Settings live in `config/engine.json` (`atm_*`). All amounts are for the whole position.

| Rule | Default |
|---|---|
| Instrument | 1 lot (65) of the at-the-money weekly option: a call if bullish, a put if bearish. It costs about ₹7,000 at today's prices, the least money of the strategies tested |
| Entry | P(up by 15:15) ≥ 0.60 for a call (≤ 0.40 for a put), between 09:20 and 13:30, at most 1 trade a day |
| Stop | loss reaches ₹2,000: exit immediately (checked every 5 s) |
| Target | profit reaches ₹5,000. If P(the move continues over the next 30 min) > 80%, hold 30 more minutes and re-check (this can repeat); otherwise exit |
| Against the prediction | NIFTY closes against the trade on 2 consecutive 5-minute bars while the trade is in profit: exit and keep the profit (`atm_against_scope`: `in_profit`, `always` or `after_target`) |
| Time | exit at 15:15 |
| News reaction | exit if a new headline's learned impact is ≥ 8 bps against the trade. It only switches on once the news-impact model passes its test |

The same code (`nifty_engine/strategies/atm_rules.py`) runs the historical replay and the live paper trader. What was tested is exactly what trades.

---

## 4. Which decides better: the model ensemble or the LLM agent? (your options)

You don't have to pick in advance. The engine runs the options side by side on paper with the same rules and the same capital, and measures them:

| Option | Book | How it decides | How it is tested |
|---|---|---|---|
| A. Model ensemble + rules | `model_rules` | the movement model's calibrated probabilities | walk-forward on 2022-2026 history, plus live paper |
| B. LLM agent + rules | `agent` | the local LLM reads the whole state (price action, VIX, the model and its measured accuracy, fresh news with learned impacts, the option chain) and decides; code guardrails always win | live paper only, explained below. Every decision's P(up 30 min) is scored against what NIFTY then did |
| C. You | `manual` | buttons in the dashboard | live paper |
| Hybrid | (both A and B) | the agent sees the model's output and its track record; the model sees the news features | compare A and B on the dashboard |

**Why the agent can't be backtested on years of history:** an LLM may "remember" what happened after old headlines (look-ahead bias). Its knowledge cutoff also limits which headlines are fair to use. So the agent is judged only on live paper decisions. Give it at least 3 months of paper trading before trusting it either way.

---

## 5. What the tests found (2026-11-08, data to 2026-11-03)

Reports: dashboard → Reports, or `store/reports/*.md`.

### Movement model (`INTRADAY_REPORT.md`)

- **Setup:** walk-forward on 942 days (Jan 2023 to Nov 2026), 62,172 predictions; the last 6 months were held out untouched.
- **Result: no edge from price and VIX alone.**
  - Next 30 minutes: accuracy 50.8% (holdout 49.9%), with no log-loss skill over the base rate (p = 0.48).
  - Now to 15:15: accuracy 49.2% (holdout 49.1%).
- Each month, the walk-forward kept choosing the base rate because nothing beat it on trailing data. **Not a single prediction reached 80%**, so the "hold if above 80%" rule never fires with a calibrated model. That is an honest property of 30-minute index moves, not a bug.

### Your ATM rules on real option prices (`ATM_REPLAY_REPORT.md`)

- **A. Intraday-model entries:** 0 trades. The model never became confident enough to enter.
- **B. Daily-model entries at 09:20** (the one signal with a proven out-of-sample edge, 54.4% on 2014-2026): 32 trades from Sept 2024 to Oct 2026, average premium about ₹6,950.
  - Every variant was positive but **none is significant** (t ≤ 1.2).
  - Your exits made +₹56 per trade: the 2-against rule fired 15 times, the stop 14, the target only 3.
  - Holding to 15:15 made +₹786 per trade.
- Tight exits cut the winning days short, the same pattern as the earlier `VARIANTS.md`. The ₹5,000 target needs about a +72% move in an ATM premium, which happened on 9-25% of trades.

### What this means

On the data available today, nothing here is proven to make money after costs. Keep everything on paper.

The two inputs not yet tested are the ones most likely to add something intraday:

- **Fresh news with learned impacts:** needs about 2-3 months of collection plus `news-backfill`.
- **Recorded option-chain positioning:** needs 60 sessions of recordings.

Re-run the reports as data accumulates (they update automatically every Friday).

---

## 6. Choosing the local LLM (16 GB MacBook Air M5)

Measure on your own Mac: `python -m nifty_engine llm-bench` scores each installed model. It checks valid JSON, direction on clear headlines, neutrality on noise, channel, consistency, speed, memory, and 3 agent scenarios.

| Model (Ollama tag) | Size (4-bit) | Fits next to the engine on 16 GB? | Notes |
|---|---|---|---|
| **qwen3.5:9b** (default) | ~6.6 GB | yes, ~4 GB headroom | 2026 model; the most consistent pick for 16 GB Macs in current guides; good at structured JSON. Estimated ~26 tokens/s on M5 |
| gemma4:12b | ~8 GB | yes, tighter | stronger reasoning, slower (~17 tokens/s estimated) |
| llama3.1:8b (previous default) | ~5 GB | yes | older (2024); reliable JSON, but generally ranked below Qwen 3.5 now |
| qwen3:8b | ~5 GB | yes | widely supported fallback |

**Is Llama 3.1 8B good enough?** For scoring headlines into structured fields, yes, it works. As a trading agent that weighs mixed evidence, newer 9-12B models do better on reasoning benchmarks, so the engine defaults to `qwen3.5:9b`. Confirm with `llm-bench` on your Mac before relying on either.

Avoid models above about 12B on a 16 GB Air: with the engine, the browser and macOS, they will swap memory and slow everything down.

**Knowledge cutoff:** set `llm_knowledge_cutoff` to the model's cutoff, or its release date if unknown. Headlines older than that are scored with the keyword lexicon in all tests.

---

## 7. Time-series models: what the research says, and what the engine uses

| Family | Examples | Evidence for daily/intraday index returns | In the engine |
|---|---|---|---|
| Gradient-boosted trees | CatBoost, XGBoost (intraday); random forest, extra trees, CatBoost, LightGBM, XGBoost, SVM, logistic (daily) | strongest in the finance literature on low signal-to-noise returns (Gu-Kelly-Xiu 2020; Rahimikia 2025 finds they beat zero-shot foundation models on daily returns) | daily model (best-3 of the daily zoo: random forest, CatBoost, extra trees) and the intraday zoo |
| Time-series foundation models | Chronos-2, TimesFM-2.5, Moirai-2, TiRex, Toto; Kronos (trained on candlesticks) | top of the GIFT-Eval leaderboard for general forecasting, but weak zero-shot on financial returns; a 2026 study ranks TimesFM-2.5 and Moirai-2 best among them on stock returns, without trading results | not yet: candidate feature group (their forecasts as inputs), to be pre-registered and tested like any other |
| Deep sequence models | TCN, PatchTST, GRU (intraday); LSTM, TabM (daily) | mixed; overfit easily on noisy returns | intraday zoo and daily zoo, to be confirmed on the Mac. MLP, CNN and Transformer were removed from the daily zoo after scoring no better than the base rate |
| LLM agent | local Qwen/Gemma/Llama | no reliable evidence of trading skill yet; useful for reading news | `agent` book, measured live |

Ensembles only help when their members have some skill. Averaging coin flips gives a coin flip. The engine therefore chooses members by measured trailing performance, and falls back to the base rate when none has skill.

---

## 8. News sources

Live check from this machine on 2026-11-08 (`python -m nifty_engine sources-check` runs it on yours):

- **Working:** RBI press releases and notifications, SEBI, US Federal Reserve, Economic Times (markets, economy), LiveMint, CNBC-TV18, BusinessLine, Google News (India markets, global macro), MarketWatch.
- **Blocked or flaky from this connection (12 of 23 worked):**
  - NSE archives returned 503 even after a browser-style session warm-up;
  - Moneycontrol returned 403 even with browser headers;
  - Business Standard and CNBC World returned 503;
  - Financial Express answered without a parseable feed;
  - NDTV Profit and Investing.com timed out at 15 s.

  They may work from your connection. A failing source backs off to one try every 30 minutes, so keeping it costs almost nothing.
- The Upstox News API (with your token) is polled every 5 minutes for the heavyweight stocks.
- GDELT is used only for the ~3-month history backfill (`news-backfill`). It is too slow for live use (15-minute batches, strict rate limit); live captures that touch it fail fast.
- **Add your own** in `config/news_sources.json`. Telegram channels work through an RSSHub route.

---

## 9. Commands

```bash
python -m nifty_engine app              # everything + dashboard (what the launchers run)
python -m nifty_engine intraday-train   # movement model, about 2 min -> INTRADAY_REPORT.md
python -m nifty_engine atm-replay       # your rules on real option prices -> ATM_REPLAY_REPORT.md
python -m nifty_engine news-backfill    # ~3 months of GDELT headlines (once), then:
python -m nifty_engine news-impact      # how NIFTY reacted to the LLM's reading -> NEWS_IMPACT_REPORT.md
python -m nifty_engine llm-bench        # pick the local model -> LLM_BENCH_REPORT.md
python -m nifty_engine sources-check    # test news sources -> SOURCES_REPORT.md
python -m nifty_engine data-report      # data inventory, gaps and how to fill them -> DATA_REPORT.md
python -m unittest discover -s tests    # 58 tests incl. look-ahead tests for every new feature
```

---

## 11. More data, 189 features, a model zoo (2026-11-08)

### Data added (`python -m nifty_engine fetch-history`, then topped up daily)

- **Bank Nifty and 10 heavyweight stocks:** 5-minute bars since Jan 2022 (Upstox V3; minute data starts in 2022).
- **Continuous near-month NIFTY futures** with volume and open interest: 23 expired monthly contracts (Upstox Plus) plus the live one. They start in Sept 2024, because Upstox serves expired contracts only about 2 years back.
- **Intraday option chain, Oct 2024 onwards,** rebuilt from the real 1-minute candles of expired options (price, volume, OI per strike). Upstox does not sell historical chain snapshots.
- **Global markets during the day:** hourly ES/NQ futures, USD/INR, crude and gold.
- **Daily macro context and the event calendar.**

### What the option chain is

The option chain is the table of every strike for one expiry. For each call and put it shows the price, bid/ask, traded volume, **open interest** (contracts still open), implied volatility and Greeks.

Traders read positioning from it:

- put/call ratios;
- where writers are building "walls";
- how much movement options are pricing (straddle, implied volatility);
- skew.

It wasn't used at first because there was no history: the engine records it live, and 0 sessions had been recorded. It is now rebuilt from the expired-option candles.

### Features (189; `nifty_engine/analytics/patterns.py`, `cross_asset.py`, `chain_intraday.py`)

| Group | What it adds |
|---|---|
| Technicals (75) | EMA 9/21/50 distances, slopes and alignment; MACD; RSI; ADX/DI; Supertrend; Bollinger %B and width; TTM squeeze; stochastic; CCI; ROC; ATR regime; efficiency ratio; variance ratio; skew; Ichimoku |
| Multi-timeframe | 15- and 60-minute versions of the above, using completed bars only |
| Candles | engulfing, hammer, shooting star, doji, marubozu, 3 soldiers/crows, stars, inside/outside, NR7, Heikin-Ashi runs |
| Structure | confirmed swing highs and lows (HH/HL), Donchian breakouts, distance to round numbers |
| Cross-asset | Bank Nifty leadership; breadth of the heavyweights; futures basis, VWAP, volume surge, OI regime (long/short build-up, covering, unwinding); option-chain PCR (OI and volume), OI flows, ATM IV and skew, implied move, walls, call-vs-put momentum; global moves during the day; macro; events |

Every group has a look-ahead test (`tests/test_feature_groups.py`).

### Models (`intraday_stack.py`, `intraday_zoo.py`; run on the Mac)

- **Tabular:** CatBoost, XGBoost. Logistic regression and LightGBM were removed after the walk-forward showed no skill (about 50.8% at 30 minutes, holdout 49.9%).
- **Sequence networks** over 48 five-minute bars of NIFTY, VIX, Bank Nifty and futures: TCN, PatchTST, GRU (plus LSTM and CNN, available but untested).
- **Optional foundation-model features:** Chronos-Bolt/Chronos-2, TimesFM-2.5, Kronos.
- **A soft majority vote,** not a learned combiner. Each month, only models whose recent out-of-sample record beats the base rate get a vote. The prediction is the average of the voters' probabilities, and the dashboard shows how many agree. If no model qualifies, the result is the base rate and there's no trade. A learned stacker was dropped because, in the daily zoo, it did worse than the plain average.

Run it with `python -m nifty_engine intraday-zoo` on the Mac (Apple GPU). Add `--tsfm chronos-bolt-small` after installing the optional packages listed at the end of `requirements.txt`.

### Results: honest, including a leak that was caught

Pre-registered ablation (`INTRADAY_ABLATION_REPORT.md`): run with logistic + LightGBM (since removed for having no skill), identical out-of-sample rows, day-clustered tests.

| Added | 30-min accuracy (same rows) | Log-loss gain p | Verdict |
|---|---|---|---|
| Technicals and patterns (75) | 50.9% → 51.1% | 0.29 | no gain |
| Bank Nifty, breadth, macro, events | 51.1% → 51.2% | 0.70 | no gain |
| Futures, option chain, global intraday | 50.2% → 50.2% | 0.26 | no gain |

**A leak was found and fixed.** The first run showed the option chain lifting accuracy to 53% (p = 1e-7; holdout 54%).

- **The cause:** each weekly expired-option download contains the strikes needed for *every* day of that week, chosen from each day's open. The set of strikes present on Monday therefore partly reflected where NIFTY went later, and the put/call ratios summed over that set.
- **The fix:** the chain features now use only a strike window fixed at 09:15. The same fix is applied to the daily `options_hist` features.
- **After the fix,** the effect vanished (50.2% → 49.6%, holdout 48.7%).

The lesson: any feature built from "which contracts exist in the data" must be checked for selection leaks.

**Conclusion so far:** with five years of NIFTY, Bank Nifty, stocks and VIX, two years of futures and option-chain history, and global markets, no feature group predicts the next 30 minutes or the move to 15:15 better than the base rate, out of sample.

The two inputs still untested are:

- **news with first-seen timestamps:** collecting now;
- **live order-book depth:** recorded only going forward.

The sequence networks and foundation models still have to be run in full on the Mac. Treat any edge they show with the same suspicion, and confirm it on the holdout and in paper trading.

### GitHub Actions collector (`docs/DATA_COLLECTION.md`)

- **What it collects:** option-chain snapshots and quotes with depth every 15 minutes in market hours; end-of-day 1/5-minute candles for NIFTY, VIX, Bank Nifty, futures and heavyweights; expired options after each expiry; news every 15 minutes, around the clock.
- **Where it goes:** a `data` branch; `python -m nifty_engine.collect sync` pulls it to the Mac.
- **Off by default:**
  - set the repository variable `COLLECT_ENABLED=true` to turn it on;
  - your repository is public, so first point `DATA_REPO` at a **private** repository;
  - exchange and broker data may not be redistributed publicly.

## 10. Moving to real money later

Only consider it after a book passes its replay test **and** holds up in at least 3 months of live paper trading.

It would then need:

- a full-access broker token (the Analytics Token is read-only);
- compliance with SEBI's 2025 retail-algo framework (broker-registered algo, static IP for order APIs);
- broker-side stop-loss orders (exchange-side, millisecond triggers), not loop-based exits;
- a daily loss kill-switch.

The rules engine is already separate from execution, so this would mean adding an order adapter, not rewriting the strategy.
