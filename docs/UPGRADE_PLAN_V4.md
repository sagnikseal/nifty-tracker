# NIFTY prediction engine: v4 upgrade plan and research findings

_Prepared 3 Nov 2026 for `sagnikseal/nifty-tracker`. All v4 numbers below come from running the code on this branch on real data (NIFTY, India VIX and 11 global series, Sep 2007 to Nov 2026) on a Windows machine. They are not projections._

---

## 0. TL;DR

1. **The current v3 model has no predictive power. Its 97% accuracy comes from label leakage.** `scripts/build_real_training_data.py` builds 9 of the 16 features as `dir_sign * random_noise`, where `dir_sign` *is the label*. It then adds 1,040 hand-written "crisis archetype" rows, shuffles everything, and tests on the same synthetic distribution. Nothing in v3 has been tested on real out-of-sample data.
2. **Realistic accuracy for daily NIFTY direction is 52–58%, not 90%+.** That holds for the peer-reviewed literature and for this engine.
   - v4's walk-forward accuracy over 3,094 unseen sessions (2014–2026) is **53.0%** for all days, and **57.1% on the 42% of days it is confident about**.
   - Papers that claim 90%+ almost always have leakage like v3's.
3. **The tradable edge is small, real-looking, and asymmetric.**
   - NIFTY loses on average between the open and the close: −6.9 bps per day, negative in 12 of the last 13 years. Its gains come overnight.
   - The model's confident *bearish* calls hit 58.2% (+15.4 bps/day across 1,140 days).
   - Its bullish calls have no edge (49.4%).
4. **How you trade the signal matters as much as the signal.** In a 10-variant replay with full 2026 NSE charges:
   - Single-leg ITM option buys made money.
   - Debit spreads lost money every time, because 4 fills of spread and slippage eat a 13-bps edge.
   - Tight 45% stops hurt in both styles.
5. **Default paper config:** ₹1,00,000 capital, 1 lot, ITM option buy, entry 09:20, exit by 15:15, signal threshold 0.56, 60% disaster stop.
   - Modeled 12.6-year replay: 1,302 trades, **+₹5.55 lakh on 1 lot**, max drawdown −16%, 66% of months positive, median month +1.1%, worst month −7.2%.
   - The last 12 months were about +9%.
   - Treat these as **upper bounds**. Premiums are modeled, not historical quotes (see §4.4).
   - **Real-premium check (§4.5):** on actual expired-option prices for Oct 2024–Oct 2026, the same strategy made about half the modeled profit: +₹43k on 1 lot over 64 trades, a 45% win rate and a −24% max drawdown on ₹1L.
6. **A one-month paper test is too short to judge the trading results.** Recently the model is confident only about 2–4 times a month. The paper trader therefore also scores **every** daily prediction (about 22 a month). Pass/fail criteria are fixed in advance in §6.

---

## 1. Audit of the current repository (v3)

| # | Severity | Finding | Where | Fix in v4 |
|---|---|---|---|---|
| 1 | **Critical** | **Label leakage.** Features are generated as `dir_sign * rng.normal(...)` where `dir_sign = +1 if label==1`. The model reads the answer, which is where its 97.2% "accuracy" and 0.038 Brier score come from. | `build_real_training_data.py` (`extract_samples_from_eod_dataset`) | Every feature is computed from real market data with explicit timing rules. A unit test proves features don't change when the future is altered. |
| 2 | **Critical** | **Synthetic "event archetypes".** 1,040 rows are hand-typed feature "centres" plus jitter, labelled 92% in the scripted direction. This is not market data. | same file, `HISTORICAL_NSE_EVENT_ARCHETYPES` | Removed. Crisis regimes are learned from real 2008/2020/2022/2024 sessions. |
| 3 | **Critical** | **Random shuffle before the train/test split.** This mixes future and past, and isotonic calibration is fitted on the same test set it is evaluated on. | `train_ml_model.py` | Expanding-window walk-forward with periodic refits. Calibration is fitted only on *earlier* out-of-sample predictions. |
| 4 | High | **Synthetic stress test.** The backtest draws features from the same generator. | `backtest_stress_test.py` | Real walk-forward replay with option pricing and full NSE charges. |
| 5 | High | **Synthetic fallbacks on live screens.** These include a sine-wave option chain, a sine-wave in-browser "engine" (`Math.sin` in `realtime_client.js`), and a synthetic EOD series when Yahoo fails. | `realtime_engine.py`, `realtime_client.js`, `fetch_nifty.py` | v4 never fabricates prices. When data is missing it stands aside, and every paper trade records its `price_source`. |
| 6 | High | **Wrong expiry day.** The code assumes Thursday, but NIFTY weekly expiry moved to **Tuesday** from Sep 2025. | `get_next_nifty_weekly_expiry` | `calendar.py` uses Tuesday from 1 Sep 2025 and Thursday before, and never trades 0DTE. Live trading uses the broker's expiry list. |
| 7 | High | **Wrong lot/multiplier.** GEX uses ×50, but the NIFTY lot is **65** from the Jan 2026 series. | `compute_pillar2_*` | `lot_size=65` (configurable). |
| 8 | Medium | **Incomplete bar written as a close of 0.0.** `data/nifty_data.json` shows 2026-11-02 with close=0.0, change −100% and a garbage pivot. | `fetch_nifty.py` | Fixed: incomplete bars are dropped. v4 also holds back today's bar until 18:00 IST. |
| 9 | Medium | **Hard-coded pillar weights (28/34/18/20%)** and hand-set "gates", never estimated from data. | `realtime_engine.py` | Weights are learned, and walk-forward decides which features survive. |
| 10 | Medium | **Only one year of history** (246 sessions). | `fetch_nifty.py period="1y"` | 19 years of daily data across 13 series. |
| 11 | Low | **Weak default admin PIN** (`1234`) on an endpoint that accepts broker tokens. | `.env.example` | v4 exposes no token-accepting endpoint. Change the legacy PIN. |

The v3 *ideas* (GEX, PCR velocity, heavyweights, macro, news) are reasonable hypotheses. They just can't be tested without real historical data for each one, which is the gap this plan closes step by step (§5).

---

## 2. What the research says (what works and what doesn't)

### 2.1 Expectations

- **Daily index direction has a very low signal-to-noise ratio.**
  - Gu, Kelly & Xiu (2020, *RFS*) find monthly out-of-sample R² of about 0.4% even with neural nets, though that still produced an S&P 500 timing Sharpe of 0.77 versus 0.51 for buy-and-hold.
  - On SPY, a careful walk-forward XGBoost gets 58.5% *after* seeing the open, against 52.6% for always-up (Chen 2026).
- **Edges decay.** Krauss, Do & Huck (2017, *EJOR*) show ML strategies on the S&P 500 went from Sharpe 6.7 in the 1990s to *losing* money after 2010 once costs were included.
- **Most published Indian "90%+" results are leakage.**
  - One 2023 *JRFM* paper reports "95% accuracy" by predicting price *levels* on a single 80/20 split with no naive baseline.
  - Kapoor & Narayanan (2023, *Patterns*) document leakage errors in 329 papers across 17 fields.

### 2.2 Signals with real evidence (and how v4 uses them)

| Signal | Evidence | v4 status |
|---|---|---|
| **Overnight global lead**: US close, Asian opening gaps, GIFT Nifty | GIFT/SGX Nifty leads price discovery (68% component share; Sarkar et al. 2025). DJIA drives SGX Nifty overnight (Sundararajan & Balasubramanian 2023). | **In.** US/commodity/FX lagged one session, HK/Japan same-morning gaps. `hsi_gap` is the #1 feature. GIFT Nifty is not on free feeds: add via broker (phase 1). |
| **Intraday vs overnight return split** | Intraday returns are often negative while overnight carries the premium (night effect). Confirmed here: NIFTY open→close averages −6.9 bps. | **In.** This is the main source of the short-side edge. |
| **Intraday momentum** (first half-hour → last half-hour) | Gao, Han, Li & Zhou (2018, *JFE*). Baltussen et al. (2021, *JFE*) attribute it to short-gamma hedging by option dealers. | **Phase 2** (needs 5-min bars; Upstox v3 has them from Jan 2022). |
| **Volatility is predictable** (unlike direction) | HAR-RV (Corsi 2009). India VIX beats GARCH for realised vol (Shaikh & Padhi 2014). Foundation models rarely beat log-HAR (Brini 2026). | **Partly in** (HAR components as features). **Phase 3:** use the vol forecast for sizing and for choosing between option buying and selling. |
| **India VIX** | Predicts volatility, not direction. Asymmetric response to falls. | **In** (level, change, z-score, implied-minus-realised). |
| **FII/FPI flows** | Daily FII flow shocks move Indian prices the same day, and about 40% reverses within 5 days (Acharya, Anshuman & Kiran Kumar 2016). | **Phase 1** via the NSE participant-wise OI and FII/DII CSV loader (already coded: `load_participant_oi`). No peer-reviewed proof that participant OI predicts NIFTY, so walk-forward must earn it. |
| **Time-series momentum** | Moskowitz, Ooi & Pedersen (2012). | **In** (r5–r60, distance to SMA20/50/200). |
| **News / LLM sentiment** | GPT-4 headline scores predict next-day drift, mostly in small caps (Lopez-Lira & Tang, WP 2025). On Indian data, news adds only small gains (Dasgupta et al.; +5.2 pts accuracy in Kotekar et al. 2025). LLMs have look-ahead and "distraction" bias in backtests (Glasserman & Lin 2023). | **Collected forward** (lexicon + GDELT tone) from day 1. It enters the model automatically after about 1 year of history. LLM scoring is phase 4, live only. |
| **Order-book deep learning** (DeepLOB) | Works on tick-level LOB data (Zhang, Zohren & Roberts 2019). | **Out of scope.** Needs paid NSE tick data and co-location economics. |

### 2.3 What not to build

- **LLM multi-agent "trading desks"** (TradingAgents, FinMem, FinGPT). After the LLM's knowledge cutoff, returns fall 50–72% and Sharpe 51–62% ("Profit Mirage", 2025). Over 20 years, the reported edge mostly vanishes once survivorship and data-snooping are controlled (FINSABER, KDD 2026). A reproduction of TradingAgents found Sharpe halving to 0.22 after costs ("Alpha Illusion", 2026). Use LLMs only to turn news into features.
- **Zero-shot time-series foundation models** (Chronos, TimesFM) for returns. Out-of-sample R² is negative (−1.4% to −2.8%) and below CatBoost (Rahimikia et al. 2025).
- **Big models on daily data.** Five thousand rows can't support them. Gu-Kelly-Xiu's best results come from cross-sections of thousands of stocks.

---

## 3. v4 architecture (shipped in this branch)

```
                      ┌────────────── broker-independent ──────────────┐
 Yahoo (free) ──► data/store (CSV) ──► features.py ──► model.py ──► registry (champion)
 NSE CSVs ──────┘   NIFTY, VIX, BankNifty,  timing-safe     walk-forward,     │
 RSS + GDELT ───┘   SPX, NDX, VIX, N225,    features        calibration,      │
                    HSI, DXY, INR, Brent,                    DM skill test     ▼
                    US10Y, Gold                                     paper/live.py ── ledger.sqlite ── api.py (/paper)
 Upstox (optional adapter) ── live LTP, option chain bid/ask ──────────┘   ▲
                                                                           │
                         scheduler.py: 08:40 data → 09:20 signal/entry → MTM → 15:15 exit → EOD → retrain
```

- **Broker independence.** The model is trained only on data every provider has: daily OHLC of NIFTY, VIX and global assets. Brokers are *adapters* for live prices. Upstox is implemented (LTP, option chain, v3 candles, TOTP auto-refresh). Adding Zerodha, Dhan, Fyers or AngelOne means implementing the same four methods.
- **Timing rules.** These are enforced in `features.py` and tested:
  - The 09:20 decision may use NIFTY's open and anything that closed before 09:15 IST.
  - US and commodity data are lagged one session.
  - Tokyo and HK opening gaps are same-day.
- **Model.** An L2 logistic regression plus shallow gradient boosting (depth 3), averaged.
  - Walk-forward refit every 21 sessions.
  - Platt calibration fitted on earlier out-of-sample predictions.
  - Three candidates compete; the challenger is promoted only if its recent out-of-sample log-loss beats the champion's.
- **Honesty gates.**
  - Baselines: always-up, always-majority, gap continuation/fade, yesterday's momentum.
  - Binomial and Diebold-Mariano tests.
  - `has_edge` requires positive log-loss skill with p < 0.05. In live mode, signals without proven skill are flagged `UNPROVEN EDGE`. Paper mode still trades them so they get tested.
- **Paper trading.**
  - Real Upstox bid/ask when a token is present, otherwise VIX-modeled premiums.
  - Full charges: STT 0.15% on option sells, 0.05% on futures sells (Budget 2026, from 1 Apr 2026); NSE transaction 0.03503%/0.00173%; SEBI ₹10/crore; stamp duty; 18% GST; ₹20/order brokerage.
  - Risk controls: risk-per-trade sizing, drawdown kill-switch, no 0DTE, one trade a day.

---

## 4. Results on real data (walk-forward, unseen sessions only)

### 4.1 Prediction quality: target = today's open→close, decided at 09:20

| Metric | Value |
|---|---|
| Test period | 31 Mar 2014 – 2 Nov 2026 (3,094 sessions) |
| Accuracy, all days | 52.97% (always-down baseline: 53.23%) |
| AUC | 0.549 |
| Log-loss skill vs base rate | +0.0025 nats, t = 1.05, **p = 0.146 (not significant)** |
| Confident days (p ≥ 0.56 or ≤ 0.44) | 1,302 days (42%), **hit 57.1%**, +13.2 bps per signal |
| p ≥ 0.58 / 0.60 | 58.4% (844 days) / 58.2% (546 days) |
| Confident **short** calls | 1,140 days, **58.2%**, +15.4 bps |
| Confident **long** calls | 162 days, 49.4%, −2.0 bps |

The close→next-close (swing) target was worse: 53.2% versus 53.5% for always-up, with p = 0.22.

**Top features** (standardised logistic coefficients):
- Hang Seng opening gap (+)
- Gap z-score (− : gaps tend to fade)
- S&P 500 previous day (− : already priced into the gap)
- 20-day range position (+)
- 10-day return (−)

### 4.2 Strategy grid (all 10 pre-declared variants reported)

Setup: 1 lot, ₹1L (options) or ₹5L (futures), kill-switch off, full charges, 2014–2026. Figures are ₹ per trade unless stated.

| Book | Variant | Trades | Win% | Gross | Charges | **Net** | Net total | Max DD |
|---|---|---|---|---|---|---|---|---|
| options | debit spread, stop 45%, thr 0.56 | 1,088 | 44.5 | −16 | 53 | **−69** | −74,680 | −74.9% |
| options | debit spread, stop 45%, thr 0.60 | 546 | 45.8 | 3 | 53 | −51 | −27,669 | −42.0% |
| options | debit spread, no stop, thr 0.56 | 1,063 | 47.3 | 10 | 52 | −42 | −44,181 | −54.6% |
| options | debit spread, no stop, thr 0.60 | 545 | 48.8 | 22 | 53 | −31 | −16,833 | −42.0% |
| options | ITM buy, stop 45%, thr 0.56 | 1,301 | 50.7 | 473 | 54 | 419 | 5,45,252 | −18.1% |
| options | ITM buy, stop 45%, thr 0.60 | 545 | 52.5 | 528 | 51 | 477 | 2,59,829 | −12.2% |
| options | **ITM buy, no stop, thr 0.56** | 1,218 | 53.0 | 575 | 54 | **522** | 6,35,289 | −9.5% |
| options | ITM buy, no stop, thr 0.60 | 502 | 54.2 | 541 | 50 | 490 | 2,46,154 | −13.2% |
| futures | thr 0.56 | 1,302 | 50.5 | 855 | **510** | 344 | 4,48,002 | −19.7% |
| futures | thr 0.60 | 546 | 50.7 | 858 | 450 | 409 | 2,23,093 | −19.2% |

What the grid shows:

- **Debit spreads lose.** Two legs mean double spread and slippage, and the short leg halves the delta, so a ~15-bps edge can't pay for 4 fills.
- **Futures** keep only 40% of gross after the 2026 STT hike.
- **Tight stops hurt.** Intraday noise hits them before the drift arrives.

### 4.3 Default paper config (ITM buy, 60% disaster stop, thr 0.56, ₹1L, 1 lot)

| | Full 2014–2026 | Notes |
|---|---|---|
| Trades | 1,302 | Signals on 42% of days |
| Win rate | 51.9% (95% CI 49.2–54.6%) | Winners are larger than losers |
| Net P&L (1 lot, not compounded) | **+₹5,54,722** | About ₹44k a year on 1 lot |
| Max drawdown | −16.2% of ₹1L | −24.4% if run on ₹50k |
| Months positive | 66% | Median month +1.1%, worst −7.2%, best +12.5% |
| Charges paid | ₹69,780 | |
| Last 12 months | about +9.2% | Only about 2–4 trades a month lately |

### 4.4 Why these numbers are upper bounds

1. **Modeled premiums.** Black-Scholes from India VIX plus a skew, half-spread and 1 pt slippage per side. Real ITM weekly quotes can be wider, and IV moves intraday. *Fix: phase 2 uses Upstox's expired-instrument candles (Upstox Plus) to replay real premiums.*
2. **Weekly options only exist from 2019.** Before that the replay prices a fictional weekly contract.
3. **Selection bias.** I tried 11 variants and the default is the best of them. Expect live results to be worse (deflated Sharpe; Bailey & López de Prado 2014).
4. **Fixed lot size and today's charges** are applied to all history. This is conservative on costs but simplifies sizing.
5. **The edge is mostly short-side and is fading.** The average intraday drift was +0.5 bps in 2025 versus about −10 bps in 2015–2021. If the night effect disappears, most of the edge goes with it.

### 4.5 Real-premium check (added 3 Nov 2026, Upstox Plus expired-options data)

The default strategy was replayed on **real 5-minute prices of the exact expired weekly contracts** for Oct 2024 – Oct 2026. That's 2,713 contracts, with entry and exit on actual option bars, and the model's 09:20 signals from the walk-forward. The same 64 trades were also run through the VIX-modeled replay.

| | Real premiums | Modeled premiums (same days) |
|---|---|---|
| Trades | 64 (62 short / 2 long) | 64 |
| Direction correct (open→close) | 64.1% | 64.1% |
| Trades with net profit | **45.3%** | 59.4% |
| Net P&L, 1 lot | **+₹43,387** (₹678/trade) | +₹88,694 (₹1,386/trade) |
| Months positive (of 22 with trades) | 54.5% | 63.6% |
| Max drawdown on ₹1L | **−23.8%** | — |
| Charges | ₹4,433 | — |
| Per-trade correlation, real vs model | 0.76 | |

**What this means:**

- **The modeled replay overstates profit by about 2×.** Real puts cost about 6 points more than the model at entry, because the market has a put skew and a forward below spot that the model misses. Real premiums also decay faster intraday than a constant-IV model assumes.
- **The edge is positive but thin, and concentrated.** Oct–Nov 2024 produced ₹30k of the ₹43k. From Jan 2025 to Oct 2026 the strategy made only about +₹5k on 1 lot, with a −23.8% drawdown on ₹1L.
- **Sample size is small.** 64 trades gives a 95% CI on the win rate of about ±12 points.
- **Action:** treat modeled results before 2024 as roughly 50% too optimistic. Future model changes will be judged on the real-premium replay (`python -m nifty_engine replay-real`), not the modeled one. Before going live, the edge has to hold up in paper trading and in real-premium replays as more expiries accumulate.

---

## 5. Upgrade roadmap

| Phase | Scope | Acceptance test (walk-forward, unseen data) |
|---|---|---|
| **0 (done, this branch)** | Leak-free features and training, real data, options/futures paper trader, charges, scheduler, Docker, CI, dashboard, daily prediction scorecard | Tests pass. Real-data walk-forward runs end to end. |
| **1: richer daily data** (2–3 weeks) | (a) NSE participant-wise OI and FII/DII cash: download the CSVs into `store/nse/participant_oi/` (loader done). (b) GIFT Nifty 06:30–09:00 move via the broker. (c) Banking heavyweights' ADR/GDR overnight (HDFC Bank, ICICI, Infosys ADRs). (d) Event calendar flags: RBI MPC, US CPI/FOMC, Budget, results days of top-10 weights. | Each feature group must improve recent out-of-sample log-loss with a DM p < 0.10, or it is dropped. |
| **2: intraday model** (3–4 weeks) | `fetch-intraday` (done) pulls 5-min NIFTY and VIX bars since Jan 2022. Build a second model at 09:45 / 10:15 / 14:30 using the first-30-min return (Gao et al.), opening-range position, VIX intraday change and an **option-chain GEX proxy** recorded live each minute (v3's GEX idea, now *recorded* and tested instead of assumed). Replay with real expired-option candles. | Beats the daily model's hit rate on the same days, net of costs. |
| **3: volatility engine** | HAR-RV + VIX forecast of today's range. Use it to (a) size positions, (b) choose option buying in high expected range versus no trade or defined-risk selling in low range, and (c) set stops in σ units. | Lower drawdown at equal return in replay. |
| **4: news and LLM features** (live only) | Headlines anonymised (company names masked, per Glasserman & Lin), scored by an LLM into structured fields (direction, surprise, horizon, India relevance). Stored daily, admitted to training after 250+ days. | Same DM-test gate. Never used as the final decision-maker. |
| **5: meta-labelling** | A second model decides *whether to take* each primary signal, based on regime, VIX and recent hit rate (López de Prado 2018). | Higher net per trade at ≥ 70% of trades kept. |
| **6: go-live readiness** | 3 months of paper trading meeting §6, real-quote fills, broker order adapter, SEBI 2025 retail-algo compliance (broker-registered algo, static IP for order APIs), kill-switches on a separate process. | Written sign-off. |

**Factors that drive NIFTY, and their status here.** Overnight global equities ✓, Asia open ✓, USD/INR ✓, DXY ✓, US 10Y ✓, Brent ✓, gold ✓, India VIX ✓, US VIX ✓, BankNifty relative strength ✓, trend/momentum/mean-reversion ✓, calendar/expiry ✓. Planned: FII/DII flows (P1), participant OI (P1), GIFT Nifty (P1), event calendar (P1), heavyweight ADRs (P1), intraday momentum (P2), dealer gamma/GEX (P2), PCR/OI walls (P2), realised-vol regime (P3), news/geopolitics (forward collection now, P4). Also worth adding: monsoon/IMD forecasts (seasonal), domestic SIP flows (monthly AMFI), China/HK policy events, and earnings season of the top-10 index weights.

---

## 6. The one-month paper test protocol (pre-registered)

**Setup.**
- `python -m nifty_engine run` on Docker or `run_engine.bat`.
- With `UPSTOX_ACCESS_TOKEN` (or the TOTP trio) set, fills use real bid/ask; without it they are VIX-modeled.

**What gets measured.**
- **Every** daily prediction, scored at EOD (`/api/v4/paper/scorecard`), about 22 a month.
- Every paper trade: entry/exit fills, price source, charges, exit reason.

**Pass/fail, decided now so the result can't be rationalised later.**

| Check | Pass | Why |
|---|---|---|
| All-day accuracy | ≥ 50% over the month | Below a coin flip on ~22 days is a red flag, though weak evidence |
| Confident-day accuracy | ≥ 55% *cumulative over 3 months* | One month's 2–6 trades can't establish this |
| Live vs modeled fill slippage | Median ≤ 1.5 pts per leg | Validates the replay's cost model |
| Net P&L | Not a pass criterion for 1 month | ~20 trades gives a ±20-pt CI on win rate |
| Operational | 0 missed sessions, 0 un-exited positions | |

**Then.** Extend to 3 months. Go live only if confident-day accuracy holds above 55% *and* live fills match the model.

---

## 7. Packaging and self-retraining

- **Run anywhere.**
  - `docker compose up -d` runs scheduler, dashboard, data and model persistence under `./store`. On first boot it downloads 19 years of data and trains.
  - Without Docker: `run_engine.bat` (Windows) or `run_engine.sh` (macOS/Linux).
  - Dashboard: `http://localhost:8010/paper`.
- **Self-retraining.**
  - The scheduler refreshes data at 08:40 and after the close.
  - It retrains every Friday, or when the champion is ≥ 21 days old.
  - It re-runs all candidates on the full walk-forward and promotes only on better recent out-of-sample log-loss.
  - Every version is kept in `store/models/<version>/` with metrics and out-of-sample predictions, so you can roll back by editing `champion.json`.
- **CI.** `.github/workflows/engine_ci.yml` runs unit and leakage tests and the full real-data backtest on GitHub runners, and uploads the report as an artifact (daily at 17:00 IST and on every push).
- **Corporate networks.** `truststore` makes Python use the Windows certificate store. This was required on the LTTS network, where TLS is inspected.

---

## 8. Commands

```
python -m nifty_engine update-data     # 13 daily series from 2007
python -m nifty_engine train           # walk-forward + champion/challenger
python -m nifty_engine backtest        # store/reports/BACKTEST_REPORT.md
python -m nifty_engine variants        # the pre-declared strategy grid (VARIANTS.md)
python -m nifty_engine signal          # today's signal (after 09:15)
python -m nifty_engine run             # always-on paper trader + dashboard
python -m nifty_engine fetch-intraday  # Upstox 5-min history for phase 2
python -m unittest discover -s tests   # 9 tests incl. leakage tests
```

---

## Sources

- Gao, Han, Li, Zhou (2018) *Market intraday momentum*, JFE: https://doi.org/10.1016/j.jfineco.2018.05.009
- Baltussen, Da, Lammers, Martens (2021) *Hedging demand and market intraday momentum*, JFE: https://doi.org/10.1016/j.jfineco.2021.04.029
- Gu, Kelly, Xiu (2020) *Empirical asset pricing via machine learning*, RFS: https://doi.org/10.1093/rfs/hhaa009
- Krauss, Do, Huck (2017) EJOR: https://doi.org/10.1016/j.ejor.2016.10.031
- Bailey, Borwein, López de Prado, Zhu, *Probability of Backtest Overfitting*: https://papers.ssrn.com/abstract=2326253 · Bailey & López de Prado (2014) *Deflated Sharpe Ratio*: https://papers.ssrn.com/abstract=2460551
- Kelly, Malamud, Zhou (2024) *Virtue of Complexity*, JF: https://doi.org/10.1111/jofi.13298
- Corsi (2009) HAR-RV: https://ideas.repec.org/a/oup/jfinec/v7y2009i2p174-196.html
- Moskowitz, Ooi, Pedersen (2012) TSMOM: https://doi.org/10.1016/j.jfineco.2011.11.003
- Lopez-Lira & Tang, *Can ChatGPT forecast stock price movements?*: https://arxiv.org/abs/2304.07619
- Glasserman & Lin (2023) look-ahead/distraction bias: https://arxiv.org/abs/2309.17322
- Rahimikia et al. (2025) TSFMs on returns: https://arxiv.org/abs/2511.18578 · Brini (2026) TSFMs on RV: https://arxiv.org/abs/2607.05291
- *Profit Mirage* (2025): https://arxiv.org/abs/2510.07920 · FINSABER (KDD 2026): https://arxiv.org/abs/2505.07078 · *The Alpha Illusion* (2026): https://arxiv.org/abs/2605.16895
- Zhang, Zohren, Roberts (2019) DeepLOB: https://arxiv.org/abs/1808.03668 · Araci (2019) FinBERT: https://arxiv.org/abs/1908.10063
- Acharya, Anshuman, Kiran Kumar (2016) FII flows in India: https://w4.stern.nyu.edu/sternfin/vacharya/public_html/pdfs/Foreign%20Fund%20Flows%20Oct%2031%202016.pdf
- Sarkar, Deb, Panda (2025) GIFT Nifty price discovery, JRFM: https://pure.jgu.edu.in/id/eprint/10305/ · Sundararajan & Balasubramanian (2023): https://doi.org/10.1108/MF-04-2022-0191
- Shaikh & Padhi (2014) India VIX: https://doi.org/10.1007/s10644-014-9149-z
- Kapoor & Narayanan (2023) leakage in ML science: https://pmc.ncbi.nlm.nih.gov/articles/PMC10499856/ · Jafar et al. (2023) example of an inflated claim: https://www.mdpi.com/1911-8074/16/10/423
- Chen (2026) walk-forward SPY: https://arxiv.org/abs/2608.26106 · Kotekar et al. (2025) NIFTY + news: https://doi.org/10.1109/ACCESS.2025.3644013 · Dasgupta et al.: https://arxiv.org/abs/2412.06794
- SEBI F&O studies: https://www.sebi.gov.in/media-and-notifications/press-releases/sep-2024/updated-sebi-study-reveals-93-of-individual-traders-incurred-losses-in-equity-fando-between-fy22-and-fy24-aggregate-losses-exceed-1-8-lakh-crores-over-three-years_86906.html · FY25 (91% lost): https://www.business-standard.com/markets/capital-market-news/sebi-sees-dip-in-derivatives-turnover-91-of-retail-traders-lose-money-in-fy25-125070800660_1.html
- STT from 1 Apr 2026: https://www.icicidirect.com/ilearn/futures-and-options/articles/stt-changes-in-budget-2026-what-f-o-traders-should-know
- NIFTY lot size 65: https://hdfcsky.com/news/nse-revises-market-lot-sizes-for-major-index-derivatives-effective-january-2026 · Tuesday expiry: https://www.business-standard.com/markets/news/nse-bids-adieu-to-thursday-expiry-as-dates-swap-come-into-effect-explained-125082800635_1.html
- Upstox APIs (token, TOTP, v3 candles, expired instruments): https://upstox.com/developer/api-documentation/get-token · https://upstox.com/developer/api-documentation/v3/get-historical-candle-data · https://upstox.com/developer/api-documentation/get-expired-historical-candle-data
- NSE terms of use (no scraping): https://www.nseindia.com/static/nse-terms-of-use
