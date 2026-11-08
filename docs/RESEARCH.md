# Research harness: how every idea is tested before it is used

The engine treats every new feature, model and strategy as a hypothesis. Each one gets a
**pre-registered** test: the pass rule is written down before the test runs. Every result is
reported, including failures. Running the app, the dashboard and the daily routine are covered in
[`GUIDE.md`](GUIDE.md); this page covers the research side.

## Commands

```bash
python -m nifty_engine research                   # every experiment -> store/reports/RESEARCH_REPORT.md
python -m nifty_engine research --only S4,S8      # just some of them
python -m nifty_engine research --quick           # logistic model only (fast)
python -m nifty_engine intraday-ablation          # which data/feature groups help the 5-minute model
python -m nifty_engine intraday-zoo               # every movement-model member + the vote (run on the Mac)
python -m nifty_engine atm-replay                 # the ATM rules on real expired-option prices
python -m nifty_engine news-impact                # how NIFTY reacted to the LLM's reading of headlines
python -m nifty_engine variants                   # daily options book: stop / threshold grid
python -m nifty_engine zoo --fresh --promote      # daily model zoo (best-3 deployed if better)
```

## The experiments (`config/experiments.json`)

The thresholds were fixed before any test ran. If you change a rule after seeing results, that
rule no longer tells you anything.

| ID | What is tested | Pass rule (summary) |
|---|---|---|
| P0 | base daily model | reported (benchmark) |
| P1-P6 | asia_extra, adr, smc, structure, regime, events feature groups, each vs the base daily model on identical days | lower log-loss, Diebold-Mariano p < 0.10, ≥ 250 days |
| P7-P9 | overnight_futures, options_hist, news (short history), stacked on the base probability | p < 0.10 on ≥ 120 out-of-sample days |
| S1-S8 | the strategy books, on real option prices where downloaded, VIX-modelled otherwise | net/trade > 0, t ≥ 2, deflated Sharpe ≥ 0.95 across all trials, ≥ 50% real prices |
| S7b, S8b | turn-of-month on futures since 2007; SMC setups in R-multiples on the index | reported |
| M1 | meta-labelling filter across all books | better net/trade while keeping ≥ 70% of trades |
| F1, F2 | the LLM pre-open brief vs the model; recorded option-chain/depth features | need 60 recorded sessions |

The movement-model ablation and the ATM replay have their own pre-registered sets. They are defined in
`intraday_model.ABLATION_SETS` and `paper/atm_replay.GRID` / `GRID_DAILY`, and each report states its rule.

**Promoting a daily feature group.** Only promote a group if it passes on a full run (not `--quick`).

1. Add it to `"feature_groups"` in `config/engine.json` (e.g. `"base,adr"`).
2. Run `python -m nifty_engine train`.

The live signal always uses exactly the groups its model was trained with, so changing the config can't silently mismatch the model. Promote one group at a time and re-run research after each. Every group you test is another trial: with 6 groups, a p-value just under 0.10 can be luck.

## The research strategy books (`nifty_engine/strategies/books.py`)

The books are off in the live app by default. To paper-trade them alongside the main books, list them in
`"strategy_books"` in `config/engine.json`. Rules are fixed in code before testing.

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

VRP = India VIX's implied daily volatility minus the HAR-RV forecast. When it is positive, options are
pricing more movement than the volatility model expects.

## Data the tests need

| Data | Command | Notes |
|---|---|---|
| Daily + hourly global markets | `update-data` | free (Yahoo); daily since 2007, hourly ~2 years |
| 1/5-minute NIFTY, VIX | `fetch-intraday` | Upstox; minute data starts Jan 2022 |
| Bank Nifty, heavyweights, NIFTY futures | `fetch-history` | Upstox (Plus for expired futures, ~2 years back) |
| Real expired-option prices | `fetch-expired-options --mode full --since 2024-10-01` | Upstox Plus; hours, resumable |
| Historical option-chain features | `build-option-features` | from the expired-option candles |
| ~3 months of news | `news-backfill` | GDELT + local-LLM scoring |

## Limits to keep in mind

- **Modelled vs real prices.** Replays fall back to VIX-modelled premiums wherever real expired prices weren't downloaded. Every trade records its `price_source`, and a strategy only passes if at least half its trades use real prices.
- **Strike-selection leak (found and fixed).** Downloaded strikes are chosen from each day's opening level, so a week's file "knows" where NIFTY went later. Option-chain features therefore use only a strike window fixed at that day's 09:15/09:20 level. Any new feature built from "which contracts exist in the data" needs the same check.
- **0DTE history.** Upstox may not serve expiry-day bars for expired weeks; S5 shows INSUFFICIENT_DATA until the recorder has collected expiry days.
- **Real money** would need a full-access token (the Analytics Token is read-only), SEBI's 2025 retail-algo framework (broker-registered algo, static IP for order APIs) and separate kill-switches. None of that is built, on purpose.
