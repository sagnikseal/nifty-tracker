"""Human-readable reports (Markdown) for backtests and the live paper month."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Dict

import pandas as pd


def _pct(x):
    return "n/a" if x is None else f"{x * 100:.2f}%"


def write_backtest_markdown(out: Dict, path: Path) -> None:
    m = out["prediction_metrics"]
    L = [f"# NIFTY v4 engine: walk-forward backtest report",
         f"_Generated {dt.datetime.now():%Y-%m-%d %H:%M}. Target: `{out['config']['target']}` "
         f"(oc = today's open->close). Model: `{out['model_spec']['name']}`._", ""]
    if m.get("n_days"):
        L += ["## 1. Prediction quality (out-of-sample, walk-forward)", "",
              f"* Period: {m['period'][0]} to {m['period'][1]} ({m['n_days']} sessions)",
              f"* Directional accuracy: **{_pct(m['accuracy'])}** (up-day rate {_pct(m['up_rate'])})",
              f"* AUC: {m['auc']:.3f} | Brier: {m['brier']:.4f} | Log-loss {m['logloss']:.4f} "
              f"vs base-rate {m['logloss_baserate']:.4f} (skill {m['skill_vs_baserate']:+.4f})",
              f"* p-value vs coin-flip: {m['p_value_vs_coinflip']:.4g}; vs always-majority: {m['p_value_vs_majority']:.4g}",
              "", "| Baseline | Accuracy |", "|---|---|"]
        L += [f"| {k} | {_pct(v)} |" for k, v in m["baselines"].items()]
        L += ["", "### Accuracy when only trading confident signals", "",
              "| Threshold | Coverage | Trades | Hit rate | Avg signed open->close (bps) |", "|---|---|---|---|---|"]
        L += [f"| {r['threshold']:.2f} | {_pct(r['coverage'])} | {r['n']} | {_pct(r['hit_rate'])} | {r['avg_ret_bps']:.1f} |"
              for r in m["by_threshold"]]
        L += ["", "### By year", "", "| Year | Accuracy | Days |", "|---|---|---|"]
        L += [f"| {y} | {_pct(v['accuracy'])} | {v['n']} |" for y, v in m["by_year"].items()]
        L += ["", "### Calibration (predicted vs realised up-probability, quintiles)", "",
              "| Predicted | Actual |", "|---|---|"]
        L += [f"| {_pct(c['pred'])} | {_pct(c['actual'])} |" for c in m["calibration"]]
    L += ["", "## 2. Paper-trading replay (modeled option premiums, full NSE charges)", "",
          "Option premiums are Black-Scholes priced from India VIX with a skew, plus half-spread and "
          f"{out['config']['option_slippage_pts']} pt slippage per leg per side. Stops and targets are checked "
          "against the day's high/low; when both are hit on the same day, the stop is assumed first.", "",
          "| Book / capital | Window | Trades | Win rate (95% CI) | Net P&L | Return | Max DD | Charges | Months +ve |",
          "|---|---|---|---|---|---|---|---|---|"]
    for key, v in out["replays"].items():
        for win in ("full", "last_month"):
            s = v[win]
            if not s.get("trades"):
                L.append(f"| {key} | {win} | 0 | - | - | - | - | - | - |")
                continue
            ci = s.get("win_rate_95ci", [None, None])
            L.append(f"| {key} | {win} | {s['trades']} | {_pct(s.get('win_rate'))} ({ci[0]}-{ci[1]}) | "
                     f"Rs {s['net_pnl']:,.0f} | {s['return_pct']:.1f}% | {s['max_drawdown_pct']:.1f}% | "
                     f"Rs {s.get('total_charges', 0):,.0f} | {s.get('months_positive_pct', '-')}{'%' if s.get('months_positive_pct') is not None else ''} |")
    L += ["", "## 3. How to read this", "",
          "* A single month is ~20 trades: the 95% interval on its win rate is about +/-20 points, so "
          "one month cannot prove or disprove the engine. Use the 'Months +ve' column and the yearly table.",
          "* `p-value vs coin-flip` below 0.01 *and* positive log-loss skill are the minimum bar for a real edge.",
          "* The live paper trader uses broker bid/ask when available; expect live results to be worse than "
          "the replay, not better."]
    path.write_text("\n".join(L), encoding="utf-8")


def live_month_report(trades: pd.DataFrame, equity: pd.DataFrame, capital: float) -> str:
    if trades.empty:
        return "# Live paper trading\n\nNo trades yet."
    closed = trades[trades["status"] == "CLOSED"]
    net = closed["net_pnl"].sum()
    wins = (closed["net_pnl"] > 0).mean() if len(closed) else 0
    L = ["# Live paper trading report", "",
         f"* Closed trades: {len(closed)} | Win rate: {wins * 100:.1f}% | Net P&L: Rs {net:,.0f} "
         f"({net / capital * 100:.2f}% of Rs {capital:,.0f})",
         f"* Charges paid: Rs {closed['charges'].sum():,.0f}", "",
         "| Date | Book | Dir | Style | Lots | Entry | Exit | Net P&L | Exit | p(up) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in closed.iterrows():
        L.append(f"| {r['date']} | {r['book']} | {'LONG' if r['direction'] > 0 else 'SHORT'} | {r['style']} | {r['lots']} | "
                 f"{r['entry_value']:.2f} | {r['exit_value']:.2f} | {r['net_pnl']:,.0f} | {r['exit_reason']} | {r['p_up']:.3f} |")
    return "\n".join(L)
