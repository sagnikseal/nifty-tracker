"""Historical paper-trading replay on walk-forward (out-of-sample) predictions."""

from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np
import pandas as pd

from ..config import EngineConfig
from .sim import decide, simulate_future_day, simulate_option_day


def replay(F: pd.DataFrame, oos: pd.DataFrame, cfg: EngineConfig, book: str = "options",
           start: Optional[str] = None, end: Optional[str] = None,
           capital: Optional[float] = None) -> Dict:
    capital = float(capital or cfg.capital)
    df = oos.join(F[[c for c in F.columns if c.startswith("px_")] + ["rv_w"]], how="left")
    df["px_vix_today"] = F["px_vix_prev"].shift(-1).reindex(df.index) if "px_vix_prev" in F else np.nan
    if start:
        df = df[df.index >= pd.Timestamp(start)]
    if end:
        df = df[df.index <= pd.Timestamp(end)]
    equity, peak, killed = capital, capital, False
    trades, curve, skipped = [], [], {"no_signal": 0, "insufficient_capital": 0, "killed": 0, "no_data": 0}
    for d, r in df.iterrows():
        direction = decide(r["p"], cfg)
        if killed:
            skipped["killed"] += 1
        elif direction == 0:
            skipped["no_signal"] += 1
        elif any(pd.isna(r.get(k)) for k in ("px_open", "px_high", "px_low", "px_close")):
            skipped["no_data"] += 1
        else:
            px = {"open": r["px_open"], "high": r["px_high"], "low": r["px_low"], "close": r["px_close"]}
            if book == "options":
                v0 = r.get("px_vix_prev")
                v0 = 15.0 if pd.isna(v0) else float(v0)
                v1 = r.get("px_vix_today")
                v1 = v0 if pd.isna(v1) else float(v1)
                t = simulate_option_day(d.date(), direction, px, v0, v1, equity, cfg)
            else:
                atr = float(r["rv_w"] * r["px_open"] * 2.0) if not pd.isna(r.get("rv_w")) else 150.0
                t = simulate_future_day(d.date(), direction, px, atr, equity, cfg)
            if t and "skipped" in t:
                skipped["insufficient_capital"] += 1
            elif t:
                t["p_up"] = round(float(r["p"]), 4)
                t["correct_direction"] = int(np.sign(r["px_close"] - r["px_open"]) == direction)
                equity += t["net_pnl"]
                t["equity_after"] = round(equity, 2)
                trades.append(t)
        peak = max(peak, equity)
        if (peak - equity) / peak * 100 >= cfg.max_drawdown_kill_pct:
            killed = True
        curve.append({"date": d, "equity": equity})
    T = pd.DataFrame(trades)
    E = pd.DataFrame(curve).set_index("date") if curve else pd.DataFrame()
    return {"trades": T, "equity": E, "summary": summarize(T, E, capital, skipped), "book": book}


def summarize(T: pd.DataFrame, E: pd.DataFrame, capital: float, skipped: Dict) -> Dict:
    if E.empty:
        return {"trades": 0}
    eq = E["equity"]
    dd = (eq / eq.cummax() - 1).min()
    rets = eq.pct_change().dropna()
    s = {
        "start_capital": capital, "end_equity": round(float(eq.iloc[-1]), 2),
        "net_pnl": round(float(eq.iloc[-1] - capital), 2),
        "return_pct": round(float((eq.iloc[-1] / capital - 1) * 100), 2),
        "max_drawdown_pct": round(float(dd * 100), 2),
        "sessions": int(len(eq)), "trades": int(len(T)), "skipped": skipped,
        "sharpe_daily_annualised": round(float(rets.mean() / rets.std() * math.sqrt(252)), 2) if rets.std() > 0 else 0.0,
    }
    if len(T):
        wins, losses = T[T["net_pnl"] > 0], T[T["net_pnl"] <= 0]
        n = len(T)
        hit = len(wins) / n
        s.update({
            "win_rate": round(hit, 4),
            "win_rate_95ci": [round(max(0, hit - 1.96 * math.sqrt(hit * (1 - hit) / n)), 3),
                              round(min(1, hit + 1.96 * math.sqrt(hit * (1 - hit) / n)), 3)],
            "direction_accuracy": round(float(T["correct_direction"].mean()), 4),
            "avg_win": round(float(wins["net_pnl"].mean()), 2) if len(wins) else 0.0,
            "avg_loss": round(float(losses["net_pnl"].mean()), 2) if len(losses) else 0.0,
            "profit_factor": round(float(wins["net_pnl"].sum() / -losses["net_pnl"].sum()), 2)
            if len(losses) and losses["net_pnl"].sum() < 0 else None,
            "total_charges": round(float(T["charges"].sum()), 2),
            "charges_pct_of_gross_abs": round(float(T["charges"].sum() / max(T["gross_pnl"].abs().sum(), 1) * 100), 1),
            "exit_reasons": T["exit_reason"].value_counts().to_dict(),
            "avg_lots": round(float(T["lots"].mean()), 2),
        })
    if len(E) > 25:
        m = E["equity"].resample("ME").last()
        prev = m.shift(1).fillna(capital)
        mret = (m / prev - 1) * 100
        s["monthly_return_pct"] = {k.strftime("%Y-%m"): round(float(v), 2) for k, v in mret.items()}
        s["months_positive_pct"] = round(float((mret > 0).mean() * 100), 1)
        s["median_month_pct"] = round(float(mret.median()), 2)
        s["worst_month_pct"] = round(float(mret.min()), 2)
        s["best_month_pct"] = round(float(mret.max()), 2)
    return s
