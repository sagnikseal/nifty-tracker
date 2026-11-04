"""Replay the options strategy on REAL historical option prices (5-minute bars of the exact
expired contracts), and compare with the VIX-modeled replay on the same days."""

from __future__ import annotations

import datetime as dt
import math
from typing import Dict, Optional

import numpy as np
import pandas as pd

from ..calendar import IST
from ..config import DATA_DIR, EngineConfig
from ..data import store
from ..data.expired_options import expired_expiries, load_expiry, sessions_per_expiry
from ..data.providers import UpstoxProvider
from ..options import half_spread
from .charges import order_charges
from .sim import decide, rates_from, simulate_option_day

REPORTS = DATA_DIR / "reports"


def _bars_for_day(opt: pd.DataFrame, d: dt.date, strike: float, typ: str) -> pd.DataFrame:
    m = (opt["ts"].dt.date == d) & (opt["strike"] == strike) & (opt["type"] == typ)
    return opt[m].sort_values("ts")


def replay_real(cfg: EngineConfig, start: str = "2024-10-01", oos: Optional[pd.DataFrame] = None) -> Dict:
    if oos is None:
        oos = pd.read_csv(REPORTS / "oos_predictions.csv", index_col=0, parse_dates=True)
    oos = oos[oos.index >= pd.Timestamp(start)]
    nifty = store.load("nifty")
    vix = store.load("indiavix")["close"]
    i5 = DATA_DIR / "intraday" / "nifty_5m.csv"
    n5 = pd.read_csv(i5, index_col=0) if i5.exists() else pd.DataFrame()
    if len(n5):
        n5.index = pd.to_datetime(n5.index, utc=True).tz_convert(IST)
    exps = expired_expiries(UpstoxProvider())
    plan = sessions_per_expiry(exps, [d.date() for d in oos.index])
    day2exp = {d: e for e, ds in plan.items() for d in ds}
    rates = rates_from(cfg)
    cache: Dict[dt.date, pd.DataFrame] = {}
    rows = []
    t_entry, t_exit = dt.time(9, 20), dt.time(15, 15)
    for ts, r in oos.iterrows():
        d = ts.date()
        direction = decide(r["p"], cfg)
        if direction == 0 or d not in day2exp:
            continue
        exp = day2exp[d]
        if exp not in cache:
            cache[exp] = load_expiry(exp)
        opt = cache[exp]
        if opt.empty:
            continue
        S = None
        if len(n5):
            b = n5[(n5.index.date == d) & (n5.index.time == t_entry)]
            S = float(b["open"].iloc[0]) if len(b) else None
        if S is None and pd.Timestamp(d) in nifty.index:
            S = float(nifty.loc[pd.Timestamp(d), "open"])
        if S is None:
            continue
        atm = round(S / 50) * 50
        typ = "CE" if direction > 0 else "PE"
        strike = atm - 50 if direction > 0 else atm + 50
        bars = _bars_for_day(opt, d, strike, typ)
        bars = bars[(bars["ts"].dt.time >= t_entry) & (bars["ts"].dt.time <= t_exit)]
        if len(bars) < 10:
            continue
        mid0 = float(bars["open"].iloc[0])
        entry = mid0 + half_spread(mid0) + cfg.option_slippage_pts
        stop_px = entry * (1 - cfg.option_stop_pct / 100)
        tgt_px = entry * (1 + cfg.option_target_pct / 100)
        reason, exit_mid = "time", float(bars["open"].iloc[-1])
        for _, b in bars.iloc[:-1].iterrows():
            if b["low"] <= stop_px:
                reason, exit_mid = "stop", min(stop_px, float(b["open"]))
                break
            if b["high"] >= tgt_px:
                reason, exit_mid = "target", max(tgt_px, float(b["open"]))
                break
        exit_px = max(0.05, exit_mid - half_spread(exit_mid) - cfg.option_slippage_pts)
        q = cfg.lot_size
        ch = (order_charges("option", "BUY", entry * q, rates)["total"] +
              order_charges("option", "SELL", exit_px * q, rates)["total"])
        gross = (exit_px - entry) * q
        # Modeled replay for the same day (for comparison)
        px = {k: float(nifty.loc[pd.Timestamp(d), k]) for k in ("open", "high", "low", "close")}
        v0 = float(vix[vix.index < pd.Timestamp(d)].iloc[-1])
        v1 = float(vix.loc[pd.Timestamp(d)]) if pd.Timestamp(d) in vix.index else v0
        c2 = EngineConfig(**{**cfg.to_dict()})
        c2.option_style, c2.max_lots = "itm_buy", 1
        m = simulate_option_day(d, direction, px, v0, v1, 10_000_000, c2) or {}
        rows.append({"date": d.isoformat(), "direction": direction, "p_up": round(float(r["p"]), 4),
                     "expiry": exp.isoformat(), "strike": strike, "type": typ, "spot_0920": round(S, 2),
                     "entry": round(entry, 2), "exit": round(exit_px, 2), "exit_reason": reason,
                     "gross_pnl": round(gross, 2), "charges": round(ch, 2), "net_pnl": round(gross - ch, 2),
                     "model_entry": m.get("entry_value"), "model_net_pnl": m.get("net_pnl"),
                     "model_exit_reason": m.get("exit_reason"),
                     "correct_direction": int(np.sign(px["close"] - px["open"]) == direction)})
    T = pd.DataFrame(rows)
    out = {"trades": T, "summary": _summ(T)}
    REPORTS.mkdir(parents=True, exist_ok=True)
    if len(T):
        T.to_csv(REPORTS / "real_premium_trades.csv", index=False)
    return out


def _summ(T: pd.DataFrame) -> Dict:
    if T.empty:
        return {"trades": 0}
    n = len(T)
    w = (T["net_pnl"] > 0).mean()
    T = T.copy()
    T["month"] = T["date"].str[:7]
    mon = T.groupby("month")[["net_pnl", "model_net_pnl"]].sum().round(0)
    return {
        "period": [T["date"].min(), T["date"].max()], "trades": n,
        "win_rate": round(float(w), 4),
        "win_rate_95ci": [round(w - 1.96 * math.sqrt(w * (1 - w) / n), 3), round(w + 1.96 * math.sqrt(w * (1 - w) / n), 3)],
        "direction_accuracy": round(float(T["correct_direction"].mean()), 4),
        "real_net_total": round(float(T["net_pnl"].sum()), 0),
        "model_net_total": round(float(T["model_net_pnl"].sum()), 0),
        "real_net_per_trade": round(float(T["net_pnl"].mean()), 1),
        "model_net_per_trade": round(float(T["model_net_pnl"].mean()), 1),
        "real_vs_model_corr": round(float(T["net_pnl"].corr(T["model_net_pnl"])), 3),
        "avg_entry_real_minus_model": round(float((T["entry"] - T["model_entry"]).mean()), 2),
        "charges_total": round(float(T["charges"].sum()), 0),
        "by_direction": {("LONG" if k > 0 else "SHORT"): {"trades": int(len(g)), "real_net": round(float(g["net_pnl"].sum()), 0),
                                                          "model_net": round(float(g["model_net_pnl"].sum()), 0),
                                                          "real_win": round(float((g["net_pnl"] > 0).mean()), 3)}
                         for k, g in T.groupby("direction")},
        "exit_reasons_real": T["exit_reason"].value_counts().to_dict(),
        "months_positive_real_pct": round(float((mon["net_pnl"] > 0).mean() * 100), 1),
        "monthly": mon.to_dict(orient="index"),
    }
