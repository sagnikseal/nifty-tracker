"""Intraday short iron condor, replayed on REAL 5-minute option prices.

Why this strategy exists in the engine: daily *direction* is ~53% predictable, but whether
NIFTY *stays inside a band* is far more predictable. India VIX overstates the realised move
on most days (variance risk premium), and a range model lifts that further. On 3,079 walk-forward days,
NIFTY stayed within 1.25 x VIX-implied daily sigma of the open all day on 86% of sessions, and
90%+ when the range model was confident.

Trade (1 lot, defined risk): at 09:20 sell the CE ~k*sigma above and the PE ~k*sigma below the
spot, buy wings `wing` points further out, exit at 15:15 or when the loss reaches
`stop_mult` x credit. Nearest weekly expiry with >= 1 day left (no 0DTE).
"""

from __future__ import annotations

import datetime as dt
import math
from typing import Dict, Optional

import numpy as np
import pandas as pd

from ..calendar import IST
from ..config import DATA_DIR
from ..data.expired_options import expired_expiries, load_expiry, sessions_per_expiry
from ..data.providers import UpstoxProvider
from ..options import half_spread
from .charges import ChargeRates, order_charges

STEP = 50


def _leg_bars(opt, d, strike, typ, t0, t1):
    m = (opt["ts"].dt.date == d) & (opt["strike"] == strike) & (opt["type"] == typ)
    b = opt[m].sort_values("ts")
    return b[(b["ts"].dt.time >= t0) & (b["ts"].dt.time <= t1)].set_index("ts")


def replay_condor(F: pd.DataFrame, p_inside: Optional[pd.Series] = None, start: str = "2024-10-01",
                  k: float = 1.25, wing: int = 200, stop_mult: float = 1.5, p_threshold: float = 0.0,
                  lot: int = 65, slippage: float = 0.5, brokerage: float = 20.0,
                  expiries=None) -> Dict:
    i5 = DATA_DIR / "intraday" / "nifty_5m.csv"
    n5 = pd.read_csv(i5, index_col=0)
    n5.index = pd.to_datetime(n5.index, utc=True).tz_convert(IST)
    days = [d for d in sorted(set(n5.index.date)) if d >= dt.date.fromisoformat(start)]
    exps = expiries if expiries is not None else expired_expiries(UpstoxProvider())
    plan = sessions_per_expiry(exps, days)
    day2exp = {d: e for e, ds in plan.items() for d in ds}
    rates = ChargeRates(brokerage_per_order=brokerage)
    cache: Dict = {}
    rows, skipped = [], {"filter": 0, "no_data": 0, "no_vix": 0}
    t0, t1 = dt.time(9, 20), dt.time(15, 15)
    for d in days:
        ts = pd.Timestamp(d)
        if ts not in F.index or d not in day2exp:
            skipped["no_data"] += 1
            continue
        if p_inside is not None and p_threshold > 0:
            pv = p_inside.get(ts, np.nan)
            if pd.isna(pv) or pv < p_threshold:
                skipped["filter"] += 1
                continue
        vix = F.loc[ts, "px_vix_prev"]
        if pd.isna(vix):
            skipped["no_vix"] += 1
            continue
        b0 = n5[(n5.index.date == d) & (n5.index.time == t0)]
        if b0.empty:
            skipped["no_data"] += 1
            continue
        S = float(b0["open"].iloc[0])
        sig = float(vix) / 100 / math.sqrt(252)
        dist = k * sig * S
        sc = math.ceil((S + dist) / STEP) * STEP
        sp = math.floor((S - dist) / STEP) * STEP
        exp = day2exp[d]
        if exp not in cache:
            cache[exp] = load_expiry(exp)
        opt = cache[exp]
        if opt.empty:
            skipped["no_data"] += 1
            continue
        legs = {"sc": (sc, "CE", -1), "lc": (sc + wing, "CE", +1), "sp": (sp, "PE", -1), "lp": (sp - wing, "PE", +1)}
        bars = {n: _leg_bars(opt, d, s, t, t0, t1) for n, (s, t, _) in legs.items()}
        if any(len(b) < 10 for b in bars.values()):
            skipped["no_data"] += 1
            continue
        idx = bars["sc"].index
        for n in bars:
            bars[n] = bars[n].reindex(idx).ffill()
        entry = {}
        for n, (s, t, sign) in legs.items():
            px = float(bars[n]["open"].iloc[0])
            cost = half_spread(px) + slippage
            entry[n] = px + cost if sign > 0 else max(0.05, px - cost)
        credit = entry["sc"] + entry["sp"] - entry["lc"] - entry["lp"]
        if credit <= 1.0:
            skipped["no_data"] += 1
            continue
        stop_cost = credit * (1 + stop_mult)
        reason, exit_mid = "time", None
        # Mark on synchronous 5-min closes (summing intrabar highs of a call AND a put that never
        # co-occur would be wrong); stop exits pay an extra 2 pts per leg of adverse slippage.
        for t in idx[1:-1]:
            mark = (bars["sc"].at[t, "close"] + bars["sp"].at[t, "close"]
                    - bars["lc"].at[t, "close"] - bars["lp"].at[t, "close"])
            if mark >= stop_cost:
                reason = "stop"
                exit_mid = {n: float(bars[n].at[t, "close"]) + (2.0 if legs[n][2] < 0 else -2.0) for n in legs}
                break
        if exit_mid is None:
            exit_mid = {n: float(bars[n]["open"].iloc[-1]) for n in legs}
        exit_px = {}
        for n, (s, t, sign) in legs.items():
            px = exit_mid[n]
            cost = half_spread(px) + slippage
            exit_px[n] = px + cost if sign < 0 else max(0.05, px - cost)   # buy back shorts, sell longs
        debit = exit_px["sc"] + exit_px["sp"] - exit_px["lc"] - exit_px["lp"]
        ch = 0.0
        for n, (s, t, sign) in legs.items():
            ch += order_charges("option", "SELL" if sign < 0 else "BUY", entry[n] * lot, rates)["total"]
            ch += order_charges("option", "BUY" if sign < 0 else "SELL", exit_px[n] * lot, rates)["total"]
        gross = (credit - debit) * lot
        S_exit = float(n5[(n5.index.date == d) & (n5.index.time == dt.time(15, 10))]["close"].iloc[0]) \
            if len(n5[(n5.index.date == d) & (n5.index.time == dt.time(15, 10))]) else np.nan
        rows.append({"date": d.isoformat(), "expiry": exp.isoformat(), "dte": (exp - d).days, "spot": round(S, 2),
                     "vix": float(vix), "short_ce": sc, "short_pe": sp, "wing": wing,
                     "credit": round(credit, 2), "debit": round(debit, 2), "exit_reason": reason,
                     "gross_pnl": round(gross, 2), "charges": round(ch, 2), "net_pnl": round(gross - ch, 2),
                     "max_loss": round((wing - credit) * lot, 0), "spot_exit": S_exit,
                     "p_inside": float(p_inside.get(ts, np.nan)) if p_inside is not None else np.nan})
    T = pd.DataFrame(rows)
    return {"trades": T, "summary": summarize_condor(T), "skipped": skipped}


def summarize_condor(T: pd.DataFrame, capital: float = 100_000) -> Dict:
    if T.empty:
        return {"trades": 0}
    n = len(T)
    w = float((T["net_pnl"] > 0).mean())
    eq = capital + T["net_pnl"].cumsum()
    m = T.assign(m=T["date"].str[:7]).groupby("m")["net_pnl"].sum()
    wins, losses = T[T["net_pnl"] > 0], T[T["net_pnl"] <= 0]
    return {"trades": n, "period": [T["date"].min(), T["date"].max()],
            "win_rate": round(w, 4),
            "win_rate_95ci": [round(w - 1.96 * math.sqrt(w * (1 - w) / n), 3), round(w + 1.96 * math.sqrt(w * (1 - w) / n), 3)],
            "net_total": round(float(T["net_pnl"].sum()), 0), "net_per_trade": round(float(T["net_pnl"].mean()), 1),
            "avg_win": round(float(wins["net_pnl"].mean()), 0) if len(wins) else 0,
            "avg_loss": round(float(losses["net_pnl"].mean()), 0) if len(losses) else 0,
            "worst_day": round(float(T["net_pnl"].min()), 0),
            "profit_factor": round(float(wins["net_pnl"].sum() / -losses["net_pnl"].sum()), 2) if len(losses) and losses["net_pnl"].sum() < 0 else None,
            "max_drawdown_pct_on_1L": round(float((eq / eq.cummax() - 1).min() * 100), 1),
            "charges_total": round(float(T["charges"].sum()), 0),
            "stops": int((T["exit_reason"] == "stop").sum()),
            "months_positive_pct": round(float((m > 0).mean() * 100), 1), "months": int(len(m)),
            "avg_credit_pts": round(float(T["credit"].mean()), 1)}


def replay_weekly_condor(F: pd.DataFrame, start: str = "2024-10-01", k: float = 1.25, wing: int = 200,
                         stop_mult: float = 1.5, lot: int = 65, slippage: float = 0.5, brokerage: float = 20.0,
                         expiries=None, skip_days: int = 0) -> Dict:
    """Positional version: enter 09:20 on the first session of each weekly cycle (+skip_days),
    hold to 15:15 of the last session before expiry (never into expiry day), stop at stop_mult x credit."""
    i5 = DATA_DIR / "intraday" / "nifty_5m.csv"
    n5 = pd.read_csv(i5, index_col=0)
    n5.index = pd.to_datetime(n5.index, utc=True).tz_convert(IST)
    days = [d for d in sorted(set(n5.index.date)) if d >= dt.date.fromisoformat(start)]
    exps = expiries if expiries is not None else expired_expiries(UpstoxProvider())
    plan = sessions_per_expiry(exps, days)
    rates = ChargeRates(brokerage_per_order=brokerage)
    rows = []
    for exp, sess in sorted(plan.items()):
        sess = sorted(sess)[skip_days:]
        if len(sess) < 1:
            continue
        d0, d1 = sess[0], sess[-1]
        ts0 = pd.Timestamp(d0)
        if ts0 not in F.index or pd.isna(F.loc[ts0, "px_vix_prev"]):
            continue
        b0 = n5[(n5.index.date == d0) & (n5.index.time == dt.time(9, 20))]
        if b0.empty:
            continue
        S = float(b0["open"].iloc[0])
        sig = float(F.loc[ts0, "px_vix_prev"]) / 100 / math.sqrt(252) * math.sqrt(len(sess))
        sc = math.ceil((S * (1 + k * sig)) / STEP) * STEP
        sp = math.floor((S * (1 - k * sig)) / STEP) * STEP
        opt = load_expiry(exp)
        if opt.empty:
            continue
        legs = {"sc": (sc, "CE", -1), "lc": (sc + wing, "CE", +1), "sp": (sp, "PE", -1), "lp": (sp - wing, "PE", +1)}
        bars = {}
        for n, (s_, t_, _) in legs.items():
            m = (opt["strike"] == s_) & (opt["type"] == t_)
            b = opt[m].set_index("ts").sort_index()
            b = b[(b.index.date >= d0) & (b.index.date <= d1)]
            b = b[(b.index.time >= dt.time(9, 20)) & (b.index.time <= dt.time(15, 15))]
            b = b[~((b.index.date == d0) & (b.index.time < dt.time(9, 20)))]
            bars[n] = b
        if any(len(b) < 10 for b in bars.values()):
            continue
        idx = bars["sc"].index
        for n in bars:
            bars[n] = bars[n].reindex(idx).ffill().bfill()
        entry = {}
        for n, (s_, t_, sign) in legs.items():
            px = float(bars[n]["open"].iloc[0])
            c_ = half_spread(px) + slippage
            entry[n] = px + c_ if sign > 0 else max(0.05, px - c_)
        credit = entry["sc"] + entry["sp"] - entry["lc"] - entry["lp"]
        if credit <= 1.0:
            continue
        reason, exit_mid = "time", None
        for t in idx[1:-1]:
            mark = (bars["sc"].at[t, "close"] + bars["sp"].at[t, "close"]
                    - bars["lc"].at[t, "close"] - bars["lp"].at[t, "close"])
            if mark >= credit * (1 + stop_mult):
                reason = "stop"
                exit_mid = {n: float(bars[n].at[t, "close"]) + (2.0 if legs[n][2] < 0 else -2.0) for n in legs}
                break
        if exit_mid is None:
            exit_mid = {n: float(bars[n]["close"].iloc[-1]) for n in legs}
        exit_px = {}
        for n, (s_, t_, sign) in legs.items():
            px = exit_mid[n]
            c_ = half_spread(px) + slippage
            exit_px[n] = px + c_ if sign < 0 else max(0.05, px - c_)
        debit = exit_px["sc"] + exit_px["sp"] - exit_px["lc"] - exit_px["lp"]
        ch = sum(order_charges("option", "SELL" if sg < 0 else "BUY", entry[n] * lot, rates)["total"] +
                 order_charges("option", "BUY" if sg < 0 else "SELL", exit_px[n] * lot, rates)["total"]
                 for n, (_, _, sg) in legs.items())
        gross = (credit - debit) * lot
        rows.append({"date": d0.isoformat(), "exit_date": d1.isoformat(), "expiry": exp.isoformat(),
                     "sessions": len(sess), "spot": round(S, 2), "vix": float(F.loc[ts0, "px_vix_prev"]),
                     "short_ce": sc, "short_pe": sp, "credit": round(credit, 2), "debit": round(debit, 2),
                     "exit_reason": reason, "gross_pnl": round(gross, 2), "charges": round(ch, 2),
                     "net_pnl": round(gross - ch, 2), "max_loss": round((wing - credit) * lot, 0)})
    T = pd.DataFrame(rows)
    return {"trades": T, "summary": summarize_condor(T)}
