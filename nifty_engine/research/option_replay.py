"""Execute a TradePlan on historical prices.

`simulate_real`: real 5-minute bars of the exact expired contracts (Upstox Plus download).
`simulate_modeled`: Black-Scholes from India VIX on NIFTY 5-minute bars, for days or strikes the
download doesn't cover. Every trade records which one was used (`price_source`).

Fill rules (same for both): entry at the open of the bar that starts at the plan's entry time,
crossing half the modeled spread plus `option_slippage_pts`; exits are checked on synchronous
5-minute closes (summing intrabar extremes of different legs would invent prices that never
co-existed); stop exits pay 1 extra point per leg; time exits use the open of the exit bar."""

from __future__ import annotations

import datetime as dt
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..calendar import IST, year_fraction
from ..config import EngineConfig
from ..data.expired_options import load_expiry
from ..options import bs_price, half_spread, smile_iv
from ..paper.charges import order_charges
from ..paper.sim import rates_from
from ..strategies.base import TradePlan

_CACHE: Dict[dt.date, pd.DataFrame] = {}


def _expiry_bars(e: dt.date) -> pd.DataFrame:
    if e not in _CACHE:
        if len(_CACHE) > 6:
            _CACHE.clear()
        _CACHE[e] = load_expiry(e)
    return _CACHE[e]


def _window(plan: TradePlan):
    t0 = pd.Timestamp(dt.datetime.combine(plan.day, plan.entry_time), tz=IST)
    t1 = pd.Timestamp(dt.datetime.combine(plan.exit_day or plan.day, plan.exit_time), tz=IST)
    return t0, t1


def _leg_series(opt: pd.DataFrame, strike: float, typ: str, t0, t1) -> pd.DataFrame:
    m = (opt["strike"] == strike) & (opt["type"] == typ) & (opt["ts"] >= t0) & (opt["ts"] <= t1)
    b = opt.loc[m, ["ts", "open", "close"]].drop_duplicates("ts").set_index("ts").sort_index()
    return b[(b.index.time >= dt.time(9, 15)) & (b.index.time <= dt.time(15, 25))]


def _run(plan: TradePlan, px_open: Dict[int, pd.Series], px_close: Dict[int, pd.Series], spot: pd.Series,
         cfg: EngineConfig, source: str) -> Optional[Dict]:
    legs = plan.legs
    idx = px_close[0].index
    slip = cfg.option_slippage_pts
    entry = []
    for i, l in enumerate(legs):
        p = float(px_open[i].iloc[0])
        c = half_spread(p) + slip
        entry.append(p + c if l.qty_sign > 0 else max(0.05, p - c))
    v0 = sum(l.qty_sign * e for l, e in zip(legs, entry))
    credit = -v0 if v0 < 0 else 0.0
    reason, exit_mid, k_exit = "time", None, len(idx) - 1
    for k in range(1, len(idx) - 1):
        v = sum(l.qty_sign * float(px_close[i].iloc[k]) for i, l in enumerate(legs))
        pnl = v - v0
        s = float(spot.iloc[k]) if spot is not None and k < len(spot) and spot.iloc[k] == spot.iloc[k] else None
        hit = None
        if v0 > 0:
            if plan.stop_pct is not None and v <= v0 * (1 - plan.stop_pct / 100):
                hit = "stop"
            elif plan.target_pct is not None and v >= v0 * (1 + plan.target_pct / 100):
                hit = "target"
        else:
            if plan.credit_stop_mult is not None and pnl <= -plan.credit_stop_mult * credit:
                hit = "stop"
            elif plan.credit_target_pct is not None and pnl >= plan.credit_target_pct / 100 * credit:
                hit = "target"
        if hit is None and s is not None and plan.direction != 0:
            if plan.underlying_stop is not None and (s - plan.underlying_stop) * plan.direction <= 0:
                hit = "stop_underlying"
            elif plan.underlying_target is not None and (s - plan.underlying_target) * plan.direction >= 0:
                hit = "target_underlying"
        if hit:
            reason, k_exit = hit, k
            extra = 1.0 if hit.startswith("stop") else 0.0
            exit_mid = [float(px_close[i].iloc[k]) + (-extra if l.qty_sign > 0 else extra) for i, l in enumerate(legs)]
            break
    if exit_mid is None:
        exit_mid = [float(px_open[i].iloc[-1]) for i in range(len(legs))]
    exit_px = []
    for l, p in zip(legs, exit_mid):
        c = half_spread(max(p, 0.05)) + slip
        exit_px.append(max(0.05, p - c) if l.qty_sign > 0 else p + c)
    v1 = sum(l.qty_sign * e for l, e in zip(legs, exit_px))
    q = cfg.lot_size
    rates = rates_from(cfg)
    ch = 0.0
    for l, a, b in zip(legs, entry, exit_px):
        ch += order_charges("option", "BUY" if l.qty_sign > 0 else "SELL", a * q, rates)["total"]
        ch += order_charges("option", "SELL" if l.qty_sign > 0 else "BUY", b * q, rates)["total"]
    gross = (v1 - v0) * q
    width = None
    if credit > 0:
        ks = sorted({l.strike for l in legs})
        width = max(ks[i + 1] - ks[i] for i in range(len(ks) - 1)) if len(ks) > 1 else None
    max_loss = (width - credit) * q if width else (v0 * q if v0 > 0 else None)
    return {"book": plan.book, "date": plan.day.isoformat(), "entry_ts": str(idx[0]), "exit_ts": str(idx[k_exit]),
            "direction": plan.direction, "style": plan.style, "expiry": plan.expiry.isoformat(),
            "legs": [{"typ": l.typ, "strike": l.strike, "sign": l.qty_sign, "entry": round(a, 2), "exit": round(b, 2)}
                     for l, a, b in zip(legs, entry, exit_px)],
            "entry_value": round(v0, 2), "exit_value": round(v1, 2), "gross_pnl": round(gross, 2),
            "charges": round(ch, 2), "net_pnl": round(gross - ch, 2), "exit_reason": reason,
            "max_loss": round(max_loss, 0) if max_loss else None, "price_source": source, **{
                f"m_{k}": v for k, v in plan.meta.items() if isinstance(v, (int, float, str, bool))}}


def _spot_on(index: pd.DatetimeIndex, n5: Optional[pd.DataFrame]) -> Optional[pd.Series]:
    if n5 is None or n5.empty:
        return None
    return n5["close"].reindex(index, method="ffill")


def simulate_real(plan: TradePlan, cfg: EngineConfig, n5: Optional[pd.DataFrame] = None) -> Optional[Dict]:
    opt = _expiry_bars(plan.expiry)
    if opt.empty:
        return None
    t0, t1 = _window(plan)
    series = [_leg_series(opt, l.strike, l.typ, t0, t1) for l in plan.legs]
    if any(len(s) < 3 for s in series):
        return None
    idx = series[0].index
    for s in series[1:]:
        idx = idx.union(s.index)
    idx = idx[(idx >= t0) & (idx <= t1)]
    if len(idx) < 3 or idx[0] - t0 > pd.Timedelta(minutes=10):
        return None
    po = {i: s["open"].reindex(idx).ffill().bfill() for i, s in enumerate(series)}
    pc = {i: s["close"].reindex(idx).ffill().bfill() for i, s in enumerate(series)}
    return _run(plan, po, pc, _spot_on(idx, n5), cfg, "real")


def simulate_modeled(plan: TradePlan, cfg: EngineConfig, n5: pd.DataFrame, vix_prev: float) -> Optional[Dict]:
    """Same fills, premiums from Black-Scholes with the VIX-anchored smile (see options.py)."""
    if n5 is None or n5.empty or not vix_prev:
        return None
    t0, t1 = _window(plan)
    b = n5[(n5.index >= t0) & (n5.index <= t1)]
    b = b[(b.index.time >= dt.time(9, 15)) & (b.index.time <= dt.time(15, 25))]
    if len(b) < 3 or b.index[0] - t0 > pd.Timedelta(minutes=10):
        return None
    exp_dt = dt.datetime.combine(plan.expiry, dt.time(15, 30), tzinfo=IST)
    po, pc = {}, {}
    for i, l in enumerate(plan.legs):
        o, c = [], []
        for ts, r in b.iterrows():
            T0 = year_fraction(ts.to_pydatetime(), exp_dt)
            T1 = year_fraction((ts + pd.Timedelta(minutes=5)).to_pydatetime(), exp_dt)
            o.append(bs_price(float(r["open"]), l.strike, T0, smile_iv(vix_prev, float(r["open"]), l.strike), l.typ))
            c.append(bs_price(float(r["close"]), l.strike, T1, smile_iv(vix_prev, float(r["close"]), l.strike), l.typ))
        po[i], pc[i] = pd.Series(o, index=b.index), pd.Series(c, index=b.index)
    return _run(plan, po, pc, b["close"], cfg, "model_vix")


def simulate(plans: List[TradePlan], cfg: EngineConfig, n5: Optional[pd.DataFrame], vix_daily: pd.Series,
             allow_modeled: bool = True) -> pd.DataFrame:
    rows = []
    for p in plans:
        r = simulate_real(p, cfg, n5)
        if r is None and allow_modeled:
            prev = vix_daily[vix_daily.index < pd.Timestamp(p.day)]
            r = simulate_modeled(p, cfg, n5, float(prev.iloc[-1]) if len(prev) else None)
        if r is not None:
            rows.append(r)
    return pd.DataFrame(rows)
