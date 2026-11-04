"""Trade decision, sizing and charge accounting shared by replay and live paper trading."""

from __future__ import annotations

import datetime as dt
import math
from typing import Dict, Optional

from ..calendar import IST, next_weekly_expiry, year_fraction
from ..config import EngineConfig
from ..options import Structure, build_structure
from .charges import ChargeRates, order_charges


def decide(p_up: float, cfg: EngineConfig) -> int:
    if p_up >= cfg.prob_threshold:
        return 1
    if p_up <= 1 - cfg.prob_threshold:
        return -1
    return 0


def rates_from(cfg: EngineConfig) -> ChargeRates:
    return ChargeRates(brokerage_per_order=cfg.brokerage_per_order)


def structure_charges(st: Structure, lots: int, lot_size: int, rates: ChargeRates, opening: bool) -> float:
    tot = 0.0
    for l in st.legs:
        px = l.entry if opening else l.exit
        buy = (l.qty_sign > 0) == opening
        tot += order_charges("option", "BUY" if buy else "SELL", px * lot_size * lots, rates)["total"]
    return tot


def size_option_trade(entry_value: float, equity: float, cfg: EngineConfig) -> int:
    """Lots allowed by (a) loss budget at the stop and (b) premium budget. 0 = skip."""
    per_lot_premium = entry_value * cfg.lot_size
    per_lot_risk = per_lot_premium * cfg.option_stop_pct / 100.0 + 4 * cfg.brokerage_per_order * 1.18
    if per_lot_premium <= 0:
        return 0
    by_risk = math.floor(equity * cfg.risk_per_trade_pct / 100.0 / per_lot_risk)
    by_prem = math.floor(equity * cfg.max_premium_pct / 100.0 / per_lot_premium)
    return int(max(0, min(cfg.max_lots, by_risk, by_prem)))


def expiry_T(day: dt.date, at: dt.time) -> tuple:
    exp = next_weekly_expiry(day, min_days=1)   # never trade 0DTE: next week's contract on expiry day
    start = dt.datetime.combine(day, at, tzinfo=IST)
    end = dt.datetime.combine(exp, dt.time(15, 30), tzinfo=IST)
    return exp, year_fraction(start, end)


def simulate_option_day(day: dt.date, direction: int, px: Dict[str, float], vix_entry: float,
                        vix_exit: float, equity: float, cfg: EngineConfig) -> Optional[Dict]:
    """Simulate one intraday option trade from daily OHLC with modeled premiums.
    Conservative path rule: if stop and target are both reachable on the day, the stop wins."""
    S0, H, L, C = px["open"], px["high"], px["low"], px["close"]
    exp, T0 = expiry_T(day, dt.time(9, 20))
    _, Tm = expiry_T(day, dt.time(12, 30))
    _, T1 = expiry_T(day, dt.time(15, 15))
    st = build_structure(direction, S0, cfg.option_style, cfg.spread_width_pts)
    entry = st.fill_value(S0, T0, vix_entry, opening=True, slippage_pts=cfg.option_slippage_pts)
    lots = size_option_trade(entry, equity, cfg)
    if lots == 0 or entry <= 0.5:
        return {"skipped": "insufficient_capital", "entry_value": entry}
    vix_mid = 0.5 * (vix_entry + vix_exit)
    adverse_S, favour_S = (L, H) if direction > 0 else (H, L)
    worst = st.value(adverse_S, Tm, vix_mid)
    best = st.value(favour_S, Tm, vix_mid)
    stop_v = entry * (1 - cfg.option_stop_pct / 100.0)
    tgt_v = entry * (1 + cfg.option_target_pct / 100.0)
    if worst <= stop_v:
        reason, mid_exit = "stop", stop_v
    elif best >= tgt_v:
        reason, mid_exit = "target", tgt_v
    else:
        reason, mid_exit = "time", None
    if mid_exit is None:
        exit_v = st.fill_value(C, T1, vix_exit, opening=False, slippage_pts=cfg.option_slippage_pts)
    else:
        # Find the underlying level where the structure's mid value hits the trigger, then
        # close every leg there, paying half-spread + slippage on each.
        lo, hi = (S0, adverse_S) if reason == "stop" else (S0, favour_S)
        for _ in range(40):
            mid_S = 0.5 * (lo + hi)
            v = st.value(mid_S, Tm, vix_mid)
            hit = v <= mid_exit if reason == "stop" else v >= mid_exit
            lo, hi = (lo, mid_S) if hit else (mid_S, hi)
        exit_v = st.fill_value(hi, Tm, vix_mid, opening=False, slippage_pts=cfg.option_slippage_pts)
    rates = rates_from(cfg)
    charges = structure_charges(st, lots, cfg.lot_size, rates, True) + structure_charges(st, lots, cfg.lot_size, rates, False)
    gross = (exit_v - entry) * cfg.lot_size * lots
    return {
        "date": day.isoformat(), "direction": direction, "style": st.style, "expiry": exp.isoformat(),
        "legs": [{"typ": l.typ, "strike": l.strike, "side": "BUY" if l.qty_sign > 0 else "SELL",
                  "entry": round(l.entry, 2), "exit": round(l.exit, 2)} for l in st.legs],
        "lots": lots, "entry_value": round(entry, 2), "exit_value": round(exit_v, 2),
        "gross_pnl": round(gross, 2), "charges": round(charges, 2), "net_pnl": round(gross - charges, 2),
        "exit_reason": reason, "underlying_move_pts": round(C - S0, 2),
    }


def simulate_future_day(day: dt.date, direction: int, px: Dict[str, float], atr_pts: float,
                        equity: float, cfg: EngineConfig) -> Optional[Dict]:
    S0, H, L, C = px["open"], px["high"], px["low"], px["close"]
    margin_per_lot = 0.10 * S0 * cfg.lot_size
    stop_pts = max(20.0, 0.6 * atr_pts)
    tgt_pts = 1.5 * stop_pts
    lots = int(min(cfg.max_lots, equity // (margin_per_lot * 1.3),
                   (equity * cfg.risk_per_trade_pct / 100) // (stop_pts * cfg.lot_size)))
    if lots <= 0:
        return {"skipped": "insufficient_margin", "margin_per_lot": round(margin_per_lot)}
    entry = S0 + direction * cfg.futures_slippage_pts
    adverse = (entry - L) if direction > 0 else (H - entry)
    favour = (H - entry) if direction > 0 else (entry - L)
    if adverse >= stop_pts:
        pts, reason = -stop_pts - cfg.futures_slippage_pts, "stop"
    elif favour >= tgt_pts:
        pts, reason = tgt_pts - cfg.futures_slippage_pts, "target"
    else:
        pts, reason = direction * (C - entry) - cfg.futures_slippage_pts, "time"
    exit_px = entry + direction * pts
    rates = rates_from(cfg)
    q = cfg.lot_size * lots
    ch = (order_charges("future", "BUY" if direction > 0 else "SELL", entry * q, rates)["total"] +
          order_charges("future", "SELL" if direction > 0 else "BUY", exit_px * q, rates)["total"])
    gross = pts * q
    return {"date": day.isoformat(), "direction": direction, "style": "future", "lots": lots,
            "entry_value": round(entry, 2), "exit_value": round(exit_px, 2), "gross_pnl": round(gross, 2),
            "charges": round(ch, 2), "net_pnl": round(gross - ch, 2), "exit_reason": reason,
            "underlying_move_pts": round(C - S0, 2)}
