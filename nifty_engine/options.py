"""Option pricing (Black-Scholes with a VIX-anchored smile), strike selection, structures.

Used (a) to model option premiums in historical replays, where free historical option
prices don't exist, and (b) as a fallback in live paper trading when the broker chain is
unavailable. Live paper fills prefer real bid/ask from the broker option chain.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List

STRIKE_STEP = 50.0
RISK_FREE = 0.065


def _ncdf(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def bs_price(S: float, K: float, T: float, iv: float, typ: str, r: float = RISK_FREE) -> float:
    T = max(T, 1e-6)
    iv = max(iv, 0.03)
    sq = iv * math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * iv * iv) * T) / sq
    d2 = d1 - sq
    if typ == "CE":
        return S * _ncdf(d1) - K * math.exp(-r * T) * _ncdf(d2)
    return K * math.exp(-r * T) * _ncdf(-d2) - S * _ncdf(-d1)


def _npdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def bs_greeks(S: float, K: float, T: float, iv: float, typ: str, r: float = RISK_FREE) -> dict:
    """Delta, gamma (per point), vega (per 1 vol point), theta (per calendar day)."""
    T = max(T, 1e-6)
    iv = max(iv, 0.01)
    sq = iv * math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * iv * iv) * T) / sq
    d2 = d1 - sq
    gamma = _npdf(d1) / (S * sq)
    vega = S * _npdf(d1) * math.sqrt(T) / 100
    if typ == "CE":
        delta = _ncdf(d1)
        theta = (-S * _npdf(d1) * iv / (2 * math.sqrt(T)) - r * K * math.exp(-r * T) * _ncdf(d2)) / 365
    else:
        delta = _ncdf(d1) - 1
        theta = (-S * _npdf(d1) * iv / (2 * math.sqrt(T)) + r * K * math.exp(-r * T) * _ncdf(-d2)) / 365
    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta}


def implied_vol(price: float, S: float, K: float, T: float, typ: str, r: float = RISK_FREE,
                lo: float = 0.01, hi: float = 3.0) -> float:
    """Bisection IV. NaN when the price is outside no-arbitrage bounds (stale/illiquid quotes)."""
    if not (price and price > 0 and S > 0 and K > 0):
        return float("nan")
    T = max(T, 1e-6)
    intrinsic = max(0.0, S - K * math.exp(-r * T)) if typ == "CE" else max(0.0, K * math.exp(-r * T) - S)
    if price < intrinsic - 0.05 or price >= (S if typ == "CE" else K):
        return float("nan")
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if bs_price(S, K, T, mid, typ, r) > price:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def smile_iv(vix: float, S: float, K: float) -> float:
    """India VIX (30d, %) -> strike IV with a mild put skew. m = ln(K/S)."""
    base = max(vix, 8.0) / 100.0
    m = math.log(K / S)
    return max(0.05, base - 0.60 * m + 4.0 * m * m)


def half_spread(premium: float) -> float:
    """Modeled half bid-ask for liquid NIFTY weeklies (points)."""
    return max(0.05, min(2.0, 0.004 * premium + 0.05))


@dataclass
class Leg:
    typ: str          # CE / PE
    strike: float
    qty_sign: int     # +1 long, -1 short (per lot)
    entry: float = 0.0
    exit: float = 0.0
    instrument_key: str = ""


@dataclass
class Structure:
    direction: int                 # +1 bullish, -1 bearish
    style: str
    legs: List[Leg] = field(default_factory=list)

    def value(self, S: float, T: float, vix: float) -> float:
        """Mid value per unit (net premium of long minus short legs)."""
        return sum(l.qty_sign * bs_price(S, l.strike, T, smile_iv(vix, S, l.strike), l.typ) for l in self.legs)

    def fill_value(self, S: float, T: float, vix: float, opening: bool, slippage_pts: float) -> float:
        """Value after crossing the spread: pay more when opening, receive less when closing."""
        tot = 0.0
        for l in self.legs:
            px = bs_price(S, l.strike, T, smile_iv(vix, S, l.strike), l.typ)
            cost = half_spread(px) + slippage_pts
            buying = (l.qty_sign > 0) == opening
            leg_px = px + cost if buying else max(0.05, px - cost)
            if opening:
                l.entry = leg_px
            else:
                l.exit = leg_px
            tot += l.qty_sign * leg_px
        return tot


def atm(S: float) -> float:
    return round(S / STRIKE_STEP) * STRIKE_STEP


def build_structure(direction: int, S: float, style: str, width: float = 150.0) -> Structure:
    typ = "CE" if direction > 0 else "PE"
    a = atm(S)
    if style == "itm_buy":
        k = a - STRIKE_STEP if direction > 0 else a + STRIKE_STEP
        return Structure(direction, style, [Leg(typ, k, +1)])
    if style == "debit_spread":
        far = a + width if direction > 0 else a - width
        return Structure(direction, style, [Leg(typ, a, +1), Leg(typ, far, -1)])
    raise ValueError(f"unknown option style {style}")
