"""Classic technical indicators for intraday and daily bars. Causal: row t uses bars <= t.

NIFTY (the index) has no traded volume, so VWAP is computed from NIFTY *futures* bars when
they are passed in; with zero volume it falls back to an equal-weight TWAP and says so."""

from __future__ import annotations

import datetime as dt
from typing import Optional

import numpy as np
import pandas as pd

from .smc import atr


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    up, dn = h.diff(), -l.diff()
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    a = atr(df, n).replace(0, np.nan)
    pdi = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / n, adjust=False).mean() / a
    mdi = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / n, adjust=False).mean() / a
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False).mean()


def supertrend(df: pd.DataFrame, n: int = 10, mult: float = 3.0) -> pd.DataFrame:
    """Standard recursive Supertrend. dir=+1 uptrend, -1 downtrend."""
    h, l, c = (df[x].to_numpy(float) for x in ("high", "low", "close"))
    a = atr(df, n).to_numpy()
    mid = (h + l) / 2
    ub, lb = mid + mult * a, mid - mult * a
    fub, flb = ub.copy(), lb.copy()
    st = np.full(len(c), np.nan)
    d = np.ones(len(c))
    for i in range(1, len(c)):
        fub[i] = ub[i] if (ub[i] < fub[i - 1] or c[i - 1] > fub[i - 1]) else fub[i - 1]
        flb[i] = lb[i] if (lb[i] > flb[i - 1] or c[i - 1] < flb[i - 1]) else flb[i - 1]
        if d[i - 1] > 0:
            d[i] = -1 if c[i] < flb[i] else 1
        else:
            d[i] = 1 if c[i] > fub[i] else -1
        st[i] = flb[i] if d[i] > 0 else fub[i]
    return pd.DataFrame({"st": st, "st_dir": d}, index=df.index)


def efficiency_ratio(close: pd.Series, n: int = 20) -> pd.Series:
    """Kaufman efficiency ratio: |net move| / path length. ~1 = trending, ~0 = choppy."""
    net = (close - close.shift(n)).abs()
    path = close.diff().abs().rolling(n).sum()
    return net / path.replace(0, np.nan)


def _session(idx: pd.DatetimeIndex) -> pd.Series:
    return pd.Series(idx.date, index=idx)


def session_vwap(df: pd.DataFrame, volume: Optional[pd.Series] = None) -> pd.DataFrame:
    """Session-anchored VWAP and volume-weighted standard deviation bands."""
    tp = (df["high"] + df["low"] + df["close"]) / 3
    v = (volume if volume is not None else df.get("volume", pd.Series(0.0, index=df.index))).reindex(df.index).fillna(0.0)
    has_vol = bool((v > 0).mean() > 0.5)
    w = v.where(v > 0, 0.0) if has_vol else pd.Series(1.0, index=df.index)
    g = _session(df.index)
    cw = w.groupby(g).cumsum().replace(0, np.nan)
    vwap = (tp * w).groupby(g).cumsum() / cw
    var = (tp * tp * w).groupby(g).cumsum() / cw - vwap ** 2
    sd = np.sqrt(var.clip(lower=0))
    return pd.DataFrame({"vwap": vwap, "vwap_sd": sd, "vwap_z": (df["close"] - vwap) / sd.replace(0, np.nan),
                         "vwap_is_twap": float(not has_vol)}, index=df.index)


def opening_range(df: pd.DataFrame, minutes: int = 15, open_time: dt.time = dt.time(9, 15)) -> pd.DataFrame:
    """Opening-range high/low, known only once the range is complete. orb_state: +1 above, -1 below."""
    t0 = dt.datetime.combine(dt.date(2000, 1, 1), open_time)
    end = (t0 + dt.timedelta(minutes=minutes)).time()
    g = _session(df.index)
    # bar timestamps are bar *starts*; a bar starting at `x` has closed by `x + bar length`
    in_rng = pd.Series(df.index.time < end, index=df.index)
    hi = df["high"].where(in_rng).groupby(g).cummax()
    lo = df["low"].where(in_rng).groupby(g).cummin()
    hi = hi.groupby(g).ffill()
    lo = lo.groupby(g).ffill()
    complete = ~in_rng
    hi, lo = hi.where(complete), lo.where(complete)
    c = df["close"]
    state = pd.Series(np.where(c > hi, 1.0, np.where(c < lo, -1.0, 0.0)), index=df.index).where(complete)
    return pd.DataFrame({"orb_hi": hi, "orb_lo": lo, "orb_state": state,
                         "orb_pos": (c - lo) / (hi - lo).replace(0, np.nan)}, index=df.index)


def prev_day_levels(df: pd.DataFrame) -> pd.DataFrame:
    """Previous session's high/low/close and floor-pivot CPR, broadcast onto intraday bars."""
    g = df.index.date
    daily = df.groupby(g).agg(high=("high", "max"), low=("low", "min"), close=("close", "last"))
    prev = daily.shift(1)
    piv = (prev["high"] + prev["low"] + prev["close"]) / 3
    bc = (prev["high"] + prev["low"]) / 2
    tc = 2 * piv - bc
    lv = pd.DataFrame({"pdh": prev["high"], "pdl": prev["low"], "pdc": prev["close"], "pivot": piv,
                       "cpr_top": np.maximum(tc, bc), "cpr_bot": np.minimum(tc, bc)})
    return lv.reindex(g).set_index(df.index)


def intraday_technical_frame(df: pd.DataFrame, volume: Optional[pd.Series] = None) -> pd.DataFrame:
    """One row per intraday bar: VWAP bands, ORB, previous-day levels, Supertrend, RSI, ADX."""
    a = atr(df, 14)
    X = pd.concat([session_vwap(df, volume), opening_range(df), prev_day_levels(df), supertrend(df)], axis=1)
    c = df["close"]
    X["rsi14"] = rsi(c)
    X["adx14"] = adx(df)
    X["er20"] = efficiency_ratio(c, 20)
    X["atr"] = a
    for lvl in ("vwap", "pdh", "pdl", "pivot", "st"):
        X[f"d_{lvl}_atr"] = (c - X[lvl]) / a.replace(0, np.nan)
    return X
