"""Historical option-chain features rebuilt from expired-contract 5-minute candles (Upstox Plus).

For a decision at 09:20 on day t (contract = nearest expiry with >= 1 day left):
  * the previous session's closing chain (last bar of t-1): PCR, walls, max pain, ATM IV, skew,
    implied move, and OI change over day t-1 (build-up)
  * the first 5-minute bar of day t (09:15-09:20, closed at 09:20): ATM IV and PCR change since the
    previous close
Needs `python -m nifty_engine fetch-expired-options --mode full` and `fetch-intraday`.
"""

from __future__ import annotations

import datetime as dt
import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..analytics.options_micro import chain_metrics
from ..calendar import IST, year_fraction
from ..config import DATA_DIR
from .expired_options import OUT, load_expiry, sessions_per_expiry

log = logging.getLogger(__name__)
FEATURES_FILE = DATA_DIR / "features" / "options_hist.csv"


def snapshot(opt: pd.DataFrame, day: dt.date, upto: dt.time) -> pd.DataFrame:
    """Chain at the close of the last 5-minute bar *starting* before `upto` - 5min on `day`."""
    d = opt[opt["ts"].dt.date == day]
    cut = (dt.datetime.combine(day, upto) - dt.timedelta(minutes=5)).time()
    d = d[d["ts"].dt.time <= cut]
    if d.empty:
        return pd.DataFrame()
    last = d.sort_values("ts").groupby(["strike", "type"]).tail(1)
    vol = d.groupby(["strike", "type"])["volume"].sum().rename("volume_day")
    snap = last.set_index(["strike", "type"]).join(vol).reset_index()
    return snap.rename(columns={"close": "ltp"})[["strike", "type", "ltp", "oi", "volume_day"]].rename(
        columns={"volume_day": "volume"})


def _window(snap: pd.DataFrame, spot_0920: Optional[float], pts: int = 100) -> pd.DataFrame:
    if snap.empty or not spot_0920:
        return pd.DataFrame()
    k0 = round(spot_0920 / 50) * 50
    return snap[(snap["strike"] >= k0 - pts) & (snap["strike"] <= k0 + pts)]


def _spot_at(n5: pd.DataFrame, day: dt.date, upto: dt.time) -> Optional[float]:
    d = n5[n5.index.date == day]
    cut = (dt.datetime.combine(day, upto) - dt.timedelta(minutes=5)).time()
    d = d[d.index.time <= cut]
    return float(d["close"].iloc[-1]) if len(d) else None


def build(n5: pd.DataFrame, expiries: List[dt.date], lot_size: int = 65, save: bool = True) -> pd.DataFrame:
    days = sorted(set(n5.index.date))
    plan = sessions_per_expiry(expiries, days)
    day2exp = {d: e for e, ds in plan.items() for d in ds}
    prev_day = {d: days[i - 1] for i, d in enumerate(days) if i > 0}
    rows, cache = [], {}
    for d in days:
        e = day2exp.get(d)
        p = prev_day.get(d)
        if e is None or p is None or not (OUT / f"{e.isoformat()}.csv").exists():
            continue
        if e not in cache:
            cache = {e: load_expiry(e)}
        opt = cache[e]
        s_prev, s_open = _spot_at(n5, p, dt.time(15, 30)), _spot_at(n5, d, dt.time(9, 20))
        # Look-ahead guard: a weekly file holds the strikes needed on every day of that week (chosen from each
        # day's open), so the strike SET leaks later prices. Use only a window fixed by each day's own 09:20 spot.
        c_prev = _window(snapshot(opt, p, dt.time(15, 30)), _spot_at(n5, p, dt.time(9, 20)))
        c_open = _window(snapshot(opt, d, dt.time(9, 20)), s_open)
        p2 = prev_day.get(p)
        c_prev2 = _window(snapshot(opt, p2, dt.time(15, 30)), _spot_at(n5, p2, dt.time(9, 20))) if p2 else pd.DataFrame()
        if c_prev.empty or not s_prev or len(c_prev) < 10:
            continue
        T_prev = year_fraction(dt.datetime.combine(p, dt.time(15, 30), tzinfo=IST), dt.datetime.combine(e, dt.time(15, 30), tzinfo=IST))
        m = chain_metrics(c_prev, s_prev, T_prev, lot_size, prev=c_prev2 if len(c_prev2) else None)
        r = {"date": pd.Timestamp(d)}
        for k in ("pcr_oi", "pcr_vol", "call_wall_dist_pct", "put_wall_dist_pct", "max_pain_dist_pct", "atm_iv",
                  "rr25", "bf25", "implied_move_pct", "gex_cr", "zero_gamma_dist_pct", "doi_put_minus_call_share"):
            r[f"oh_{k}"] = m.get(k, np.nan)
        if len(c_open) >= 10 and s_open:
            T_open = year_fraction(dt.datetime.combine(d, dt.time(9, 20), tzinfo=IST), dt.datetime.combine(e, dt.time(15, 30), tzinfo=IST))
            mo = chain_metrics(c_open, s_open, T_open, lot_size, with_gex=False)
            r["oh_atm_iv_open_chg"] = mo.get("atm_iv", np.nan) - m.get("atm_iv", np.nan)
            r["oh_rr25_open_chg"] = mo.get("rr25", np.nan) - m.get("rr25", np.nan)
        rows.append(r)
    X = pd.DataFrame(rows).set_index("date").sort_index() if rows else pd.DataFrame()
    if save and len(X):
        FEATURES_FILE.parent.mkdir(parents=True, exist_ok=True)
        X.to_csv(FEATURES_FILE)
    return X


def load() -> pd.DataFrame:
    if not FEATURES_FILE.exists():
        return pd.DataFrame()
    return pd.read_csv(FEATURES_FILE, index_col=0, parse_dates=True)
