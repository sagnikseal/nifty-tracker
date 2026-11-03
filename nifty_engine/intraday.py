"""Intraday direction model (phase 2): decide at T (e.g. 10:15 IST), predict NIFTY from T to 15:15.

Why intraday: by mid-morning the session has revealed information that a 09:20 model can't
see: the first-half-hour move (Gao, Han, Li & Zhou 2018, intraday momentum), the opening-
range break, the VIX reaction and how far the gap has been filled. Every feature uses only 5-minute bars that have
*closed* by T. Requires store/intraday/nifty_5m.csv (Upstox v3, from Jan-2022).
"""

from __future__ import annotations

import datetime as dt
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .calendar import IST
from .config import DATA_DIR

EXIT_BAR = dt.time(15, 10)        # bar 15:10-15:15 -> its close is the 15:15 exit price


def load_5m(name: str = "nifty_5m") -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "intraday" / f"{name}.csv", index_col=0)
    df.index = pd.to_datetime(df.index, utc=True).tz_convert(IST)
    df = df[(df.index.time >= dt.time(9, 15)) & (df.index.time <= dt.time(15, 25))]
    return df.sort_index()


def build_intraday_frame(decision_times: List[str], daily_feats: Optional[pd.DataFrame] = None,
                         daily_p: Optional[pd.Series] = None, prev_close: Optional[pd.Series] = None) -> pd.DataFrame:
    """One row per (session, decision time). Columns: features + y_ret/y_up + px_T + T."""
    n5 = load_5m("nifty_5m")
    try:
        v5 = load_5m("indiavix_5m")
    except FileNotFoundError:
        v5 = pd.DataFrame()
    rows = []
    days = sorted(set(n5.index.date))
    prev_day_close = {}
    last_close = None
    for d in days:
        prev_day_close[d] = last_close
        g = n5[n5.index.date == d]
        if len(g):
            last_close = float(g["close"].iloc[-1])
    vix_by_day = {d: g for d, g in v5.groupby(v5.index.date)} if len(v5) else {}
    vix_prev = {}
    lc = None
    for d in days:
        vix_prev[d] = lc
        if d in vix_by_day and len(vix_by_day[d]):
            lc = float(vix_by_day[d]["close"].iloc[-1])
    for d, g in n5.groupby(n5.index.date):
        if len(g) < 70:
            continue  # special / truncated sessions
        pc = None
        if prev_close is not None and pd.Timestamp(d) in prev_close.index:
            pc = prev_close.loc[pd.Timestamp(d)]
        pc = pc if pc is not None and not pd.isna(pc) else prev_day_close.get(d)
        if not pc:
            continue
        o = float(g["open"].iloc[0])
        ex = g[g.index.time == EXIT_BAR]
        if ex.empty:
            continue
        p_exit = float(ex["close"].iloc[0])
        r5 = np.log(g["close"]).diff()
        vg = vix_by_day.get(d)
        for T in decision_times:
            t = dt.datetime.strptime(T, "%H:%M").time()
            closed = g[g.index.time < t]              # bars fully closed by T
            if len(closed) < 3:
                continue
            pT = float(closed["close"].iloc[-1])
            hi, lo = float(closed["high"].max()), float(closed["low"].min())
            orb = g[g.index.time < dt.time(9, 45)]
            orb = orb if len(orb) and orb.index[-1].time() < t else closed
            orb_h, orb_l = float(orb["high"].max()), float(orb["low"].min())
            first30 = g[g.index.time < dt.time(9, 45)]
            p945 = float(first30["close"].iloc[-1]) if len(first30) and t >= dt.time(9, 45) else pT
            rr = r5[closed.index].dropna()
            rv = float(rr.std() * np.sqrt(75)) if len(rr) > 2 else np.nan
            twap = float(closed["close"].mean())
            last6 = closed["close"].iloc[-7:]
            f = {
                "date": pd.Timestamp(d), "T": T, "t_min": t.hour * 60 + t.minute,
                "gap": np.log(o / pc),
                "r_open_T": np.log(pT / o),
                "r_pc_T": np.log(pT / pc),
                "r_first30": np.log(p945 / pc),
                "r_last30": np.log(pT / float(last6.iloc[0])) if len(last6) > 1 else 0.0,
                "range_pos": (pT - lo) / (hi - lo) if hi > lo else 0.5,
                "range_pct": (hi - lo) / o,
                "rv_sofar": rv,
                "dist_twap": np.log(pT / twap),
                "orb_break": (1.0 if pT > orb_h else (-1.0 if pT < orb_l else 0.0)),
                "orb_dist": (pT - (orb_h + orb_l) / 2) / max(orb_h - orb_l, 1.0),
                "frac_above_open": float((closed["close"] > o).mean()),
                "gap_fill": (pT - o) / (pc - o) if abs(pc - o) > 1 else 0.0,
                "up_bars": float((closed["close"] > closed["open"]).mean()),
            }
            if vg is not None and len(vg) and vix_prev.get(d):
                vc = vg[vg.index.time < t]
                if len(vc):
                    f["vix_chg_T"] = np.log(float(vc["close"].iloc[-1]) / vix_prev[d])
                    f["vix_chg_open_T"] = np.log(float(vc["close"].iloc[-1]) / float(vc["open"].iloc[0]))
            f["y_ret"] = np.log(p_exit / pT)
            f["y_up"] = float(p_exit > pT)
            f["px_T"] = pT
            f["px_exit"] = p_exit
            rows.append(f)
    X = pd.DataFrame(rows)
    X = X.set_index("date")
    if daily_feats is not None:
        keep = [c for c in daily_feats.columns if not c.startswith(("y_", "px_")) and c not in X.columns and c != "gap"]
        X = X.join(daily_feats[keep], how="left")
    if daily_p is not None:
        X = X.join(daily_p.rename("p_daily"), how="left")
    return X.replace([np.inf, -np.inf], np.nan)
