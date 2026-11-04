"""Rule-based SMC / ICT strategy, as described by the user's friend, coded WITHOUT look-ahead.

Process (his words -> our rules):
  * Daily bias, daily timeframe: "BOS and OGCL / OLCH". The bias is the sum of:
      - daily break-of-structure state (close beyond the last *confirmed* daily swing high/low),
      - previous day's candle: open~low & close~high (OLCH, +1) or open~high & close~low (OHCL, -1),
      - today's first 15-min candle: open~low (+1) or open~high (-1)   (known at 09:30).
    Trade only in the bias direction; no trades when the score is 0.
  * Liquidity, 15-min: previous-day high/low + confirmed 15-min swing highs/lows (prev + current day;
    equal highs/lows are just coinciding swings). A sweep = a 1-min bar trading through a level
    and closing back inside within 3 bars.
  * FVG, all timeframes -> entry on 1-min: after a sweep, a 1-min fair value gap in the bias
    direction; entry when a 1-min candle closes beyond the FVG after price has tapped it ("over
    FVG"), or breaks out of a tight 5-bar range ("small range").
  * ATM option (CE long / PE short), nearest expiry INCLUDING expiry day; up to 10 trades a day.
  * Exits are discretionary for him ("get out when gain or loss feels decent"), so several
    exit styles are simulated and ALL are reported.

Swing points are confirmed only after `k` later bars have closed, so nothing uses future data.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .calendar import IST
from .config import DATA_DIR

T_START, T_LAST_ENTRY, T_EXIT = dt.time(9, 30), dt.time(14, 45), dt.time(15, 20)


def load_1m() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "intraday" / "nifty_1m.csv", index_col=0)
    df.index = pd.to_datetime(df.index, utc=True).tz_convert(IST)
    df = df[(df.index.time >= dt.time(9, 15)) & (df.index.time <= dt.time(15, 29))]
    return df.sort_index()


def _fractals(h: np.ndarray, l: np.ndarray, k: int = 2):
    """Swing highs/lows with k bars on each side. Returns lists of (confirm_index, price)."""
    sh, sl = [], []
    for i in range(k, len(h) - k):
        if h[i] == max(h[i - k:i + k + 1]):
            sh.append((i + k, h[i]))     # known only once bar i+k has closed
        if l[i] == min(l[i - k:i + k + 1]):
            sl.append((i + k, l[i]))
    return sh, sl


def daily_bias_table(daily: pd.DataFrame) -> pd.DataFrame:
    """Pre-open bias components for each session (uses bars up to the previous day only)."""
    d = daily.sort_index()
    o, h, l, c = (d[x].to_numpy() for x in ("open", "high", "low", "close"))
    sh, sl = _fractals(h, l, 2)
    rows, state = [], 0
    shp = slp = None
    j_h = j_l = 0
    for i in range(1, len(d)):
        # update confirmed swings available at the end of day i-1
        while j_h < len(sh) and sh[j_h][0] <= i - 1:
            shp = sh[j_h][1]; j_h += 1
        while j_l < len(sl) and sl[j_l][0] <= i - 1:
            slp = sl[j_l][1]; j_l += 1
        pc = c[i - 1]
        if shp is not None and pc > shp:
            state = 1
        elif slp is not None and pc < slp:
            state = -1
        rng = max(h[i - 1] - l[i - 1], 1e-9)
        olch = int((o[i - 1] - l[i - 1]) <= 0.1 * rng and (h[i - 1] - pc) <= 0.1 * rng)
        ohcl = int((h[i - 1] - o[i - 1]) <= 0.1 * rng and (pc - l[i - 1]) <= 0.1 * rng)
        rows.append({"date": d.index[i], "bos_state": state, "prev_olch": olch - ohcl,
                     "pdh": h[i - 1], "pdl": l[i - 1]})
    return pd.DataFrame(rows).set_index("date")


@dataclass
class Setup:
    day: dt.date
    t_entry: pd.Timestamp
    side: int
    spot: float
    sweep_extreme: float
    target_level: float
    trigger: str
    bias: int
    level_kind: str


def find_setups(n1: pd.DataFrame, bias_tab: pd.DataFrame, max_per_day: int = 10,
                require_bias: bool = True, min_bars: int = 300) -> List[Setup]:
    out: List[Setup] = []
    days = sorted(set(n1.index.date))
    prev15 = None
    for d in days:
        g = n1[n1.index.date == d]
        if len(g) < min_bars or pd.Timestamp(d) not in bias_tab.index:
            prev15 = None
            continue
        b = bias_tab.loc[pd.Timestamp(d)]
        first = g[g.index.time < dt.time(9, 30)]
        fo, fh, fl = float(first["open"].iloc[0]), float(first["high"].max()), float(first["low"].min())
        frng = max(fh - fl, 1e-9)
        open_sig = 1 if (fo - fl) <= max(0.1 * frng, 3) else (-1 if (fh - fo) <= max(0.1 * frng, 3) else 0)
        score = int(b["bos_state"]) + int(b["prev_olch"]) + open_sig
        bias = int(np.sign(score))
        g15 = g.resample("15min", label="left", closed="left").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
        base15 = pd.concat([prev15, g15]) if prev15 is not None else g15
        h15, l15 = base15["high"].to_numpy(), base15["low"].to_numpy()
        sh15, sl15 = _fractals(h15, l15, 2)
        # confirmation time of each swing = close of bar (confirm_index) -> timestamp + 15min
        tconf_h = [(base15.index[ci] + pd.Timedelta(minutes=15), p) for ci, p in sh15]
        tconf_l = [(base15.index[ci] + pd.Timedelta(minutes=15), p) for ci, p in sl15]
        prev15 = g15
        if bias == 0 and require_bias:
            continue
        sides = [bias] if (bias != 0 and require_bias) else [1, -1]
        H, L, C, O = (g[x].to_numpy() for x in ("high", "low", "close", "open"))
        idx = g.index
        n_tr, busy_until = 0, -1
        for side in sides:
            pending = None          # dict with sweep info
            fvg = None
            for i in range(3, len(g) - 1):
                t = idx[i]
                if t.time() < T_START or t.time() > T_LAST_ENTRY or i <= busy_until or n_tr >= max_per_day:
                    continue
                # liquidity levels known at t
                lows = [float(b["pdl"])] + [p for tc, p in tconf_l if tc <= t]
                highs = [float(b["pdh"])] + [p for tc, p in tconf_h if tc <= t]
                if side > 0:
                    cand = [lv for lv in lows if L[i] < lv - 0.5 and C[i] > lv]
                    if cand:
                        lv = max(cand)
                        pending = {"i": i, "ext": L[i], "lvl": lv, "kind": "PDL" if lv == float(b["pdl"]) else "15m_low"}
                        fvg = None
                        continue
                else:
                    cand = [lv for lv in highs if H[i] > lv + 0.5 and C[i] < lv]
                    if cand:
                        lv = min(cand)
                        pending = {"i": i, "ext": H[i], "lvl": lv, "kind": "PDH" if lv == float(b["pdh"]) else "15m_high"}
                        fvg = None
                        continue
                if pending is None or i - pending["i"] > 30:
                    pending, fvg = None, None
                    continue
                pending["ext"] = min(pending["ext"], L[i]) if side > 0 else max(pending["ext"], H[i])
                trig = None
                # 1-min FVG in the bias direction, formed after the sweep
                if i - 2 > pending["i"]:
                    if side > 0 and L[i] > H[i - 2] + 1.0:
                        fvg = {"top": L[i], "bot": H[i - 2], "tapped": False}
                    elif side < 0 and H[i] < L[i - 2] - 1.0:
                        fvg = {"top": L[i - 2], "bot": H[i], "tapped": False}
                if fvg is not None:
                    if side > 0:
                        if L[i] <= fvg["top"]:
                            fvg["tapped"] = True
                        elif fvg["tapped"] and C[i] > fvg["top"]:
                            trig = "over_fvg"
                    else:
                        if H[i] >= fvg["bot"]:
                            fvg["tapped"] = True
                        elif fvg["tapped"] and C[i] < fvg["bot"]:
                            trig = "over_fvg"
                if trig is None and i - 5 >= pending["i"]:
                    rh, rl = H[i - 5:i].max(), L[i - 5:i].min()
                    if rh - rl <= 0.0005 * C[i]:
                        if side > 0 and C[i] > rh:
                            trig = "small_range"
                        elif side < 0 and C[i] < rl:
                            trig = "small_range"
                if trig:
                    spot = O[i + 1]
                    if side > 0:
                        tgts = [x for x in highs if x > spot + 5]
                        tgt = min(tgts) if tgts else spot + 2 * (spot - pending["ext"])
                    else:
                        tgts = [x for x in lows if x < spot - 5]
                        tgt = max(tgts) if tgts else spot - 2 * (pending["ext"] - spot)
                    out.append(Setup(d, idx[i + 1], side, float(spot), float(pending["ext"]), float(tgt),
                                     trig, bias, pending["kind"]))
                    n_tr += 1
                    busy_until = i + 15        # roughly one position at a time
                    pending, fvg = None, None
    return out


# ----------------------------------------------------------------------------- option simulation
def simulate_on_options(setups: List[Setup], n1: pd.DataFrame, exit_style: str, lot: int = 65,
                        slippage: float = 0.5, brokerage: float = 20.0) -> pd.DataFrame:
    from .data.expired_options import load_expiry, sessions_per_expiry
    from .options import half_spread
    from .paper.charges import ChargeRates, order_charges
    import os
    rates = ChargeRates(brokerage_per_order=brokerage)
    folder = DATA_DIR / "options_hist_1m"
    exps = sorted(dt.date.fromisoformat(f[:10]) for f in os.listdir(folder)) if folder.exists() else []
    days = sorted({s.day for s in setups})
    plan = sessions_per_expiry(exps, days, min_days=0)
    d2e = {d: e for e, ds in plan.items() for d in ds}
    cache: Dict = {}
    rows = []
    for s in setups:
        if s.day not in d2e:
            continue
        e = d2e[s.day]
        if e not in cache:
            o = load_expiry(e, one_minute=True)
            cache[e] = {k: v.set_index("ts").sort_index() for k, v in o.groupby([o.ts.dt.date, "strike", "type"])} if len(o) else {}
        strike = round(s.spot / 50) * 50
        typ = "CE" if s.side > 0 else "PE"
        b = cache[e].get((s.day, float(strike), typ))
        if b is None:
            continue
        b = b[(b.index >= s.t_entry) & (b.index.time <= T_EXIT)]
        if len(b) < 3:
            continue
        p0 = float(b["open"].iloc[0])
        entry = p0 + half_spread(p0) + slippage
        u = n1.loc[s.t_entry:].copy()
        u = u[u.index.date == s.day]
        exit_mid, reason, t_exit = None, "time", None
        if exit_style.startswith("pct"):
            tp, sl = (float(x) / 100 for x in exit_style[3:].split("_"))
            for t, r in b.iloc[1:].iterrows():
                if r["low"] <= entry * (1 - sl):
                    exit_mid, reason, t_exit = entry * (1 - sl) - 1.0, "stop", t
                    break
                if r["high"] >= entry * (1 + tp):
                    exit_mid, reason, t_exit = entry * (1 + tp), "target", t
                    break
        elif exit_style == "structure":
            for t, r in u.iloc[1:].iterrows():
                hit_stop = (r["low"] <= s.sweep_extreme - 2) if s.side > 0 else (r["high"] >= s.sweep_extreme + 2)
                hit_tgt = (r["high"] >= s.target_level) if s.side > 0 else (r["low"] <= s.target_level)
                if hit_stop or hit_tgt:
                    reason, t_exit = ("stop" if hit_stop else "target"), t
                    break
        elif exit_style.startswith("time"):
            t_exit = s.t_entry + pd.Timedelta(minutes=int(exit_style[4:]))
        if exit_mid is None:
            bb = b[b.index >= t_exit] if t_exit is not None else b.iloc[[-1]]
            if bb.empty:
                bb = b.iloc[[-1]]
            exit_mid = float(bb["open"].iloc[0])
            t_exit = bb.index[0]
        exit_px = max(0.05, exit_mid - half_spread(exit_mid) - slippage)
        ch = (order_charges("option", "BUY", entry * lot, rates)["total"]
              + order_charges("option", "SELL", exit_px * lot, rates)["total"])
        gross = (exit_px - entry) * lot
        rows.append({"day": s.day.isoformat(), "t_entry": s.t_entry, "t_exit": t_exit, "side": s.side,
                     "trigger": s.trigger, "level": s.level_kind, "dte": (e - s.day).days, "strike": strike,
                     "entry": round(entry, 2), "exit": round(exit_px, 2), "reason": reason,
                     "gross": round(gross, 2), "charges": round(ch, 2), "net": round(gross - ch, 2)})
    return pd.DataFrame(rows)


def index_outcomes(setups: List[Setup], n1: pd.DataFrame, minutes: int = 15) -> pd.DataFrame:
    """Direction accuracy on the index itself (no option data needed): did NIFTY move in the
    trade's favour over the next `minutes`, and did structure target come before structure stop?"""
    rows = []
    for s in setups:
        u = n1.loc[s.t_entry:]
        u = u[(u.index.date == s.day) & (u.index.time <= T_EXIT)]
        if len(u) < 2:
            continue
        p0 = float(u["open"].iloc[0])
        k = min(len(u) - 1, minutes)
        fwd = float(u["close"].iloc[k]) - p0
        res = "time"
        for _, r in u.iloc[1:].iterrows():
            st = (r["low"] <= s.sweep_extreme - 2) if s.side > 0 else (r["high"] >= s.sweep_extreme + 2)
            tg = (r["high"] >= s.target_level) if s.side > 0 else (r["low"] <= s.target_level)
            if st:
                res = "stop"; break
            if tg:
                res = "target"; break
        rows.append({"day": s.day, "side": s.side, "trigger": s.trigger, "level": s.level_kind,
                     "fwd_pts": s.side * fwd, "struct_result": res,
                     "risk_pts": abs(p0 - s.sweep_extreme), "reward_pts": abs(s.target_level - p0)})
    return pd.DataFrame(rows)
