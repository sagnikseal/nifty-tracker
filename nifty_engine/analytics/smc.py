"""Smart Money Concepts (SMC) price structure, computed causally.

Row t of every output uses only bars 0..t, i.e. what is known at the *close* of bar t.

Why this matters: a swing high at bar i needs `k` bars on each side, so it is only
*confirmed* at bar i+k. Most public SMC code (including the popular `smartmoneyconcepts`
package) uses centered windows, which mark swings k bars before anyone could see them and
make backtests look far better than live trading. Here a swing is used from bar i+k onward.

Concepts implemented (one consistent, mechanical definition each, so they can be tested):
  * swing high / low      : high[i] is the max of high[i-k..i+k] (strictly above the left side)
  * BOS / CHoCH           : close beyond the last unbroken confirmed swing; BOS if it continues
                            the current structure trend, CHoCH if it flips it
  * order block (OB)      : on a bullish break, the last bearish candle since the swing low that
                            started the leg (mirror for bearish); valid until a close beyond it
  * fair value gap (FVG)  : 3-candle imbalance, low[t] > high[t-2] (bullish) or high[t] < low[t-2];
                            live until price trades back through the far edge
  * liquidity sweep       : wick beyond a liquidity level (unbroken swing, equal highs/lows or an
                            external level such as the previous day's high/low) with the close
                            back inside
  * equal highs / lows    : two consecutive confirmed swings within `eq_tol` x ATR
  * premium / discount    : position of the close inside the current dealing range
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


def true_range(df: pd.DataFrame) -> pd.Series:
    pc = df["close"].shift(1)
    return pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()],
                     axis=1).max(axis=1)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Wilder ATR (exponential, so it is defined from the first bar)."""
    return true_range(df).ewm(alpha=1 / n, adjust=False).mean()


def confirmed_swings(high: np.ndarray, low: np.ndarray, k: int):
    """Swings indexed by the bar on which they become known (i + k).

    Returns (sh, sh_origin, sl, sl_origin): price arrays that are NaN except on confirmation
    bars, and the index of the bar that made the swing."""
    if k < 1:
        raise ValueError("k must be >= 1")
    n = len(high)
    sh, sl = np.full(n, np.nan), np.full(n, np.nan)
    sh_i, sl_i = np.full(n, -1), np.full(n, -1)
    for i in range(k, n - k):
        hw, lw = high[i - k:i + k + 1], low[i - k:i + k + 1]
        if high[i] >= hw.max() and high[i] > high[i - k:i].max():
            sh[i + k], sh_i[i + k] = high[i], i
        if low[i] <= lw.min() and low[i] < low[i - k:i].min():
            sl[i + k], sl_i[i + k] = low[i], i
    return sh, sh_i, sl, sl_i


@dataclass
class Zone:
    top: float
    bot: float
    born: int
    dead: bool = False
    touched: bool = False


def _nearest_support(zones, price):
    """Highest zone whose bottom is at/below price (price may be inside it)."""
    best = None
    for z in zones:
        if z.bot <= price and (best is None or z.top > best.top):
            best = z
    return best


def _nearest_resistance(zones, price):
    best = None
    for z in zones:
        if z.top >= price and (best is None or z.bot < best.bot):
            best = z
    return best


def smc_features(df: pd.DataFrame, k: int = 3, atr_n: int = 14, eq_tol: float = 0.15,
                 max_zones: int = 6, ob_lookback: int = 60,
                 ext_high: Optional[pd.Series] = None, ext_low: Optional[pd.Series] = None) -> pd.DataFrame:
    """SMC state per bar. `ext_high` / `ext_low` are optional external liquidity levels known at
    each bar (e.g. previous day's high/low), aligned to df.index."""
    o, h, l, c = (df[x].to_numpy(dtype=float) for x in ("open", "high", "low", "close"))
    n = len(df)
    a = atr(df, atr_n).to_numpy()
    eh = ext_high.reindex(df.index).to_numpy(dtype=float) if ext_high is not None else np.full(n, np.nan)
    el = ext_low.reindex(df.index).to_numpy(dtype=float) if ext_low is not None else np.full(n, np.nan)
    sh, sh_o, sl, sl_o = confirmed_swings(h, l, k)

    cols = ["ms_trend", "bos", "choch", "sweep", "sweep_ext", "fvg_new", "swing_hi", "swing_lo", "pd_pos",
            "bull_ob_top", "bull_ob_bot", "bull_ob_born", "bear_ob_top", "bear_ob_bot", "bear_ob_born",
            "bull_fvg_top", "bull_fvg_bot", "bull_fvg_born", "bear_fvg_top", "bear_fvg_bot", "bear_fvg_born",
            "in_bull_ob", "in_bear_ob", "in_bull_fvg", "in_bear_fvg", "n_bull_fvg", "n_bear_fvg",
            "eqh_level", "eql_level", "bars_since_bull_sweep", "bars_since_bear_sweep",
            "bars_since_bos_up", "bars_since_bos_dn"]
    out = {cn: np.full(n, np.nan) for cn in cols}

    trend = 0
    last_sh = last_sl = np.nan
    last_sh_o = last_sl_o = -1
    sh_live = sl_live = False          # an unbroken confirmed swing exists
    bull_ob, bear_ob, bull_fvg, bear_fvg = [], [], [], []
    eqh, eql = [], []                  # unswept equal-high / equal-low pools
    t_bull_sweep = t_bear_sweep = t_up = t_dn = -10 ** 9

    for t in range(n):
        at = a[t] if a[t] > 0 else np.nan
        # 1) swings confirmed on this bar
        if not np.isnan(sh[t]):
            if sh_live and not np.isnan(at) and abs(sh[t] - last_sh) <= eq_tol * at:
                eqh.append(max(sh[t], last_sh))
            last_sh, last_sh_o, sh_live = sh[t], sh_o[t], True
        if not np.isnan(sl[t]):
            if sl_live and not np.isnan(at) and abs(sl[t] - last_sl) <= eq_tol * at:
                eql.append(min(sl[t], last_sl))
            last_sl, last_sl_o, sl_live = sl[t], sl_o[t], True

        # 2) liquidity sweeps: wick through, close back inside
        up_levels = ([last_sh] if sh_live else []) + eqh
        dn_levels = ([last_sl] if sl_live else []) + eql
        swept_up = any(h[t] > L and c[t] < L for L in up_levels)
        swept_dn = any(l[t] < L and c[t] > L for L in dn_levels)
        if swept_up:
            eqh = [L for L in eqh if not h[t] > L]
        if swept_dn:
            eql = [L for L in eql if not l[t] < L]
        sweep = (1 if swept_dn else 0) - (1 if swept_up else 0)
        ext = 0
        if not np.isnan(el[t]) and l[t] < el[t] and c[t] > el[t]:
            ext += 1
        if not np.isnan(eh[t]) and h[t] > eh[t] and c[t] < eh[t]:
            ext -= 1
        if sweep > 0 or ext > 0:
            t_bull_sweep = t
        if sweep < 0 or ext < 0:
            t_bear_sweep = t

        # 3) structure breaks on the close
        bos = choch = 0
        if sh_live and c[t] > last_sh:
            bos, choch = (1, 0) if trend >= 0 else (0, 1)
            trend, sh_live, t_up = 1, False, t
            start = max(last_sl_o if last_sl_o >= 0 else 0, t - ob_lookback)
            for j in range(t - 1, start - 1, -1):
                if c[j] < o[j]:
                    bull_ob.append(Zone(h[j], l[j], t))
                    break
        if sl_live and c[t] < last_sl:
            bos, choch = (-1, 0) if trend <= 0 else (0, -1)
            trend, sl_live, t_dn = -1, False, t
            start = max(last_sh_o if last_sh_o >= 0 else 0, t - ob_lookback)
            for j in range(t - 1, start - 1, -1):
                if c[j] > o[j]:
                    bear_ob.append(Zone(h[j], l[j], t))
                    break

        # 4) FVGs created by this bar
        fvg_new = 0
        if t >= 2 and l[t] > h[t - 2]:
            bull_fvg.append(Zone(l[t], h[t - 2], t))
            fvg_new = 1
        if t >= 2 and h[t] < l[t - 2]:
            bear_fvg.append(Zone(l[t - 2], h[t], t))
            fvg_new = -1

        # 5) mitigation / invalidation by this bar (zones born now are untouched by it)
        for z in bull_ob:
            if z.born < t:
                z.touched |= l[t] <= z.top
                z.dead |= c[t] < z.bot
        for z in bear_ob:
            if z.born < t:
                z.touched |= h[t] >= z.bot
                z.dead |= c[t] > z.top
        for z in bull_fvg:
            if z.born < t:
                z.touched |= l[t] < z.top
                z.dead |= l[t] <= z.bot
        for z in bear_fvg:
            if z.born < t:
                z.touched |= h[t] > z.bot
                z.dead |= h[t] >= z.top
        bull_ob = [z for z in bull_ob if not z.dead][-max_zones:]
        bear_ob = [z for z in bear_ob if not z.dead][-max_zones:]
        bull_fvg = [z for z in bull_fvg if not z.dead][-max_zones:]
        bear_fvg = [z for z in bear_fvg if not z.dead][-max_zones:]

        # 6) record state
        px = c[t]
        out["ms_trend"][t], out["bos"][t], out["choch"][t] = trend, bos, choch
        out["sweep"][t], out["sweep_ext"][t], out["fvg_new"][t] = sweep, ext, fvg_new
        out["swing_hi"][t] = last_sh
        out["swing_lo"][t] = last_sl
        if not np.isnan(last_sh) and not np.isnan(last_sl) and last_sh > last_sl:
            out["pd_pos"][t] = (px - last_sl) / (last_sh - last_sl)
        for name, zs, finder in (("bull_ob", bull_ob, _nearest_support), ("bear_ob", bear_ob, _nearest_resistance),
                                 ("bull_fvg", bull_fvg, _nearest_support), ("bear_fvg", bear_fvg, _nearest_resistance)):
            z = finder(zs, px)
            if z is not None:
                out[f"{name}_top"][t], out[f"{name}_bot"][t], out[f"{name}_born"][t] = z.top, z.bot, z.born
                out[f"in_{name}"][t] = float(z.bot <= px <= z.top or (l[t] <= z.top and h[t] >= z.bot))
            else:
                out[f"in_{name}"][t] = 0.0
        out["n_bull_fvg"][t], out["n_bear_fvg"][t] = len(bull_fvg), len(bear_fvg)
        above = [L for L in eqh if L >= px]
        below = [L for L in eql if L <= px]
        out["eqh_level"][t] = min(above) if above else np.nan
        out["eql_level"][t] = max(below) if below else np.nan
        out["bars_since_bull_sweep"][t] = t - t_bull_sweep if t_bull_sweep >= 0 else np.nan
        out["bars_since_bear_sweep"][t] = t - t_bear_sweep if t_bear_sweep >= 0 else np.nan
        out["bars_since_bos_up"][t] = t - t_up if t_up >= 0 else np.nan
        out["bars_since_bos_dn"][t] = t - t_dn if t_dn >= 0 else np.nan

    res = pd.DataFrame(out, index=df.index)
    res["atr"] = a
    return res


def distance_features(S: pd.DataFrame, ref: pd.Series, prefix: str = "smc_") -> pd.DataFrame:
    """Turn the level columns of `smc_features` into ATR-scaled distances from a reference price
    (e.g. today's open, known at decision time). Positive = ref above the level."""
    a = S["atr"].replace(0, np.nan)
    X = pd.DataFrame(index=S.index)
    X[f"{prefix}d_bull_ob"] = (ref - S["bull_ob_top"]) / a
    X[f"{prefix}d_bear_ob"] = (S["bear_ob_bot"] - ref) / a
    X[f"{prefix}d_bull_fvg"] = (ref - S["bull_fvg_top"]) / a
    X[f"{prefix}d_bear_fvg"] = (S["bear_fvg_bot"] - ref) / a
    X[f"{prefix}d_swing_hi"] = (S["swing_hi"] - ref) / a
    X[f"{prefix}d_swing_lo"] = (ref - S["swing_lo"]) / a
    X[f"{prefix}d_eqh"] = (S["eqh_level"] - ref) / a
    X[f"{prefix}d_eql"] = (ref - S["eql_level"]) / a
    return X.clip(-20, 20)


def smc_setups(df: pd.DataFrame, S: Optional[pd.DataFrame] = None, window: int = 12, rr: float = 2.0,
               min_stop_atr: float = 0.25, session: Optional[pd.Series] = None,
               start_time=None, end_time=None, one_per_session: bool = True, **smc_kw) -> pd.DataFrame:
    """Mechanical 'liquidity sweep -> structure break -> retrace into the new imbalance' setup.

    Long: a bullish sweep at bar s; a bullish BOS/CHoCH at bar b in (s, s+window]; then the first
    bar r in (b, b+window] that trades into a bullish FVG or OB born at/after s. The signal is
    known at the close of r and executes at the open of bar r+1. Stop: lowest low from s to r
    minus 0.1 ATR. Target: entry + rr x risk. Shorts mirror this. All inputs are causal."""
    S = S if S is not None else smc_features(df, **smc_kw)
    o, h, l = df["open"].to_numpy(float), df["high"].to_numpy(float), df["low"].to_numpy(float)
    idx = df.index
    sess = session.reindex(idx).to_numpy() if session is not None else np.zeros(len(df))
    times = idx.time if hasattr(idx, "time") else None
    rows, used = [], set()
    n = len(df)
    for direction in (1, -1):
        sw = S["bars_since_bull_sweep" if direction > 0 else "bars_since_bear_sweep"].to_numpy()
        brk = S["bos"].to_numpy() + S["choch"].to_numpy()
        in_fvg = S["in_bull_fvg" if direction > 0 else "in_bear_fvg"].to_numpy()
        in_ob = S["in_bull_ob" if direction > 0 else "in_bear_ob"].to_numpy()
        fvg_born = S["bull_fvg_born" if direction > 0 else "bear_fvg_born"].to_numpy()
        ob_born = S["bull_ob_born" if direction > 0 else "bear_ob_born"].to_numpy()
        a = S["atr"].to_numpy()
        state = None   # (sweep_bar, break_bar)
        for t in range(1, n - 1):
            if sw[t] == 0:
                state = (t, None)
                continue
            if state is None:
                continue
            s, b = state
            if sess[t] != sess[s] or t - s > 2 * window:
                state = None
                continue
            if b is None:
                if np.sign(brk[t]) == direction and t - s <= window:
                    state = (s, t)
                continue
            if t - b > window:
                state = None
                continue
            zone_ok = (in_fvg[t] == 1 and fvg_born[t] >= s) or (in_ob[t] == 1 and ob_born[t] >= s)
            if not zone_ok:
                continue
            if times is not None and ((start_time and times[t] < start_time) or (end_time and times[t] > end_time)):
                continue
            key = (sess[t], direction) if one_per_session else t
            if key in used:
                continue
            entry = o[t + 1]
            stop = (np.min(l[s:t + 1]) - 0.1 * a[t]) if direction > 0 else (np.max(h[s:t + 1]) + 0.1 * a[t])
            risk = (entry - stop) * direction
            if not np.isfinite(risk) or risk < min_stop_atr * a[t]:
                continue
            used.add(key)
            rows.append({"ts_signal": idx[t], "ts_entry": idx[t + 1], "direction": direction,
                         "entry": float(entry), "stop": float(stop), "target": float(entry + direction * rr * risk),
                         "risk_pts": float(risk), "sweep_ts": idx[s], "break_ts": idx[b]})
            state = None
    return pd.DataFrame(rows).sort_values("ts_signal").reset_index(drop=True) if rows else pd.DataFrame(
        columns=["ts_signal", "ts_entry", "direction", "entry", "stop", "target", "risk_pts", "sweep_ts", "break_ts"])
