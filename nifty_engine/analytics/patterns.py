"""Technical-analysis and pattern features for the movement model, all strictly causal.

Input: a continuous, start-stamped 5-minute OHLC(V) frame over many sessions. Output: one row per
5-minute bar (same index) whose values use only that bar and earlier ones; the decision at
T = bar start + 5 minutes may use the row of its last closed bar. 15- and 60-minute versions are
built from completed higher-timeframe bars only (a 15-minute bar is usable once its last 5-minute
bar has closed).

Every distance is expressed in units of ATR (or as a bounded oscillator), so the model sees the
same scale in 2022 (NIFTY ~17,000) and 2026 (~25,000). Groups (for ablation tests):
  tech      trend / momentum / volatility indicators on 5-minute bars
  tech_mtf  the same family on 15- and 60-minute bars (multi-timeframe alignment)
  candles   candlestick and bar patterns (engulfing, hammer, doji, inside/outside, NR7, Heikin-Ashi ...)
  structure swing structure (higher highs / lower lows, breaks), Donchian breakouts, round-number levels
Evidence note: studies of minute-level data find raw price-derived inputs carry most of the
signal and classic indicators add little on their own; their value, if any, is in combinations a
tree model can find, which is why every group is tested against the base model before use.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .smc import atr as _atr
from .technical import adx as _adx, ema, rsi, supertrend


def _safe(x: pd.Series, a: pd.Series) -> pd.Series:
    return x / a.replace(0, np.nan)


def _di(df: pd.DataFrame, n: int = 14):
    h, l = df["high"], df["low"]
    up, dn = h.diff(), -l.diff()
    a = _atr(df, n).replace(0, np.nan)
    pdi = 100 * pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index).ewm(alpha=1 / n, adjust=False).mean() / a
    mdi = 100 * pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index).ewm(alpha=1 / n, adjust=False).mean() / a
    return pdi, mdi


def indicator_block(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Trend, momentum, volatility and oscillator family on one timeframe."""
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    a = _atr(df, 14)
    X = pd.DataFrame(index=df.index)
    e9, e21, e50 = ema(c, 9), ema(c, 21), ema(c, 50)
    X[f"{prefix}ema9_dist"] = _safe(c - e9, a)
    X[f"{prefix}ema21_dist"] = _safe(c - e21, a)
    X[f"{prefix}ema50_dist"] = _safe(c - e50, a)
    X[f"{prefix}ema9_21"] = _safe(e9 - e21, a)
    X[f"{prefix}ema21_slope"] = _safe(e21 - e21.shift(3), a)
    X[f"{prefix}ema_align"] = np.sign(e9 - e21) + np.sign(e21 - e50)
    m = ema(c, 12) - ema(c, 26)
    sig = ema(m, 9)
    X[f"{prefix}macd"] = _safe(m, a)
    X[f"{prefix}macd_hist"] = _safe(m - sig, a)
    X[f"{prefix}macd_hist_d"] = X[f"{prefix}macd_hist"].diff()
    X[f"{prefix}rsi14"] = rsi(c, 14)
    X[f"{prefix}rsi5"] = rsi(c, 5)
    X[f"{prefix}adx"] = _adx(df, 14)
    pdi, mdi = _di(df, 14)
    X[f"{prefix}di_diff"] = pdi - mdi
    st = supertrend(df, 10, 3.0)
    X[f"{prefix}st_dir"] = st["st_dir"]
    X[f"{prefix}st_dist"] = _safe(c - st["st"], a)
    ma20, sd20 = c.rolling(20).mean(), c.rolling(20).std()
    X[f"{prefix}bb_pctb"] = _safe(c - (ma20 - 2 * sd20), 4 * sd20)
    bw = _safe(4 * sd20, ma20)
    X[f"{prefix}bb_width_z"] = (bw - bw.rolling(100).mean()) / bw.rolling(100).std()
    kc = 1.5 * a.rolling(20).mean()
    X[f"{prefix}squeeze"] = ((2 * sd20) < kc).astype(float)              # TTM squeeze: Bollinger inside Keltner
    ll, hh = l.rolling(14).min(), h.rolling(14).max()
    k = 100 * _safe(c - ll, hh - ll)
    X[f"{prefix}stoch_k"] = k.rolling(3).mean()
    X[f"{prefix}stoch_kd"] = X[f"{prefix}stoch_k"] - X[f"{prefix}stoch_k"].rolling(3).mean()
    tp = (h + l + c) / 3
    X[f"{prefix}cci"] = _safe(tp - tp.rolling(20).mean(), 0.015 * (tp - tp.rolling(20).mean()).abs().rolling(20).mean())
    X[f"{prefix}roc12"] = np.log(c / c.shift(12))
    X[f"{prefix}atr_pct"] = a / c
    X[f"{prefix}atr_ratio"] = _safe(a, a.rolling(100).mean())
    net = (c - c.shift(20)).abs()
    X[f"{prefix}efficiency"] = _safe(net, c.diff().abs().rolling(20).sum())
    r = np.log(c).diff()
    v1, v6 = r.rolling(48).var(), np.log(c).diff(6).rolling(48).var()
    X[f"{prefix}var_ratio"] = _safe(v6, 6 * v1)                          # >1 trending, <1 mean-reverting
    X[f"{prefix}ret_skew"] = r.rolling(48).skew()
    tenkan = (h.rolling(9).max() + l.rolling(9).min()) / 2
    kijun = (h.rolling(26).max() + l.rolling(26).min()) / 2
    span_a = ((tenkan + kijun) / 2).shift(26)
    span_b = ((h.rolling(52).max() + l.rolling(52).min()) / 2).shift(26)
    X[f"{prefix}ichi_tk"] = _safe(tenkan - kijun, a)
    top, bot = np.maximum(span_a, span_b), np.minimum(span_a, span_b)
    X[f"{prefix}ichi_cloud"] = np.where(c > top, 1.0, np.where(c < bot, -1.0, 0.0))
    if "volume" in df and df["volume"].fillna(0).sum() > 0:
        v = df["volume"].astype(float)
        X[f"{prefix}vol_z"] = (v - v.rolling(60).mean()) / v.rolling(60).std()
        obv = (np.sign(c.diff()).fillna(0) * v).cumsum()
        X[f"{prefix}obv_slope"] = _safe(obv - obv.shift(12), v.rolling(60).mean() * 12)
    return X.replace([np.inf, -np.inf], np.nan)


def candle_block(df: pd.DataFrame) -> pd.DataFrame:
    """Candlestick / bar patterns of the last bars (5-minute)."""
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    rng = (h - l).replace(0, np.nan)
    body = (c - o)
    ab = body.abs()
    upw, low_w = h - np.maximum(o, c), np.minimum(o, c) - l
    a = _atr(df, 14)
    X = pd.DataFrame(index=df.index)
    X["cdl_body"] = _safe(body, a)
    X["cdl_body_frac"] = ab / rng
    X["cdl_upper_wick"] = upw / rng
    X["cdl_lower_wick"] = low_w / rng
    X["cdl_doji"] = (ab / rng < 0.1).astype(float)
    X["cdl_hammer"] = ((low_w > 2 * ab) & (upw < ab) & (ab / rng < 0.4)).astype(float)
    X["cdl_shooting_star"] = ((upw > 2 * ab) & (low_w < ab) & (ab / rng < 0.4)).astype(float)
    X["cdl_marubozu"] = np.where(ab / rng > 0.9, np.sign(body), 0.0)
    po, pc = o.shift(1), c.shift(1)
    X["cdl_engulf"] = np.where((body > 0) & (pc < po) & (c >= po) & (o <= pc), 1.0,
                               np.where((body < 0) & (pc > po) & (c <= po) & (o >= pc), -1.0, 0.0))
    X["cdl_inside"] = ((h < h.shift(1)) & (l > l.shift(1))).astype(float)
    X["cdl_outside"] = ((h > h.shift(1)) & (l < l.shift(1))).astype(float)
    X["cdl_nr7"] = ((h - l) <= (h - l).rolling(7).min()).astype(float)
    up3 = (body > 0) & (body.shift(1) > 0) & (body.shift(2) > 0) & (c > c.shift(1)) & (c.shift(1) > c.shift(2))
    dn3 = (body < 0) & (body.shift(1) < 0) & (body.shift(2) < 0) & (c < c.shift(1)) & (c.shift(1) < c.shift(2))
    X["cdl_three"] = np.where(up3, 1.0, np.where(dn3, -1.0, 0.0))
    small_mid = ab.shift(1) < 0.3 * ab.shift(2)
    X["cdl_star"] = np.where(small_mid & (body.shift(2) < 0) & (body > 0) & (c > (o.shift(2) + c.shift(2)) / 2), 1.0,
                             np.where(small_mid & (body.shift(2) > 0) & (body < 0) & (c < (o.shift(2) + c.shift(2)) / 2), -1.0, 0.0))
    # Heikin-Ashi trend
    ha_c = (o + h + l + c) / 4
    ha_o = ha_c.copy()
    vals_o, vals_c = o.to_numpy(float), ha_c.to_numpy(float)
    hao = np.empty(len(o))
    hao[0] = vals_o[0]
    for i in range(1, len(o)):
        hao[i] = (hao[i - 1] + vals_c[i - 1]) / 2
    ha_o = pd.Series(hao, index=df.index)
    ha_dir = np.sign(ha_c - ha_o)
    run = ha_dir.groupby((ha_dir != ha_dir.shift()).cumsum()).cumcount() + 1
    X["ha_dir"] = ha_dir
    X["ha_run"] = run * ha_dir
    return X.replace([np.inf, -np.inf], np.nan)


def structure_block(df: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    """Swing structure, Donchian breakouts, round numbers. Swings are confirmed only k bars later."""
    h, l, c = df["high"].to_numpy(float), df["low"].to_numpy(float), df["close"].to_numpy(float)
    n = len(c)
    a = _atr(df, 14).to_numpy()
    last_hi, prev_hi, last_lo, prev_lo = np.nan, np.nan, np.nan, np.nan
    hh = np.zeros(n)
    hl = np.zeros(n)
    sh_dist = np.full(n, np.nan)
    sl_dist = np.full(n, np.nan)
    for i in range(n):
        j = i - k                     # bar j is confirmed as a swing once bars j+1..j+k exist (i = j + k)
        if j - k >= 0:
            if h[j] == h[j - k:j + k + 1].max():
                prev_hi, last_hi = last_hi, h[j]
            if l[j] == l[j - k:j + k + 1].min():
                prev_lo, last_lo = last_lo, l[j]
        hh[i] = 0 if np.isnan(prev_hi) else np.sign(last_hi - prev_hi)
        hl[i] = 0 if np.isnan(prev_lo) else np.sign(last_lo - prev_lo)
        if a[i] and a[i] == a[i]:
            sh_dist[i] = (c[i] - last_hi) / a[i] if last_hi == last_hi else np.nan
            sl_dist[i] = (c[i] - last_lo) / a[i] if last_lo == last_lo else np.nan
    X = pd.DataFrame(index=df.index)
    X["sw_hh"] = hh                      # +1 higher high, -1 lower high
    X["sw_hl"] = hl                      # +1 higher low,  -1 lower low
    X["sw_trend"] = hh + hl              # +2 clean uptrend ... -2 clean downtrend
    X["sw_high_dist"] = sh_dist
    X["sw_low_dist"] = sl_dist
    cs = df["close"]
    a_s = _atr(df, 14)
    for nwin in (20, 60):
        dh, dl = df["high"].rolling(nwin).max(), df["low"].rolling(nwin).min()
        X[f"don{nwin}_pos"] = _safe(cs - dl, dh - dl)
        X[f"don{nwin}_break"] = np.where(cs >= dh.shift(1), 1.0, np.where(cs <= dl.shift(1), -1.0, 0.0))
    for step in (100, 500):
        nearest = (cs / step).round() * step
        X[f"round{step}_dist"] = _safe(cs - nearest, a_s)
    return X.replace([np.inf, -np.inf], np.nan)


def _resample_closed(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Higher-timeframe bars per session, stamped with the time they become available (bar end)."""
    out = []
    for _, g in df.groupby(df.index.date):
        r = g.resample(f"{minutes}min", origin="start_day", offset="15min", label="left", closed="left").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
        # bar end = start + minutes, but never later than the last 5-minute bar that went into it
        last5 = g["close"].resample(f"{minutes}min", origin="start_day", offset="15min", label="left",
                                    closed="left").apply(lambda s: s.index[-1] if len(s) else pd.NaT).dropna()
        r["avail"] = [pd.Timestamp(t) + pd.Timedelta(minutes=5) for t in last5.reindex(r.index)]
        out.append(r)
    return pd.concat(out) if out else pd.DataFrame()


def mtf_block(df: pd.DataFrame) -> pd.DataFrame:
    """15- and 60-minute indicators aligned to each 5-minute bar using only completed bars."""
    ends = df.index + pd.Timedelta(minutes=5)          # when each 5-minute bar becomes available
    frames = []
    for m, p in ((15, "m15_"), (60, "m60_")):
        r = _resample_closed(df, m)          # "avail" = when the bar's last 5-minute bar has closed
        if r.empty:
            continue
        I = indicator_block(r[["open", "high", "low", "close"]], p)[[f"{p}ema9_21", f"{p}ema21_dist", f"{p}macd_hist",
                                                                        f"{p}rsi14", f"{p}adx", f"{p}di_diff", f"{p}st_dir",
                                                                        f"{p}bb_pctb", f"{p}efficiency"]]
        I["t"] = pd.DatetimeIndex(pd.to_datetime(r["avail"])).tz_convert("UTC").tz_localize(None)
        I = I.dropna(subset=["t"]).sort_values("t")
        left = pd.DataFrame({"t": ends.tz_convert("UTC").tz_localize(None)})
        m_ = pd.merge_asof(left, I, on="t", direction="backward")
        frames.append(m_.drop(columns=["t"]).set_index(df.index))
    X = pd.concat(frames, axis=1) if frames else pd.DataFrame(index=df.index)
    if "m15_st_dir" in X and "m60_st_dir" in X:
        X["mtf_align"] = X["m15_st_dir"] + X["m60_st_dir"]
    return X


def tech_features(df5: pd.DataFrame, groups=("tech", "tech_mtf", "candles", "structure")) -> pd.DataFrame:
    """All requested groups for a continuous 5-minute frame; index = 5-minute bar start."""
    df5 = df5.sort_index()
    parts = []
    if "tech" in groups:
        parts.append(indicator_block(df5, "t5_"))
    if "tech_mtf" in groups:
        parts.append(mtf_block(df5))
    if "candles" in groups:
        parts.append(candle_block(df5))
    if "structure" in groups:
        parts.append(structure_block(df5))
    return pd.concat(parts, axis=1) if parts else pd.DataFrame(index=df5.index)
