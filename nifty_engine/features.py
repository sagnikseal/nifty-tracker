"""Daily feature engineering with explicit information-timing rules.

Decision time for target "oc" is ~09:20 IST on day t. Allowed information:
  * NIFTY / India VIX / BankNifty: full bars up to t-1, plus NIFTY's *open* on t.
  * US, Europe, commodities, FX, US yields: bars dated <= t-1 (they close before 09:15 IST).
  * Japan / Hong Kong: *opening gap* on day t (Tokyo opens 05:30 IST, HK 07:00 IST);
    their full-day return only with a one-day lag.
For target "cc" (close_t -> close_t+1, decided at ~15:20 on t) NIFTY's own bar for t is
also allowed, everything else keeps the same lags (conservative).

Every feature is built by shifting, never by looking forward. `tests/test_leakage.py`
verifies this by perturbing future data and checking features don't change.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from .calendar import days_to_expiry, is_expiry_day

LAGGED_GLOBAL = ["spx", "ndx", "usvix", "dxy", "us10y", "brent", "gold", "usdinr"]
SAME_DAY_OPEN = ["n225", "hsi"]


def _gk_vol(df: pd.DataFrame) -> pd.Series:
    """Garman-Klass daily volatility (in log-return units)."""
    hl = np.log(df["high"] / df["low"]) ** 2
    co = np.log(df["close"] / df["open"]) ** 2
    var = 0.5 * hl - (2 * np.log(2) - 1) * co
    return np.sqrt(var.clip(lower=1e-10))


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def _align_backward(target_index: pd.DatetimeIndex, s: pd.Series, avail_lag_days: int) -> pd.Series:
    """Align series `s` (indexed by its own trading date) onto NIFTY dates, using the last
    observation that became available on/before each NIFTY date. availability = date + lag."""
    if s is None or s.empty:
        return pd.Series(np.nan, index=target_index)
    s = s.dropna().copy()
    s.index = pd.DatetimeIndex(s.index) + pd.Timedelta(days=avail_lag_days)
    left = pd.DataFrame({"date": target_index})
    right = pd.DataFrame({"date": s.index, "v": s.values}).sort_values("date")
    m = pd.merge_asof(left.sort_values("date"), right, on="date", direction="backward",
                      tolerance=pd.Timedelta(days=6))
    return pd.Series(m["v"].values, index=target_index)


def build_daily_features(raw: Dict[str, pd.DataFrame], target: str = "oc",
                         extra: Optional[Dict[str, pd.Series]] = None, range_k: float = 1.25) -> pd.DataFrame:
    """Return a frame indexed by NIFTY trading date with features, label columns and the
    day's OHLC (for the paper-trading replay). Labels: `y_ret`, `y_up`."""
    nf = raw["nifty"].copy().sort_index()
    nf = nf[~nf.index.duplicated(keep="last")]
    idx = nf.index
    o, h, l, c = nf["open"], nf["high"], nf["low"], nf["close"]
    lc = np.log(c)
    F = pd.DataFrame(index=idx)

    # "lag" = 1 for oc (decide at open of t, so yesterday's close is the latest close)
    # "lag" = 0 for cc (decide near close of t)
    lag = 0 if target == "cc" else 1     # "oc" and "range" decide at 09:20

    r = lc.diff()
    gk = _gk_vol(nf)
    F["gap"] = np.log(o / c.shift(1)) if target == "oc" else np.log(o / c.shift(1))
    if target == "cc":
        F["oc_today"] = np.log(c / o)
    for n in (1, 2, 5, 10, 20, 60):
        F[f"r{n}"] = (lc - lc.shift(n)).shift(lag)
    F["oc_lag1"] = np.log(c / o).shift(lag)
    F["on_lag1"] = np.log(o / c.shift(1)).shift(lag)
    F["rv_d"] = gk.shift(lag)
    F["rv_w"] = gk.rolling(5).mean().shift(lag)
    F["rv_m"] = gk.rolling(22).mean().shift(lag)
    F["rv_ratio"] = F["rv_w"] / F["rv_m"]
    F["clv_lag1"] = ((c - l) / (h - l).replace(0, np.nan)).shift(lag)
    for n in (20, 50, 200):
        F[f"dist_sma{n}"] = (np.log(c / c.rolling(n).mean()) / (gk.rolling(22).mean() * np.sqrt(n))).shift(lag)
    F["rsi14"] = _rsi(c).shift(lag)
    hi20, lo20 = h.rolling(20).max(), l.rolling(20).min()
    F["range_pos20"] = ((c - lo20) / (hi20 - lo20).replace(0, np.nan)).shift(lag)
    sign = np.sign(r)
    streak = sign.groupby((sign != sign.shift()).cumsum()).cumcount() + 1
    F["streak"] = (streak * sign).shift(lag)
    F["gap_z"] = F["gap"] / F["rv_m"]

    if "banknifty" in raw and len(raw["banknifty"]):
        bn = raw["banknifty"]["close"].reindex(idx)
        F["bn_rel_r1"] = (np.log(bn).diff() - r).shift(lag)
        F["bn_rel_r5"] = (np.log(bn).diff(5) - lc.diff(5)).shift(lag)

    if "indiavix" in raw and len(raw["indiavix"]):
        v = raw["indiavix"]["close"].reindex(idx).ffill(limit=3)
        lv = np.log(v)
        F["vix"] = v.shift(lag)
        F["vix_chg1"] = lv.diff().shift(lag)
        F["vix_chg5"] = lv.diff(5).shift(lag)
        F["vix_z60"] = ((v - v.rolling(60).mean()) / v.rolling(60).std()).shift(lag)
        F["vrp"] = (v / 100 / np.sqrt(252) - gk.rolling(5).mean()).shift(lag)

    # ---- global assets: one-session lag via availability = date + 1 calendar day
    for name in LAGGED_GLOBAL:
        df = raw.get(name)
        if df is None or df.empty:
            continue
        cl = df["close"]
        if name == "us10y":
            F["us10y_chg1"] = _align_backward(idx, cl.diff(), 1)
            F["us10y_chg5"] = _align_backward(idx, cl.diff(5), 1)
            continue
        if name == "usvix":
            F["usvix"] = _align_backward(idx, cl, 1)
            F["usvix_chg1"] = _align_backward(idx, np.log(cl).diff(), 1)
            continue
        F[f"{name}_r1"] = _align_backward(idx, np.log(cl).diff(), 1)
        F[f"{name}_r5"] = _align_backward(idx, np.log(cl).diff(5), 1)

    # ---- Asia: same-day opening gap (known before 09:15 IST), full-day return lagged
    for name in SAME_DAY_OPEN:
        df = raw.get(name)
        if df is None or df.empty:
            continue
        gap = np.log(df["open"] / df["close"].shift(1))
        F[f"{name}_gap"] = _align_backward(idx, gap, 0)
        F[f"{name}_r1"] = _align_backward(idx, np.log(df["close"]).diff(), 1)

    # ---- optional forward-collected series (participant OI, news tone ...), lagged 1 day
    for name, s in (extra or {}).items():
        F[name] = _align_backward(idx, s, 1)

    # ---- calendar
    dates = idx.date
    F["dow"] = idx.dayofweek
    F["is_expiry"] = [int(is_expiry_day(d)) for d in dates]
    F["dte"] = [days_to_expiry(d) for d in dates]
    F["month_turn"] = ((idx.day >= 27) | (idx.day <= 3)).astype(int)
    # Turn-of-month (TD-2..TD+3) is real for close-to-close NIFTY returns (see docs 4.8) but added
    # nothing to this open->close model in walk-forward tests, so it is not a feature here.

    # ---- labels
    if target == "range":
        # 1 = NIFTY never moved more than range_k x (VIX-implied daily sigma) from the open all day
        vprev = raw["indiavix"]["close"].reindex(idx).ffill(limit=3).shift(1) if "indiavix" in raw else None
        sig = vprev / 100 / np.sqrt(252)
        exc = np.maximum(np.log(h / o), -np.log(l / o))
        F["y_ret"] = pd.Series(1.0, index=idx).where(sig.notna() & exc.notna())
        F["y_up"] = (exc < range_k * sig).astype(float).where(sig.notna() & exc.notna())
    else:
        y = np.log(c / o) if target == "oc" else lc.shift(-1) - lc
        F["y_ret"] = y
        F["y_up"] = (y > 0).astype(float).where(y.notna())
    for col in ("open", "high", "low", "close"):
        F[f"px_{col}"] = nf[col]
    F["px_prev_close"] = c.shift(1)
    if "indiavix" in raw and len(raw["indiavix"]):
        F["px_vix_prev"] = raw["indiavix"]["close"].reindex(idx).ffill(limit=3).shift(1)
    return F.replace([np.inf, -np.inf], np.nan)


def feature_columns(F: pd.DataFrame) -> list:
    return [c for c in F.columns if not c.startswith(("y_", "px_"))]


def add_price_structure_features(F: pd.DataFrame, nifty: pd.DataFrame, target: str = "oc") -> pd.DataFrame:
    """Legacy 'pillar 1' daily structure (CPR, PDH/PDL, sweeps, BOS, FVG, inside/outside days),
    re-expressed with strict timing: for target 'oc' only bars <= t-1 plus today's open are used."""
    nf = nifty.reindex(F.index)
    o, h, l, c = nf["open"], nf["high"], nf["low"], nf["close"]
    lag = 0 if target == "cc" else 1     # "oc" and "range" decide at 09:20
    H, L, C = h.shift(lag), l.shift(lag), c.shift(lag)          # "previous day" relative to decision
    H2, L2, C2 = h.shift(lag + 1), l.shift(lag + 1), c.shift(lag + 1)
    atr = (h - l).rolling(14).mean().shift(lag)
    ref = o if target == "oc" else c                            # price at decision time
    pivot = (H + L + C) / 3
    bc = (H + L) / 2
    tc = 2 * pivot - bc
    top, bot = np.maximum(tc, bc), np.minimum(tc, bc)
    X = pd.DataFrame(index=F.index)
    X["cpr_width"] = (top - bot) / pivot
    X["cpr_width_z"] = (X["cpr_width"] - X["cpr_width"].rolling(60).mean()) / X["cpr_width"].rolling(60).std()
    X["ref_vs_cpr"] = np.where(ref > top, 1.0, np.where(ref < bot, -1.0, 0.0))
    X["ref_vs_pivot_atr"] = (ref - pivot) / atr
    X["ref_vs_pdh_atr"] = (ref - H) / atr
    X["ref_vs_pdl_atr"] = (ref - L) / atr
    X["ref_vs_r1_atr"] = (ref - (2 * pivot - L)) / atr
    X["ref_vs_s1_atr"] = (ref - (2 * pivot - H)) / atr
    X["inside_day"] = ((H < H2) & (L > L2)).astype(float)
    X["outside_day"] = ((H > H2) & (L < L2)).astype(float)
    X["pdh_sweep"] = ((H > H2) & (C < H2)).astype(float)
    X["pdl_sweep"] = ((L < L2) & (C > L2)).astype(float)
    hh = h.shift(lag + 1).rolling(5).max()
    ll = l.shift(lag + 1).rolling(5).min()
    X["bos"] = np.where(C > hh, 1.0, np.where(C < ll, -1.0, 0.0))
    X["fvg"] = np.where(L > h.shift(lag + 2), 1.0, np.where(H < l.shift(lag + 2), -1.0, 0.0))
    wk = nifty[["high", "low"]].resample("W-FRI").agg({"high": "max", "low": "min"})
    wk.index = wk.index + pd.Timedelta(days=1)   # weekly bar available from the next Saturday on
    pwh = _align_backward(F.index, wk["high"], 0)
    pwl = _align_backward(F.index, wk["low"], 0)
    X["ref_vs_pwh_atr"] = (ref - pwh) / atr
    X["ref_vs_pwl_atr"] = (ref - pwl) / atr
    X["body_lag1"] = ((C - o.shift(lag)) / atr)
    X["upper_wick_lag1"] = (H - np.maximum(C, o.shift(lag))) / atr
    X["lower_wick_lag1"] = (np.minimum(C, o.shift(lag)) - L) / atr
    return F.join(X.replace([np.inf, -np.inf], np.nan))
