"""Volatility forecast and market regime, causal.

HAR-RV (Corsi 2009): log RV_t+1 = b0 + bd log RV_d + bw log RV_w + bm log RV_m, re-fitted on an
expanding window. Volatility, unlike direction, is genuinely forecastable, so it drives
(a) which strategy family fits the day (buy options when the expected move beats what is priced,
sell when the market overprices it) and (b) position size.
"""

from __future__ import annotations

import bisect
import math

import numpy as np
import pandas as pd

from .smc import atr
from .technical import efficiency_ratio


def _gk(df: pd.DataFrame) -> pd.Series:
    hl = np.log(df["high"] / df["low"]) ** 2
    co = np.log(df["close"] / df["open"]) ** 2
    return np.sqrt((0.5 * hl - (2 * np.log(2) - 1) * co).clip(lower=1e-10))


def har_forecast(daily: pd.DataFrame, min_obs: int = 400, refit_every: int = 21) -> pd.Series:
    """Forecast of day t's Garman-Klass vol (log-return units) made at the close of t-1."""
    rv = _gk(daily)
    lr = np.log(rv)
    X = pd.DataFrame({"d": lr, "w": np.log(rv.rolling(5).mean()), "m": np.log(rv.rolling(22).mean())})
    y = lr.shift(-1)                                  # row t: regressors at t, target at t+1
    out = pd.Series(np.nan, index=daily.index)
    beta = None
    n = len(daily)
    for t in range(n):
        if t >= min_obs and (beta is None or t % refit_every == 0):
            tr = pd.concat([X.iloc[:t - 1], y.iloc[:t - 1]], axis=1).dropna()   # targets <= t-1
            if len(tr) > 50:
                A = np.c_[np.ones(len(tr)), tr[["d", "w", "m"]].to_numpy()]
                beta = np.linalg.lstsq(A, tr.iloc[:, -1].to_numpy(), rcond=None)[0]
        if beta is not None and t >= 1:
            x = X.iloc[t - 1]
            if x.notna().all():
                out.iloc[t] = math.exp(beta[0] + beta[1:] @ x.to_numpy())
    return out


def expanding_percentile(s: pd.Series, min_obs: int = 250) -> pd.Series:
    """Percentile of each value within all *earlier* values (no look-ahead)."""
    seen, out = [], []
    for v in s.to_numpy():
        if v == v and len(seen) >= min_obs:
            out.append(bisect.bisect_left(seen, v) / len(seen))
        else:
            out.append(np.nan)
        if v == v:
            bisect.insort(seen, v)
    return pd.Series(out, index=s.index)


def regime_features(daily: pd.DataFrame, vix: pd.Series | None = None, lag: int = 1) -> pd.DataFrame:
    """Daily regime features for a decision on day t (lag=1: uses bars <= t-1)."""
    fc = har_forecast(daily)                          # already a forecast *for* day t
    X = pd.DataFrame(index=daily.index)
    X["har_vol_fc"] = fc if lag == 1 else fc.shift(-1)
    X["har_vol_pct"] = expanding_percentile(X["har_vol_fc"])
    c = daily["close"]
    X["er20"] = efficiency_ratio(c, 20).shift(lag)
    X["er60"] = efficiency_ratio(c, 60).shift(lag)
    X["atr_pct"] = (atr(daily, 14) / c).shift(lag)
    if vix is not None:
        v = vix.reindex(daily.index).ffill(limit=3).shift(lag)
        implied = v / 100 / math.sqrt(252)
        X["vrp_har"] = implied - X["har_vol_fc"]          # >0: options price more than HAR expects
        X["vix_pct_1y"] = v.rolling(250, min_periods=120).rank(pct=True)
    X["trend_regime"] = np.sign(np.log(c / c.rolling(50).mean())).shift(lag) * (X["er20"] > 0.3)
    return X
