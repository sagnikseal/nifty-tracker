"""Cross-asset intraday features at each decision time T. Every input is a bar that has closed
by T (5-minute bars: start + 5 min <= T; hourly global bars: end <= T).

Groups (each tested against the base model before use):
  banknifty   Bank Nifty move since its open / last 30 min, and relative to NIFTY (leadership, divergence)
  futures     near-month NIFTY futures: basis (premium over spot) and its change, true session VWAP from
              futures volume, volume surge vs the same time of day on the previous 20 sessions, open-interest
              change and the price/OI regime (long build-up, short build-up, short covering, long unwinding)
  breadth     heavyweight stocks: share above their own open, share rising over 30 min, leadership of the
              three largest (HDFC Bank, Reliance, ICICI Bank), dispersion
  global_intraday  US equity futures, USD/INR, crude, gold: moves since 09:00 IST and over the last 2 hours
                   (hourly bars that have closed)
  daily_ctx   the previous sessions' global and volatility context (strictly lagged, from features.py)
  events      scheduled event flags for the day (RBI, FOMC, Budget, ...)
"""

from __future__ import annotations

import datetime as dt
from typing import Dict, Optional

import numpy as np
import pandas as pd

from ..calendar import IST

STOCK_WEIGHTS = {"hdfcbank": 0.13, "reliance": 0.09, "icicibank": 0.09, "infy": 0.05, "bhartiartl": 0.05,
                 "lt": 0.04, "tcs": 0.03, "itc": 0.03, "sbin": 0.03, "axisbank": 0.03}
DAILY_CTX = ["spx_r1", "ndx_r1", "usvix", "usvix_chg1", "dxy_r1", "us10y_chg1", "brent_r1", "usdinr_r1", "gold_r1",
             "n225_gap", "hsi_gap", "vix", "vix_z60", "vix_chg1", "vrp", "rv_m", "rv_ratio", "r5", "r20",
             "dist_sma50", "dist_sma200", "bn_rel_r5"]


def _at(s: pd.Series, dec: pd.DatetimeIndex) -> np.ndarray:
    """Value of a start-stamped 5-minute series at its last bar closed by each T (same session only)."""
    return s.reindex(dec - pd.Timedelta(minutes=5)).to_numpy(float)


def _day_first(s: pd.Series) -> pd.Series:
    return s.groupby(s.index.date).transform("first")


def banknifty_features(dec, nifty5: pd.DataFrame, bn5: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=dec)
    bc, nc = bn5["close"], nifty5["close"]
    bo, no = _day_first(bn5["open"]), _day_first(nifty5["open"])
    b_now, n_now = _at(bc, dec), _at(nc, dec)
    b_open, n_open = _at(bo, dec), _at(no, dec)
    X["bn_r_open"] = np.log(b_now / b_open)
    X["bn_rel_open"] = X["bn_r_open"] - np.log(n_now / n_open)
    b30, n30 = bc.shift(6), nc.shift(6)
    same = pd.Series(bc.index.date, index=bc.index) == pd.Series(bc.index.date, index=bc.index).shift(6)
    X["bn_r_30m"] = np.log(b_now / _at(b30.where(same), dec))
    X["bn_rel_30m"] = X["bn_r_30m"] - np.log(n_now / _at(n30.reindex(bc.index).where(same), dec))
    return X


def futures_features(dec, nifty5: pd.DataFrame, fut5: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=dec)
    f = fut5.copy()
    fc, fv = f["close"], f["volume"].astype(float)
    day = pd.Series(f.index.date, index=f.index)
    tp = (f["high"] + f["low"] + f["close"]) / 3
    vwap = (tp * fv).groupby(day).cumsum() / fv.groupby(day).cumsum().replace(0, np.nan)
    spot = nifty5["close"].reindex(f.index)
    basis = (fc / spot - 1) * 1e4
    X["fut_basis_bps"] = _at(basis, dec)
    X["fut_basis_chg"] = X["fut_basis_bps"] - _at(basis.groupby(day).transform("first"), dec)
    X["fut_vwap_dist"] = (_at(fc, dec) / _at(vwap, dec) - 1) * 100
    # volume surge vs the same 5-minute slot on the previous 20 sessions
    slot = pd.Series(f.index.strftime("%H%M"), index=f.index)
    piv = pd.DataFrame({"v": fv.values, "d": day.values, "s": slot.values}).pivot_table(index="d", columns="s", values="v")
    base = piv.shift(1).rolling(20, min_periods=5).mean()
    ratio = (piv / base).stack(future_stack=True)
    ratio.index = pd.DatetimeIndex([pd.Timestamp(f"{d} {s[:2]}:{s[2:]}", tz=IST) for d, s in ratio.index])
    X["fut_vol_surge"] = np.log(_at(ratio.reindex(f.index), dec))
    cum = fv.groupby(day).cumsum()
    v30 = cum - cum.groupby(day).shift(6).fillna(0)
    X["fut_vol_30m_share"] = _at(v30 / cum.replace(0, np.nan), dec)
    if "oi" in f and f["oi"].fillna(0).sum() > 0:
        oi = f["oi"].astype(float).replace(0, np.nan).groupby(day).ffill()
        X["fut_oi_chg_open"] = _at(oi / oi.groupby(day).transform("first") - 1, dec) * 100
        oi30 = oi.groupby(day).shift(6)
        d_oi = _at(oi / oi30 - 1, dec)
        d_px = _at(fc / fc.groupby(day).shift(6) - 1, dec)
        X["fut_oi_chg_30m"] = d_oi * 100
        X["fut_oi_regime"] = np.where((d_px > 0) & (d_oi > 0), 1.0, np.where((d_px < 0) & (d_oi > 0), -1.0,
                                      np.where((d_px > 0) & (d_oi < 0), 0.5, np.where((d_px < 0) & (d_oi < 0), -0.5, 0.0))))
    return X


def breadth_features(dec, nifty5: pd.DataFrame, stocks: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    X = pd.DataFrame(index=dec)
    if not stocks:
        return X
    above, up30, rop, wsum = [], [], {}, 0.0
    for name, df in stocks.items():
        c = df["close"]
        o = _day_first(df["open"])
        now = _at(c, dec)
        r_open = np.log(now / _at(o, dec))
        same = pd.Series(c.index.date, index=c.index) == pd.Series(c.index.date, index=c.index).shift(6)
        r30 = np.log(now / _at(c.shift(6).where(same), dec))
        above.append(r_open > 0)
        up30.append(r30 > 0)
        rop[name] = r_open
    A, U = np.vstack(above).astype(float), np.vstack(up30).astype(float)
    X["br_above_open"] = np.nanmean(np.where(np.isnan(np.vstack(list(rop.values()))), np.nan, A), axis=0)
    X["br_up_30m"] = U.mean(axis=0)
    R = pd.DataFrame(rop, index=dec)
    w = pd.Series({k: STOCK_WEIGHTS.get(k, 0.02) for k in R.columns})
    X["br_weighted_r_open"] = (R * w).sum(axis=1) / w.sum()
    n_open = np.log(_at(nifty5["close"], dec) / _at(_day_first(nifty5["open"]), dec))
    X["br_lead_vs_index"] = X["br_weighted_r_open"] - n_open
    X["br_dispersion"] = R.std(axis=1)
    top = [c for c in ("hdfcbank", "reliance", "icicibank") if c in R]
    if top:
        X["br_top3_r_open"] = R[top].mean(axis=1)
    return X


def global_intraday_features(dec, hourly: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Hourly bars are start-stamped (Yahoo); a bar is usable once start + 1h <= T."""
    X = pd.DataFrame(index=dec)
    for name, df in hourly.items():
        if df is None or df.empty:
            continue
        c = df["close"].copy()
        c.index = pd.DatetimeIndex(c.index).tz_convert(IST) + pd.Timedelta(hours=1)     # stamp = availability
        c = c[~c.index.duplicated(keep="last")].sort_index()
        now = c.reindex(dec, method="ffill").to_numpy(float)
        two = c.reindex(dec - pd.Timedelta(hours=2), method="ffill").to_numpy(float)
        nine = c.reindex(pd.DatetimeIndex([dt.datetime.combine(t.date(), dt.time(9, 0), tzinfo=IST) for t in dec]),
                         method="ffill").to_numpy(float)
        X[f"gl_{name}_r_since0900"] = np.log(now / nine)
        X[f"gl_{name}_r_2h"] = np.log(now / two)
    return X.replace([np.inf, -np.inf], np.nan)


def daily_context(dec, daily_feats: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in DAILY_CTX if c in daily_feats]
    D = daily_feats[cols].copy()
    D.index = pd.DatetimeIndex(D.index).normalize()
    days = pd.DatetimeIndex([pd.Timestamp(t.date()) for t in dec])
    out = D.reindex(days)
    out.index = dec
    return out.add_prefix("dc_")


def events_features(dec) -> pd.DataFrame:
    from ..data.events import event_features
    days = pd.DatetimeIndex(sorted({pd.Timestamp(t.date()) for t in dec}))
    E = event_features(days)
    E.index = pd.DatetimeIndex(E.index).normalize()
    out = E.reindex(pd.DatetimeIndex([pd.Timestamp(t.date()) for t in dec]))
    out.index = dec
    return out.add_prefix("evt_")
