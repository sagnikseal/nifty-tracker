"""Intraday option-chain features rebuilt from REAL 1-minute candles of the expired weekly NIFTY
options (store/options_hist_1m: price, volume and open interest per strike per minute).

Upstox does not sell historical option-chain snapshots, but the expired-contract candles carry the
same information for the strikes we downloaded (about ATM +/- 4-7 strikes each day, both CE and PE),
so the positioning features can be tested on ~2 years of history instead of waiting months for
live recordings. At decision time T only 1-minute bars that have closed by T are used.

Features (prefix oc_):
  oc_pcr_oi, oc_pcr_oi_chg       put/call open-interest ratio near the money, and its change since 09:20
  oc_ce_oi_chg, oc_pe_oi_chg     % change in call / put open interest since 09:20 (writers adding = "walls")
  oc_net_oi_flow                 put OI added minus call OI added (+ = put writing = support below)
  oc_pcr_vol, oc_pcr_vol_30m     put/call traded volume ratio today / last 30 minutes
  oc_atm_iv, oc_iv_chg, oc_iv_chg_30m   at-the-money implied volatility and its changes
  oc_skew                        IV of the put 100 points below minus the call 100 points above
  oc_implied_move, oc_straddle_chg_30m  ATM straddle as % of spot (expected move to expiry) and its change
  oc_call_wall_dist, oc_put_wall_dist   distance (%) from spot to the strikes with the largest call / put OI
  oc_cp_mom_30m                  ATM call return minus ATM put return over 30 minutes (option-flow momentum)
  oc_dte                         days to the expiry used
"""

from __future__ import annotations

import datetime as dt
import logging
import math
from typing import Dict, List

import numpy as np
import pandas as pd

from ..calendar import IST, year_fraction
from ..config import DATA_DIR
from ..options import implied_vol

log = logging.getLogger(__name__)
OPT_DIR = DATA_DIR / "options_hist_1m"
FEATURES_FILE = DATA_DIR / "features" / "chain_intraday.csv.gz"


def _iv(price, S, K, T, typ):
    try:
        v = implied_vol(float(price), float(S), float(K), T, typ)
        return v if v == v else np.nan
    except Exception:  # noqa: BLE001
        return np.nan


WINDOW_PTS = 100   # strikes used: the 09:15 at-the-money strike +/- 100 points (known at the open)


def day_features(opt_day: pd.DataFrame, spot5: pd.Series, decisions: List[pd.Timestamp], expiry: dt.date,
                 window_pts: int = WINDOW_PTS) -> List[Dict]:
    """opt_day: one session's 1-minute rows (ts,close,volume,oi,strike,type) of one expiry.

    Look-ahead guard: a weekly download contains the strikes needed on EVERY day of that week (chosen from each
    day's open), so the set of strikes present on Monday partly reflects where NIFTY went later in the week.
    Only strikes within +/- window_pts of the at-the-money strike at the session's first 5-minute close are
    used, a set that is fixed by information available at 09:20 (and identical live)."""
    if opt_day.empty:
        return []
    day = opt_day["ts"].iloc[0].date() if hasattr(opt_day["ts"].iloc[0], "date") else None
    sd = spot5[spot5.index.date == day] if day is not None else spot5
    if sd.empty:
        return []
    k0 = round(float(sd.iloc[0]) / 50) * 50
    o = opt_day[(opt_day["strike"] >= k0 - window_pts) & (opt_day["strike"] <= k0 + window_pts)].sort_values("ts")
    if o.empty:
        return []
    keys = sorted(set(zip(o["strike"].astype(float), o["type"])))
    piv = {f: o.pivot_table(index="ts", columns=["strike", "type"], values=f, aggfunc="last").reindex(columns=keys)
           for f in ("close", "volume", "oi")}
    close = piv["close"].ffill()
    oi = piv["oi"].ffill()
    vol = piv["volume"].fillna(0.0)
    cumvol = vol.cumsum()
    ts = close.index
    rows, first = [], None
    exp_close = dt.datetime.combine(expiry, dt.time(15, 30), tzinfo=IST)
    for T in decisions:
        k = ts.searchsorted(T - pd.Timedelta(minutes=1), side="right") - 1      # last 1-min bar closed by T
        s_prev = T - pd.Timedelta(minutes=5)
        if k < 0 or s_prev not in spot5.index:
            continue
        S = float(spot5.loc[s_prev])
        K = min(max(round(S / 50) * 50, k0 - window_pts), k0 + window_pts)   # nearest strike inside the fixed window
        c, oi_now, cv = close.iloc[k], oi.iloc[k], cumvol.iloc[k]
        f: Dict[str, float] = {"ts": T, "oc_dte": (expiry - T.date()).days}
        ce_oi = oi_now.xs("CE", level="type") if ("CE" in oi_now.index.get_level_values(1)) else pd.Series(dtype=float)
        pe_oi = oi_now.xs("PE", level="type") if ("PE" in oi_now.index.get_level_values(1)) else pd.Series(dtype=float)
        f["oc_pcr_oi"] = float(pe_oi.sum() / ce_oi.sum()) if ce_oi.sum() > 0 else np.nan
        ce_v, pe_v = cv.xs("CE", level="type").sum(), cv.xs("PE", level="type").sum()
        f["oc_pcr_vol"] = float(pe_v / ce_v) if ce_v > 0 else np.nan
        k30 = ts.searchsorted(T - pd.Timedelta(minutes=31), side="right") - 1
        if k30 >= 0:
            dv = cv - cumvol.iloc[k30]
            ce30, pe30 = dv.xs("CE", level="type").sum(), dv.xs("PE", level="type").sum()
            f["oc_pcr_vol_30m"] = float(pe30 / ce30) if ce30 > 0 else np.nan
        if len(ce_oi.dropna()) and ce_oi.max() > 0:
            f["oc_call_wall_dist"] = (float(ce_oi.idxmax()) / S - 1) * 100
        if len(pe_oi.dropna()) and pe_oi.max() > 0:
            f["oc_put_wall_dist"] = (float(pe_oi.idxmax()) / S - 1) * 100
        Ty = year_fraction(T.to_pydatetime(), exp_close)
        ce_atm, pe_atm = c.get((float(K), "CE"), np.nan), c.get((float(K), "PE"), np.nan)
        ivs = [_iv(ce_atm, S, K, Ty, "CE"), _iv(pe_atm, S, K, Ty, "PE")]
        f["oc_atm_iv"] = float(np.nanmean(ivs)) * 100 if not all(np.isnan(ivs)) else np.nan
        f["oc_implied_move"] = (ce_atm + pe_atm) / S * 100 if ce_atm == ce_atm and pe_atm == pe_atm else np.nan
        pe_lo, ce_hi = c.get((float(K - 100), "PE"), np.nan), c.get((float(K + 100), "CE"), np.nan)
        if pe_lo == pe_lo and ce_hi == ce_hi:
            f["oc_skew"] = (_iv(pe_lo, S, K - 100, Ty, "PE") - _iv(ce_hi, S, K + 100, Ty, "CE")) * 100
        if k30 >= 0:
            c30 = close.iloc[k30]
            a, b = c30.get((float(K), "CE"), np.nan), c30.get((float(K), "PE"), np.nan)
            if a and b and a == a and b == b and ce_atm == ce_atm and pe_atm == pe_atm and a > 0 and b > 0:
                f["oc_cp_mom_30m"] = math.log(ce_atm / a) - math.log(pe_atm / b)
                f["oc_straddle_chg_30m"] = math.log((ce_atm + pe_atm) / (a + b))
        if first is None:
            first = {"ce_oi": ce_oi.copy(), "pe_oi": pe_oi.copy(), "pcr": f["oc_pcr_oi"], "iv": f.get("oc_atm_iv")}
        common_c = ce_oi.index.intersection(first["ce_oi"].index)
        common_p = pe_oi.index.intersection(first["pe_oi"].index)
        c0, p0 = first["ce_oi"][common_c].sum(), first["pe_oi"][common_p].sum()
        f["oc_ce_oi_chg"] = float(ce_oi[common_c].sum() / c0 - 1) if c0 > 0 else np.nan
        f["oc_pe_oi_chg"] = float(pe_oi[common_p].sum() / p0 - 1) if p0 > 0 else np.nan
        if f["oc_ce_oi_chg"] == f["oc_ce_oi_chg"] and f["oc_pe_oi_chg"] == f["oc_pe_oi_chg"]:
            f["oc_net_oi_flow"] = f["oc_pe_oi_chg"] - f["oc_ce_oi_chg"]
        f["oc_pcr_oi_chg"] = f["oc_pcr_oi"] - first["pcr"] if first["pcr"] == first["pcr"] else np.nan
        if first.get("iv") is not None and first["iv"] == first["iv"] and f.get("oc_atm_iv") == f.get("oc_atm_iv"):
            f["oc_iv_chg"] = f["oc_atm_iv"] - first["iv"]
        rows.append(f)
    if rows:
        ivs = pd.Series([r.get("oc_atm_iv", np.nan) for r in rows])
        for i, r in enumerate(rows):
            if i >= 6 and ivs.iloc[i] == ivs.iloc[i] and ivs.iloc[i - 6] == ivs.iloc[i - 6]:
                r["oc_iv_chg_30m"] = float(ivs.iloc[i] - ivs.iloc[i - 6])
    return rows


def build(spot5: pd.Series, decision_index: pd.DatetimeIndex, save: bool = True) -> pd.DataFrame:
    """Chain features for every decision time that has option data (nearest expiry >= 1 day away)."""
    exps = sorted(dt.date.fromisoformat(p.stem) for p in OPT_DIR.glob("*.csv"))
    if not exps:
        return pd.DataFrame()
    by_day: Dict[dt.date, List[pd.Timestamp]] = {}
    for T in decision_index:
        by_day.setdefault(T.date(), []).append(T)
    plan: Dict[dt.date, List[dt.date]] = {}
    for d in by_day:
        nxt = [e for e in exps if (e - d).days >= 1]
        if nxt and (nxt[0] - d).days <= 10:
            plan.setdefault(nxt[0], []).append(d)
    rows = []
    for e, days in sorted(plan.items()):
        df = pd.read_csv(OPT_DIR / f"{e.isoformat()}.csv", usecols=["ts", "close", "volume", "oi", "strike", "type"])
        df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert(IST)
        for d in days:
            od = df[df["ts"].dt.date == d]
            rows += day_features(od, spot5, sorted(by_day[d]), e)
    X = pd.DataFrame(rows).set_index("ts").sort_index() if rows else pd.DataFrame()
    if save and len(X):
        FEATURES_FILE.parent.mkdir(parents=True, exist_ok=True)
        X.to_csv(FEATURES_FILE)
    log.info("chain features: %d decisions on %d days", len(X), len(set(X.index.date)) if len(X) else 0)
    return X


def load() -> pd.DataFrame:
    if not FEATURES_FILE.exists():
        return pd.DataFrame()
    X = pd.read_csv(FEATURES_FILE, index_col=0)
    X.index = pd.to_datetime(X.index, utc=True).tz_convert(IST)
    return X
