"""Option-chain microstructure metrics from one chain snapshot (live broker chain, the feed
recorder, or a snapshot rebuilt from expired-contract candles).

Input `chain`: one row per (strike, type) with columns strike, type (CE/PE), ltp, and where
available bid, ask, oi, volume, iv (decimal or %), delta, gamma. Missing IV/greeks are
recomputed from the last price with Black-Scholes.

Assumptions are stated where the literature disagrees:
  * GEX sign convention (SqueezeMetrics-style): dealers are assumed long the calls and short the
    puts that customers trade, so GEX = sum(call gamma x OI) - sum(put gamma x OI). Positive GEX
    = dealers dampen moves; negative = they amplify. This is a hypothesis to be *tested*, not a fact:
    NSE does not publish who is on which side.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np
import pandas as pd

from ..options import bs_greeks, implied_vol


def _clean(chain: pd.DataFrame, spot: float, T: float) -> pd.DataFrame:
    ch = chain.copy()
    for c in ("ltp", "bid", "ask", "oi", "volume", "iv", "delta", "gamma"):
        if c not in ch:
            ch[c] = np.nan
        ch[c] = pd.to_numeric(ch[c], errors="coerce")
    ch["strike"] = ch["strike"].astype(float)
    mid = np.where((ch["bid"] > 0) & (ch["ask"] > 0), (ch["bid"] + ch["ask"]) / 2, ch["ltp"])
    ch["mid"] = mid
    iv = ch["iv"].where(ch["iv"] > 0)
    iv = np.where(iv > 3, iv / 100.0, iv)       # brokers report IV in % or decimals
    ch["iv"] = iv
    need = ch["iv"].isna() & (ch["mid"] > 0)
    if need.any():
        ch.loc[need, "iv"] = [implied_vol(float(m), spot, float(k), T, t)
                              for m, k, t in zip(ch.loc[need, "mid"], ch.loc[need, "strike"], ch.loc[need, "type"])]
    g_need = ch["gamma"].isna() | ch["delta"].isna()
    if g_need.any():
        gk = [bs_greeks(spot, float(k), T, float(v), t) if v == v and v > 0 else {"delta": np.nan, "gamma": np.nan}
              for k, v, t in zip(ch.loc[g_need, "strike"], ch.loc[g_need, "iv"], ch.loc[g_need, "type"])]
        ch.loc[g_need, "delta"] = [g["delta"] for g in gk]
        ch.loc[g_need, "gamma"] = [g["gamma"] for g in gk]
    return ch


def max_pain(ch: pd.DataFrame) -> float:
    """Strike at which option writers' total payout at expiry is smallest."""
    ks = np.sort(ch["strike"].unique())
    calls = ch[ch["type"] == "CE"].set_index("strike")["oi"].fillna(0)
    puts = ch[ch["type"] == "PE"].set_index("strike")["oi"].fillna(0)
    pay = []
    for s in ks:
        pc = float(((s - calls.index.values).clip(min=0) * calls.values).sum())
        pp = float(((puts.index.values - s).clip(min=0) * puts.values).sum())
        pay.append(pc + pp)
    return float(ks[int(np.argmin(pay))]) if len(ks) else float("nan")


def gex_profile(ch: pd.DataFrame, spot: float, T: float, lot_size: int, grid_pct: float = 0.05,
                steps: int = 41) -> pd.DataFrame:
    """Dealer gamma exposure (Rs crore per 1% move) re-evaluated across a grid of spot levels."""
    grid = np.linspace(spot * (1 - grid_pct), spot * (1 + grid_pct), steps)
    rows = []
    calls = ch[(ch["type"] == "CE") & (ch["iv"] > 0)]
    puts = ch[(ch["type"] == "PE") & (ch["iv"] > 0)]
    for s in grid:
        gc = sum(bs_greeks(s, k, T, v, "CE")["gamma"] * oi for k, v, oi in zip(calls["strike"], calls["iv"], calls["oi"].fillna(0)))
        gp = sum(bs_greeks(s, k, T, v, "PE")["gamma"] * oi for k, v, oi in zip(puts["strike"], puts["iv"], puts["oi"].fillna(0)))
        rows.append({"spot": s, "gex_cr": (gc - gp) * lot_size * s * s * 0.01 / 1e7})
    return pd.DataFrame(rows)


def zero_gamma(profile: pd.DataFrame) -> float:
    g = profile["gex_cr"].to_numpy()
    s = profile["spot"].to_numpy()
    sgn = np.sign(g)
    for i in range(1, len(g)):
        if sgn[i] != sgn[i - 1] and sgn[i] != 0:
            return float(s[i - 1] + (s[i] - s[i - 1]) * (-g[i - 1]) / (g[i] - g[i - 1]))
    return float("nan")


def _iv_at_delta(ch: pd.DataFrame, typ: str, target: float) -> float:
    side = ch[(ch["type"] == typ) & ch["delta"].notna() & (ch["iv"] > 0)]
    if side.empty:
        return float("nan")
    i = (side["delta"] - target).abs().idxmin()
    return float(side.loc[i, "iv"])


def chain_metrics(chain: pd.DataFrame, spot: float, T: float, lot_size: int = 65,
                  prev: Optional[pd.DataFrame] = None, with_gex: bool = True) -> Dict[str, float]:
    """Summary statistics of one snapshot. `prev` = an earlier snapshot of the same expiry, for
    OI-change / build-up classification. T = year fraction to expiry."""
    ch = _clean(chain, spot, T)
    calls, puts = ch[ch["type"] == "CE"], ch[ch["type"] == "PE"]
    atm = float(ch.loc[(ch["strike"] - spot).abs().idxmin(), "strike"]) if len(ch) else float("nan")
    c_atm = calls[calls["strike"] == atm]
    p_atm = puts[puts["strike"] == atm]
    straddle = float(c_atm["mid"].iloc[0] + p_atm["mid"].iloc[0]) if len(c_atm) and len(p_atm) else float("nan")
    m: Dict[str, float] = {
        "spot": spot, "atm": atm,
        "pcr_oi": float(puts["oi"].sum() / calls["oi"].sum()) if calls["oi"].sum() > 0 else float("nan"),
        "pcr_vol": float(puts["volume"].sum() / calls["volume"].sum()) if calls["volume"].sum() > 0 else float("nan"),
        "call_wall": float(calls.loc[calls["oi"].idxmax(), "strike"]) if calls["oi"].notna().any() else float("nan"),
        "put_wall": float(puts.loc[puts["oi"].idxmax(), "strike"]) if puts["oi"].notna().any() else float("nan"),
        "max_pain": max_pain(ch) if ch["oi"].notna().any() else float("nan"),
        "atm_iv": float(np.nanmean([c_atm["iv"].mean() if len(c_atm) else np.nan,
                                    p_atm["iv"].mean() if len(p_atm) else np.nan])),
        "straddle": straddle,
        "implied_move_pct": straddle / spot * 100 if straddle == straddle else float("nan"),
    }
    c25, p25 = _iv_at_delta(ch, "CE", 0.25), _iv_at_delta(ch, "PE", -0.25)
    m["rr25"] = p25 - c25                       # put skew: >0 = puts richer than calls
    m["bf25"] = 0.5 * (p25 + c25) - m["atm_iv"]
    m["call_wall_dist_pct"] = (m["call_wall"] / spot - 1) * 100
    m["put_wall_dist_pct"] = (m["put_wall"] / spot - 1) * 100
    m["max_pain_dist_pct"] = (m["max_pain"] / spot - 1) * 100
    if "bid" in ch and "ask" in ch:
        near = ch[(ch["strike"] - spot).abs() <= 300]
        spr = (near["ask"] - near["bid"]) / near["mid"].replace(0, np.nan)
        m["rel_spread_near_atm"] = float(spr[spr > 0].median()) if (spr > 0).any() else float("nan")
    if with_gex and ch["oi"].notna().any():
        prof = gex_profile(ch, spot, T, lot_size)
        mid = prof.iloc[(prof["spot"] - spot).abs().argmin()]
        m["gex_cr"] = float(mid["gex_cr"])
        m["zero_gamma"] = zero_gamma(prof)
        m["zero_gamma_dist_pct"] = (m["zero_gamma"] / spot - 1) * 100 if m["zero_gamma"] == m["zero_gamma"] else float("nan")
    if prev is not None and len(prev):
        p = _clean(prev, spot, T)[["strike", "type", "oi", "mid"]].rename(columns={"oi": "oi0", "mid": "mid0"})
        j = ch.merge(p, on=["strike", "type"], how="inner")
        j["doi"], j["dpx"] = j["oi"] - j["oi0"], j["mid"] - j["mid0"]
        for side in ("CE", "PE"):
            s = j[j["type"] == side]
            m[f"doi_{side.lower()}"] = float(s["doi"].sum())
        b = j.apply(lambda r: buildup(r["dpx"], r["doi"]), axis=1) if len(j) else pd.Series(dtype=object)
        for lab in ("long_buildup", "short_buildup", "short_covering", "long_unwinding"):
            for side in ("CE", "PE"):
                m[f"{lab}_{side.lower()}"] = float(((b == lab) & (j["type"] == side)).sum())
        tot = abs(m["doi_ce"]) + abs(m["doi_pe"])
        m["doi_put_minus_call_share"] = (m["doi_pe"] - m["doi_ce"]) / tot if tot > 0 else 0.0
    return m


def buildup(dpx: float, doi: float) -> str:
    """Price/OI change classification used by Indian F&O desks."""
    if not (dpx == dpx and doi == doi) or doi == 0:
        return "none"
    if dpx >= 0 and doi > 0:
        return "long_buildup"
    if dpx < 0 and doi > 0:
        return "short_buildup"
    if dpx > 0 and doi < 0:
        return "short_covering"
    return "long_unwinding"


def depth_imbalance(bid_qty: float, ask_qty: float) -> float:
    """(bid - ask) / (bid + ask) over the visible book: +1 all bids, -1 all asks."""
    tot = (bid_qty or 0) + (ask_qty or 0)
    return ((bid_qty or 0) - (ask_qty or 0)) / tot if tot > 0 else float("nan")


def term_structure(atm_iv_near: float, atm_iv_next: float) -> float:
    """Near minus next ATM IV. >0 (inverted) usually means event risk or stress."""
    return atm_iv_near - atm_iv_next if atm_iv_near == atm_iv_near and atm_iv_next == atm_iv_next else float("nan")
