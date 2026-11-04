"""Meta-labelling (Lopez de Prado 2018, ch. 3): a second model that decides whether to *take* each
trade a primary book proposes, from the context at entry (regime, VIX, news, the book's own recent
hit rate). It never changes the trade's direction, only filters and sizes it.

Walk-forward in trade order: the filter for trade i is trained only on trades that had *closed*
before trade i was entered. Pre-registered success test: higher net P&L per trade while keeping
at least 70% of trades (otherwise it is just trading less)."""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DEFAULT_FEATURES = ["har_vol_pct", "vrp_har", "vix_pct_1y", "er20", "gap_abs", "nw_dir", "nw_shock",
                    "book_hit_20", "book_net_20", "p_conf"]


def add_trade_context(T: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """Join daily context (already causal for each session) and the book's own recent record."""
    T = T.copy()
    T["date_ts"] = pd.to_datetime(T["date"])
    ctx = daily.reindex(T["date_ts"]).reset_index(drop=True)
    for c in ctx.columns:
        T[c] = ctx[c].values
    if "gap" in T:
        T["gap_abs"] = T["gap"].abs()
    p = T.get("m_p_up", pd.Series(np.nan, index=T.index))
    T["p_conf"] = (p - 0.5).abs()
    T = T.sort_values(["entry_ts"]).reset_index(drop=True)
    T["exit_ts_v"] = pd.to_datetime(T["exit_ts"], utc=True, errors="coerce")
    T["entry_ts_v"] = pd.to_datetime(T["entry_ts"], utc=True, errors="coerce")
    hit, netm = [], []
    for i, r in T.iterrows():
        past = T[(T["book"] == r["book"]) & (T["exit_ts_v"] < r["entry_ts_v"])].tail(20)
        hit.append(float((past["net_pnl"] > 0).mean()) if len(past) >= 5 else np.nan)
        netm.append(float(past["net_pnl"].mean()) if len(past) >= 5 else np.nan)
    T["book_hit_20"], T["book_net_20"] = hit, netm
    return T


def walk_forward_meta(T: pd.DataFrame, features: Optional[List[str]] = None, min_train: int = 60,
                      refit_every: int = 20, threshold: float = 0.5) -> pd.DataFrame:
    feats = [f for f in (features or DEFAULT_FEATURES) if f in T.columns]
    T = T.copy()
    T["meta_p"] = np.nan
    if not feats:
        return T
    model, last_fit = None, -10 ** 9
    for i in range(len(T)):
        closed = T.iloc[:i]
        closed = closed[closed["exit_ts_v"] < T["entry_ts_v"].iloc[i]]
        if len(closed) >= min_train and (model is None or i - last_fit >= refit_every):
            y = (closed["net_pnl"] > 0).astype(int)
            if y.nunique() == 2:
                model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                      LogisticRegression(C=0.3, max_iter=1000)).fit(closed[feats], y)
                last_fit = i
        if model is not None:
            T.loc[T.index[i], "meta_p"] = float(model.predict_proba(T[feats].iloc[[i]])[0, 1])
    T["meta_take"] = (T["meta_p"].isna()) | (T["meta_p"] >= threshold)
    return T


def evaluate_meta(T: pd.DataFrame) -> Dict[str, float]:
    scored = T[T["meta_p"].notna()]
    if scored.empty:
        return {"scored_trades": 0}
    kept = scored[scored["meta_take"]]
    return {"scored_trades": int(len(scored)), "kept_share": round(float(len(kept) / len(scored)), 3),
            "net_per_trade_all": round(float(scored["net_pnl"].mean()), 1),
            "net_per_trade_kept": round(float(kept["net_pnl"].mean()), 1) if len(kept) else None,
            "win_rate_all": round(float((scored["net_pnl"] > 0).mean()), 3),
            "win_rate_kept": round(float((kept["net_pnl"] > 0).mean()), 3) if len(kept) else None,
            "net_total_all": round(float(scored["net_pnl"].sum()), 0),
            "net_total_kept": round(float(kept["net_pnl"].sum()), 0)}


def kelly_lots(p_win: float, avg_win: float, avg_loss: float, max_loss_per_lot: float, equity: float,
               fraction: float = 0.25, cap_pct: float = 10.0, max_lots: int = 1) -> int:
    """Fractional Kelly on the trade's own win/loss profile, capped by a loss budget per trade."""
    if not (avg_win > 0 and avg_loss < 0 and 0 < p_win < 1 and max_loss_per_lot > 0):
        return 0
    b = avg_win / -avg_loss
    f = max(0.0, (p_win * b - (1 - p_win)) / b) * fraction
    budget = equity * min(f, cap_pct / 100)
    return int(max(0, min(max_lots, math.floor(budget / max_loss_per_lot))))
