"""Statistics that keep a backtest honest when many ideas are tried.

* Deflated Sharpe ratio (Bailey & Lopez de Prado 2014): the probability that the true Sharpe is
  above zero after accounting for how many strategies were tried, and for skew / fat tails.
* Probability of backtest overfitting (Bailey, Borwein, Lopez de Prado & Zhu 2015, CSCV): how
  often the variant that looks best in-sample ranks below the median out-of-sample.
* Diebold-Mariano test on per-day log-loss, for "does feature group X improve the model on the
  same days".
"""

from __future__ import annotations

import itertools
import math
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

EULER = 0.5772156649


def _ncdf(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2))


def _nppf(p: float) -> float:
    """Inverse normal CDF (Acklam's rational approximation, |error| < 1.2e-9)."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00]
    p = min(max(p, 1e-12), 1 - 1e-12)
    if p < 0.02425:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > 1 - 0.02425:
        return -_nppf(1 - p)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def sharpe(x: Sequence[float]) -> float:
    x = np.asarray(x, float)
    return float(x.mean() / x.std(ddof=1)) if len(x) > 1 and x.std(ddof=1) > 0 else 0.0


def deflated_sharpe(returns: Sequence[float], n_trials: int, sr_trials_var: Optional[float] = None) -> Dict[str, float]:
    """DSR for one strategy's per-trade (or per-period) returns. `sr_trials_var`: variance of the
    Sharpe ratios of all trials; if unknown, the variance of the SR estimator (1/T) is used."""
    r = np.asarray(returns, float)
    T = len(r)
    if T < 3:
        return {"sr": float("nan"), "sr0": float("nan"), "dsr": float("nan"), "n": T}
    sr = sharpe(r)
    sk = float(pd.Series(r).skew())
    ku = float(pd.Series(r).kurt()) + 3.0
    v = sr_trials_var if sr_trials_var and sr_trials_var > 0 else 1.0 / T
    N = max(int(n_trials), 1)
    if N > 1:
        sr0 = math.sqrt(v) * ((1 - EULER) * _nppf(1 - 1 / N) + EULER * _nppf(1 - 1 / (N * math.e)))
    else:
        sr0 = 0.0
    den = math.sqrt(max(1e-12, 1 - sk * sr + (ku - 1) / 4 * sr * sr))
    dsr = _ncdf((sr - sr0) * math.sqrt(T - 1) / den)
    return {"sr": sr, "sr0": sr0, "dsr": dsr, "n": T, "skew": sk, "kurt": ku}


def pbo_cscv(M: pd.DataFrame, S: int = 10) -> Dict[str, float]:
    """M: rows = time periods (e.g. months), columns = strategy variants, values = returns.
    Splits rows into S blocks; for every half/half combination, picks the best in-sample variant
    and records its out-of-sample rank. PBO = share of splits where it ranks below the median."""
    M = M.dropna(how="all").fillna(0.0)
    if M.shape[1] < 2 or len(M) < S:
        return {"pbo": float("nan"), "splits": 0}
    blocks = np.array_split(np.arange(len(M)), S)
    logits = []
    for comb in itertools.combinations(range(S), S // 2):
        is_idx = np.concatenate([blocks[i] for i in comb])
        oos_idx = np.concatenate([blocks[i] for i in range(S) if i not in comb])
        is_perf = M.iloc[is_idx].mean() / M.iloc[is_idx].std(ddof=1).replace(0, np.nan)
        oos_perf = M.iloc[oos_idx].mean() / M.iloc[oos_idx].std(ddof=1).replace(0, np.nan)
        best = is_perf.fillna(-np.inf).idxmax()
        rank = oos_perf.rank(pct=True).get(best, np.nan)
        if rank == rank:
            w = min(max(rank * len(M.columns) / (len(M.columns) + 1), 1e-6), 1 - 1e-6)
            logits.append(math.log(w / (1 - w)))
    lg = np.array(logits)
    return {"pbo": float((lg <= 0).mean()) if len(lg) else float("nan"), "splits": int(len(lg))}


def dm_logloss(p_a: np.ndarray, p_b: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Does forecast B beat forecast A? Positive t => B has lower log-loss. One-sided p-value."""
    pa, pb = np.clip(p_a, 1e-6, 1 - 1e-6), np.clip(p_b, 1e-6, 1 - 1e-6)
    la = -(y * np.log(pa) + (1 - y) * np.log(1 - pa))
    lb = -(y * np.log(pb) + (1 - y) * np.log(1 - pb))
    d = la - lb
    n = len(d)
    if n < 10 or d.std(ddof=1) == 0:
        return {"n": n, "t": float("nan"), "p": float("nan"), "delta_logloss": float(d.mean()) if n else float("nan")}
    t = d.mean() / (d.std(ddof=1) / math.sqrt(n))
    return {"n": n, "t": float(t), "p": float(1 - _ncdf(t)), "delta_logloss": float(d.mean()),
            "acc_a": float(np.mean((pa >= 0.5) == y)), "acc_b": float(np.mean((pb >= 0.5) == y))}


def t_stat(x: Sequence[float]) -> float:
    x = np.asarray(x, float)
    return float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x)))) if len(x) > 2 and x.std(ddof=1) > 0 else 0.0


def bootstrap_mean_ci(x: Sequence[float], n_boot: int = 2000, seed: int = 7) -> tuple:
    x = np.asarray(x, float)
    if len(x) < 5:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    m = rng.choice(x, (n_boot, len(x)), replace=True).mean(axis=1)
    return (float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)))


def accuracy_coverage(p: np.ndarray, y: np.ndarray, steps=(1.0, 0.8, 0.6, 0.4, 0.3, 0.2, 0.1)) -> list:
    """Hit rate when trading only the most confident fraction of days."""
    conf = np.abs(np.asarray(p) - 0.5)
    order = np.argsort(-conf)
    out = []
    for c in steps:
        k = max(1, int(len(p) * c))
        sel = order[:k]
        out.append({"coverage": c, "n": k, "hit_rate": float(np.mean((np.asarray(p)[sel] >= 0.5) == np.asarray(y)[sel]))})
    return out


def trade_summary(T: pd.DataFrame, capital: float = 100_000) -> Dict[str, float]:
    if T is None or T.empty:
        return {"trades": 0}
    n = len(T)
    net = T["net_pnl"].astype(float)
    w = float((net > 0).mean())
    eq = pd.concat([pd.Series([float(capital)]), capital + net.cumsum()], ignore_index=True)
    lo, hi = bootstrap_mean_ci(net.values)
    months = T.assign(m=T["date"].astype(str).str[:7]).groupby("m")["net_pnl"].sum()
    wins, losses = net[net > 0], net[net <= 0]
    return {"trades": n, "period": [str(T["date"].min()), str(T["date"].max())],
            "win_rate": round(w, 4), "net_total": round(float(net.sum()), 0),
            "net_per_trade": round(float(net.mean()), 1), "net_per_trade_95ci": [round(lo, 1), round(hi, 1)],
            "t_stat": round(t_stat(net.values), 2), "gross_per_trade": round(float(T["gross_pnl"].mean()), 1),
            "charges_per_trade": round(float(T["charges"].mean()), 1),
            "avg_win": round(float(wins.mean()), 0) if len(wins) else 0.0,
            "avg_loss": round(float(losses.mean()), 0) if len(losses) else 0.0,
            "profit_factor": round(float(wins.sum() / -losses.sum()), 2) if len(losses) and losses.sum() < 0 else None,
            "worst_trade": round(float(net.min()), 0),
            "max_drawdown_pct": round(float((eq / eq.cummax() - 1).min() * 100), 1),
            "months": int(len(months)), "months_positive_pct": round(float((months > 0).mean() * 100), 1),
            "real_price_share": round(float((T.get("price_source", pd.Series(["?"])) == "real").mean()), 3)}
