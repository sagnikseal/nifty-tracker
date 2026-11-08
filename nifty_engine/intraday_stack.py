"""Movement model zoo: tabular + sequence models (+ time-series foundation-model
features), trained walk-forward. Built to run on the Mac (Apple-silicon GPU via MPS):

    python -m nifty_engine intraday-zoo                # full: 4 tabular + 3 sequence members, monthly refits
    python -m nifty_engine intraday-zoo --quick        # quarterly refits, 1 seed (fast check)
    python -m nifty_engine intraday-zoo --tsfm chronos-bolt-small,timesfm-2.5   # add foundation-model features

Members
  tabular   CatBoost, XGBoost on the full feature frame (189 features: price, technicals,
            multi-timeframe, candles, structure, Bank Nifty, breadth, futures, option chain, global, macro, events)
  sequence  TCN, PatchTST, GRU on the last 48 five-minute bars of NIFTY, VIX, Bank Nifty and NIFTY futures
            (intraday_zoo.SeqMember), with the strongest tabular features concatenated after the encoder
  vote      soft majority vote: each month only members whose trailing out-of-sample log-loss beats the base rate
            vote, and their calibrated probabilities are averaged (no learned weights); base rate if nobody qualifies
Every member and the vote are reported with the same day-clustered tests and the untouched holdout.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config import EngineConfig
from .intraday_model import (HOLDOUT_START, HORIZONS, REGISTRY, REPORTS, INTRADAY_GROUPS, Platt, _ll, build_training_frame_groups,
                             columns_for, day_clustered_test, evaluate, load_series, walk_forward)

log = logging.getLogger(__name__)
SEQ_SERIES = ("nifty", "vix", "banknifty", "niftyfut")


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def add_tsfm(X: pd.DataFrame, models: List[str], close5: pd.Series) -> pd.DataFrame:
    from .intraday_zoo import tsfm_features
    for m in models:
        try:
            T = tsfm_features(close5, X.index, model=m)
            X = X.join(T)
            log.info("TSFM %s: %d columns", m, T.shape[1])
        except ImportError as e:
            log.warning("TSFM %s skipped: %s", m, e)
    return X


def seq_oos(X: pd.DataFrame, label: str, archs, series, warmup_days: int, refit_days: int, quick: bool,
            tab_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """Raw out-of-sample probabilities of each sequence member, aligned to X's rows (NaN before warm-up)."""
    from .intraday_zoo import SeqMember, build_panel, build_sequences, walk_forward_member
    data = X.dropna(subset=[label])
    panel = build_panel({k: series[k] for k in SEQ_SERIES if k in series and len(series[k])})
    seq = build_sequences(panel, data.index, lookback=48)
    tab = None
    if tab_cols:
        tab = data[tab_cols].astype(float).to_numpy()
        tab = np.where(np.isfinite(tab), tab, np.nan)
        med = np.nanmedian(tab, axis=0)
        tab = np.where(np.isnan(tab), med, tab).astype(np.float32)
    days = data["date"].values
    y = data[label].astype(int).to_numpy()
    out = pd.DataFrame(index=data.index)
    for arch in archs:
        t0 = time.time()
        mk = (lambda a=arch: SeqMember(arch=a, quick=quick, seeds=1 if quick else 3))
        o = walk_forward_member(seq, y, days, mk, tab=tab, warmup_days=warmup_days, refit_days=refit_days)
        col = np.full(len(data), np.nan)
        col[o["row"].to_numpy(int)] = o["p"].to_numpy(float)
        out[f"p_seq_{arch}"] = col
        log.info("sequence member %s (%s): %d OOS rows in %.0f s", arch, label, len(o), time.time() - t0)
    return out


def vote_oos(M: pd.DataFrame, y: pd.Series, days: pd.Series, block_days: int = 21, trail_rows: int = 8000,
             min_rows: int = 3000) -> pd.DataFrame:
    """Soft majority vote, walk-forward. For each block, every member is calibrated on its own earlier
    out-of-sample predictions; only members whose log-loss over the trailing `trail_rows` rows beats the base
    rate get a vote; the prediction is the average of the voters' probabilities, and `agree` is the share of
    voters on the same side. No qualified voter -> base rate (no trade). No learned weights: with weak,
    noisy members a plain average beats a fitted combiner (the daily zoo measured exactly that)."""
    cols = list(M.columns)
    ud = np.array(sorted(days.unique()))
    out = pd.DataFrame(index=M.index, columns=["p_vote", "agree", "voters"], dtype=object)
    for i in range(0, len(ud), block_days):
        blk = ud[i:i + block_days]
        te = days.isin(blk).values
        past = (days < blk[0]).values
        P = M[past].dropna(how="any")
        if len(P) < min_rows:
            continue
        T = P.iloc[-trail_rows:]
        old = P.iloc[:-trail_rows] if len(P) > trail_rows + 2000 else P
        yt, yo = y.loc[T.index].astype(int).values, y.loc[old.index].astype(int).values
        base = float(y.loc[P.index].mean())
        base_ll = float(_ll(np.full(len(yt), base), yt).mean())
        voters = [c for c in cols if float(_ll(Platt().fit(old[c].values, yo)(T[c].values), yt).mean()) < base_ll]
        idx = M.index[te]
        if not voters:
            out.loc[idx, "p_vote"], out.loc[idx, "voters"] = base, "base"
            continue
        yp = y.loc[P.index].astype(int).values
        Q = np.column_stack([Platt().fit(P[c].values, yp)(M.loc[idx, c].fillna(0.5).values) for c in voters])
        pv = Q.mean(axis=1)
        out.loc[idx, "p_vote"] = pv
        out.loc[idx, "agree"] = ((Q >= 0.5) == (pv >= 0.5)[:, None]).mean(axis=1)
        out.loc[idx, "voters"] = "+".join(voters)
    return out


def run(cfg: EngineConfig, groups=INTRADAY_GROUPS, tab_members=("cat", "xgb"),
        seq_archs=("tcn", "patchtst", "gru"), tsfm: Optional[List[str]] = None, quick: bool = False,
        X: Optional[pd.DataFrame] = None) -> Dict:
    refit = 63 if quick else 21
    X = build_training_frame_groups(cfg, groups) if X is None else X
    series = load_series()
    if tsfm:
        X = add_tsfm(X, tsfm, series["nifty"]["close"])
        groups = list(groups) + ["tsfm"]
    cols = columns_for(X, groups)
    report: Dict = {"members": {}, "vote": {}, "groups": list(groups), "n_features": len(cols), "quick": quick,
                    "tab_members": list(tab_members), "seq_archs": list(seq_archs), "tsfm": tsfm or []}
    oos_store = {}
    for h, label in HORIZONS.items():
        t0 = time.time()
        tab = walk_forward(X, label, members=tab_members, refit_days=refit, cols=cols, max_missing=0.9)
        M = tab[[f"p_{m}" for m in tab_members]].copy()
        if seq_archs:
            try:
                top = _top_tab_features(X, cols, label, k=24)
                S = seq_oos(X, label, seq_archs, series, warmup_days=250, refit_days=refit, quick=quick, tab_cols=top)
                M = M.join(S.reindex(M.index))
            except ImportError as e:
                log.warning("sequence members skipped (install torch): %s", e)
        y = tab["y"]
        days = tab["date"]
        st = vote_oos(M, y, days, block_days=refit)
        for c in M.columns:
            o = tab[["date", "y", "ret", "base", "t_min", "r_30m"]].copy()
            o["p"] = Platt().fit(M[c].dropna().values[:5000], y[M[c].notna()].values[:5000])(M[c].values) \
                if M[c].notna().sum() > 6000 else M[c].values
            o = o.dropna(subset=["p"])
            report["members"][f"{h}|{c[2:]}"] = _summ(o)
        o = tab[["date", "y", "ret", "base", "t_min", "r_30m"]].copy()
        o["p"] = pd.to_numeric(st["p_vote"], errors="coerce")
        o = o.dropna(subset=["p"])
        report["vote"][f"{h}|vote"] = _summ(o)
        report["vote"][f"{h}|voters_history"] = st["voters"].dropna().groupby(
            pd.to_datetime(days[st["voters"].notna()]).dt.year).agg(lambda s: s.value_counts().to_dict()).to_dict()
        oos_store[h] = pd.concat([tab, M.drop(columns=[c for c in M.columns if c in tab]), st], axis=1)
        log.info("zoo %s done in %.0f s", h, time.time() - t0)
    version = "zoo-" + dt.datetime.now().strftime("%Y%m%d-%H%M")
    vdir = REGISTRY / version
    vdir.mkdir(parents=True, exist_ok=True)
    for h, o in oos_store.items():
        o.to_csv(vdir / f"oos_{h}.csv.gz")
    (vdir / "zoo_report.json").write_text(json.dumps(report, indent=1, default=str))
    write_report(report, REPORTS / "INTRADAY_ZOO_REPORT.md")
    return report


def _top_tab_features(X, cols, label, k=24) -> List[str]:
    """Most useful tabular features by LightGBM gain on the first 40% of days only (no look-ahead into OOS)."""
    import lightgbm as lgb
    d = X.dropna(subset=[label])
    cut = np.array(sorted(d["date"].unique()))[int(d["date"].nunique() * 0.4)]
    tr = d[d["date"] < cut]
    m = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.03, num_leaves=7, min_child_samples=2000,
                           colsample_bytree=0.6, verbose=-1).fit(tr[cols], tr[label].astype(int))
    imp = pd.Series(m.booster_.feature_importance("gain"), index=cols).sort_values(ascending=False)
    return list(imp.index[:k])


def _summ(o: pd.DataFrame) -> Dict:
    r = evaluate(o)
    keep = {k: r.get(k) for k in ("n", "days", "accuracy", "auc", "logloss_skill", "p_value", "share_above_80")}
    ho = o[o["date"] >= pd.Timestamp(HOLDOUT_START)]
    if len(ho):
        rh = evaluate(ho)
        keep.update({"holdout_accuracy": rh.get("accuracy"), "holdout_skill": rh.get("logloss_skill"),
                     "holdout_p": rh.get("p_value")})
    keep["hit_rate_at_0.60"] = next((t["hit_rate"] for t in r.get("by_threshold", []) if abs(t["threshold"] - 0.6) < 1e-9), None)
    keep["coverage_at_0.60"] = next((t["coverage"] for t in r.get("by_threshold", []) if abs(t["threshold"] - 0.6) < 1e-9), None)
    return keep


def write_report(rep: Dict, path) -> None:
    def f(v, pct=False, nd=3):
        if v is None or (isinstance(v, float) and v != v):
            return "—"
        return f"{v:.2%}" if pct else (f"{v:+.5f}" if nd == 5 else f"{v:.{nd}g}")
    L = ["# Movement model zoo: every member and the vote, out-of-sample", "",
         f"Feature groups: {', '.join(rep['groups'])} ({rep['n_features']} features). Tabular: {rep['tab_members']}. "
         f"Sequence: {rep['seq_archs']}. Foundation-model features: {rep['tsfm'] or 'none'}. "
         f"{'Quick run (quarterly refits).' if rep['quick'] else 'Monthly refits.'}", "",
         "Skill = log-loss improvement over always predicting the historical up-rate; p is day-clustered. "
         "A member is only useful with positive skill and p < 0.05, confirmed on the holdout.", "",
         "| Horizon | Model | Days | Accuracy | AUC | Skill | p | Holdout acc | Holdout skill | Hit rate at ≥60% (coverage) | ≥80% share |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for part in ("members", "vote"):
        for k, r in rep[part].items():
            if not isinstance(r, dict) or "accuracy" not in r or r.get("n") in (None, 0):
                continue
            h, name = k.split("|")
            L.append(f"| {h} | {'**' + name + '**' if part == 'vote' else name} | {r['days']} | {f(r['accuracy'], True)} | "
                     f"{f(r['auc'])} | {f(r['logloss_skill'], nd=5)} | {f(r['p_value'])} | {f(r.get('holdout_accuracy'), True)} | "
                     f"{f(r.get('holdout_skill'), nd=5)} | {f(r.get('hit_rate_at_0.60'), True)} ({f(r.get('coverage_at_0.60'), True)}) | "
                     f"{f(r.get('share_above_80'), True)} |")
    L += ["", "Which members had a vote each year (trailing log-loss better than the base rate):", ""]
    for k, v in rep["vote"].items():
        if k.endswith("voters_history"):
            L.append(f"* {k.split('|')[0]}: {v}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L), encoding="utf-8")
