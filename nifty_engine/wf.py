"""Date-grouped walk-forward (several rows per session allowed: never splits a day)."""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from .model import Ensemble, ModelSpec, PlattCalibrator


def walk_forward_grouped(X: pd.DataFrame, cols: List[str], spec: ModelSpec, first_test_days: int = 250,
                         step_days: int = 21, deadzone: float = 0.0, max_train_days: Optional[int] = None,
                         model_factory=None) -> pd.DataFrame:
    X = X.sort_index()
    dates = np.array(sorted(X.index.unique()))
    out, past_p, past_y = [], [], []
    i = first_test_days
    while i < len(dates):
        j = min(len(dates), i + step_days)
        start, end = dates[i], dates[j - 1]
        lo = dates[max(0, i - max_train_days)] if max_train_days else dates[0]
        tr = X[(X.index < start) & (X.index >= lo)]
        tr = tr[tr["y_ret"].abs() >= deadzone]
        te = X[(X.index >= start) & (X.index <= end)]
        if len(tr) < 200 or tr["y_up"].nunique() < 2:
            i = j
            continue
        m = (model_factory() if model_factory else Ensemble(spec)).fit(tr[cols], tr["y_up"].astype(int))
        p_raw = m.predict_proba(te[cols])
        cal = PlattCalibrator().fit(np.array(past_p), np.array(past_y))
        o = te[["y_ret", "y_up"]].copy()
        for c in ("T", "px_T", "px_exit", "gap", "r_open_T", "r_first30", "r1"):
            if c in te:
                o[c] = te[c]
        o["p_raw"], o["p"] = p_raw, cal(p_raw)
        o["base_up"] = float(tr["y_up"].mean())
        out.append(o)
        past_p.extend(p_raw.tolist())
        past_y.extend(te["y_up"].astype(int).tolist())
        i = j
    return pd.concat(out) if out else pd.DataFrame()


def acc_table(o: pd.DataFrame, thresholds=(0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8)) -> pd.DataFrame:
    rows = []
    for t in thresholds:
        m = (o["p"] >= t) | (o["p"] <= 1 - t)
        if m.sum() == 0:
            continue
        pred = (o.loc[m, "p"] >= 0.5).astype(int)
        rows.append({"thr": t, "n": int(m.sum()), "coverage": round(float(m.mean()), 3),
                     "acc": round(float((pred == o.loc[m, "y_up"]).mean()), 4),
                     "avg_bps": round(float((np.where(pred == 1, 1, -1) * o.loc[m, "y_ret"]).mean() * 1e4), 2)})
    return pd.DataFrame(rows)
