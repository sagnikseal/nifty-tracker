"""Walk-forward training, calibration, honest evaluation and a champion/challenger registry.

Design choices (and the research behind them):
  * Expanding-window walk-forward with periodic refits; purge + embargo where labels overlap
    (Lopez de Prado 2018). No shuffling: the legacy v3 trainer shuffled rows, which mixes the
    future into training.
  * Small, regularised ensemble (L2 logistic + shallow gradient boosting). Daily index direction
    has a very low signal-to-noise ratio (Gu-Kelly-Xiu 2020: monthly R^2 ~0.4%); big models
    mostly fit noise.
  * Calibration is fitted only on *earlier* out-of-sample predictions, so reported
    probabilities are themselves out-of-sample.
  * Every result is compared with naive baselines and a binomial test; if the model cannot
    beat the base-rate log-loss, the engine reports NO_EDGE and the paper trader stands aside.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import DATA_DIR, EngineConfig
from .features import feature_columns

log = logging.getLogger(__name__)
REGISTRY = DATA_DIR / "models"


# ----------------------------------------------------------------------------- estimator
@dataclass
class ModelSpec:
    name: str = "ens_v1"
    use_logit: bool = True
    use_gbm: bool = True
    gbm_iter: int = 250
    gbm_depth: int = 3
    gbm_lr: float = 0.03
    logit_C: float = 0.05
    max_missing: float = 0.2
    members: tuple = ()          # non-empty -> zoo ensemble of these members (see zoo.py)
    combine: str = "mean"        # zoo combine: "mean" or "stack"


def make_model(spec: "ModelSpec"):
    if spec.members:
        from .zoo import ZooEnsemble
        return ZooEnsemble(list(spec.members), spec.combine, spec.max_missing)
    return Ensemble(spec)


CANDIDATES: List[ModelSpec] = [
    ModelSpec("ens_v1"),
    ModelSpec("logit_only", use_gbm=False),
    ModelSpec("gbm_only", use_logit=False, gbm_iter=300, gbm_depth=2),
]


class Ensemble:
    def __init__(self, spec: ModelSpec):
        self.spec = spec
        self.cols: List[str] = []
        self.models = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "Ensemble":
        miss = X.isna().mean()
        self.cols = [c for c in X.columns if miss[c] <= self.spec.max_missing and X[c].nunique() > 1]
        Xc = X[self.cols]
        self.models = []
        if self.spec.use_logit:
            m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                              LogisticRegression(C=self.spec.logit_C, max_iter=2000))
            m.fit(Xc, y)
            self.models.append(m)
        if self.spec.use_gbm:
            g = HistGradientBoostingClassifier(max_iter=self.spec.gbm_iter, max_depth=self.spec.gbm_depth,
                                               learning_rate=self.spec.gbm_lr, min_samples_leaf=60,
                                               l2_regularization=1.0, early_stopping=False, random_state=7)
            g.fit(Xc, y)
            self.models.append(g)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        Xc = X.reindex(columns=self.cols)
        ps = [m.predict_proba(Xc)[:, 1] for m in self.models]
        return np.clip(np.mean(ps, axis=0), 1e-4, 1 - 1e-4)

    def explain(self) -> Dict[str, float]:
        """Standardised logistic coefficients (direction + strength of each feature)."""
        for m in self.models:
            if hasattr(m, "named_steps"):
                coef = m.named_steps["logisticregression"].coef_[0]
                return dict(sorted(zip(self.cols, map(float, coef)), key=lambda kv: -abs(kv[1])))
        return {}


class PlattCalibrator:
    """1-D logistic map on the logit of raw probabilities, fitted on past OOS predictions."""

    def __init__(self):
        self.a, self.b, self.fitted = 1.0, 0.0, False

    def fit(self, p: np.ndarray, y: np.ndarray) -> "PlattCalibrator":
        if len(p) < 150 or len(np.unique(y)) < 2:
            return self
        z = np.log(p / (1 - p)).reshape(-1, 1)
        lr = LogisticRegression(C=1.0).fit(z, y)
        self.a, self.b, self.fitted = float(lr.coef_[0][0]), float(lr.intercept_[0]), True
        return self

    def __call__(self, p: np.ndarray) -> np.ndarray:
        if not self.fitted:
            return p
        z = np.log(p / (1 - p))
        return 1 / (1 + np.exp(-(self.a * z + self.b)))


# ----------------------------------------------------------------------------- walk-forward
def walk_forward(F: pd.DataFrame, cfg: EngineConfig, spec: ModelSpec,
                 start: Optional[str] = None, model_factory=None) -> pd.DataFrame:
    """Expanding-window walk-forward. Returns one OOS prediction per test day."""
    cols = feature_columns(F)
    from .zoo import set_context
    set_context(F[cols])
    data = F.dropna(subset=["y_ret"]).copy()
    data = data[data.index >= pd.Timestamp(cfg.history_start)]
    # Need ~200 sessions of warm-up for the slow features
    data = data.iloc[200:] if len(data) > 400 else data
    n = len(data)
    first_test = max(cfg.min_train_days, int(n * 0.3)) if n > 600 else int(n * 0.5)
    if start:
        first_test = max(first_test, int(np.searchsorted(data.index, pd.Timestamp(start))))
    purge = 1 if cfg.target == "cc" else 0
    dz = cfg.deadzone_bps / 1e4
    out, past_p, past_y = [], [], []
    i = first_test
    while i < n:
        j = min(n, i + cfg.retrain_every_days)
        train = data.iloc[: max(0, i - purge)]
        train = train[train["y_ret"].abs() >= dz]
        test = data.iloc[i:j]
        if len(train) < 150 or train["y_up"].nunique() < 2:
            i = j
            continue
        model = (model_factory() if model_factory else make_model(spec)).fit(train[cols], train["y_up"].astype(int))
        p_raw = model.predict_proba(test[cols])
        cal = PlattCalibrator().fit(np.array(past_p), np.array(past_y))
        p_cal = cal(p_raw)
        base_up = float(train["y_up"].mean())
        for k, d in enumerate(test.index):
            out.append({"date": d, "p_raw": float(p_raw[k]), "p": float(p_cal[k]),
                        "y_up": float(test["y_up"].iloc[k]), "y_ret": float(test["y_ret"].iloc[k]),
                        "base_up": base_up, "gap": float(test["gap"].iloc[k]) if "gap" in test else np.nan,
                        "r1": float(test["r1"].iloc[k]) if "r1" in test else np.nan,
                        "train_rows": len(train)})
        past_p.extend(p_raw.tolist())
        past_y.extend(test["y_up"].astype(int).tolist())
        i = j
    return pd.DataFrame(out).set_index("date") if out else pd.DataFrame()


# ----------------------------------------------------------------------------- metrics
def _logloss(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _binom_pvalue(k: int, n: int, p0: float) -> float:
    """One-sided P(X >= k) for X~Bin(n,p0) via normal approximation with continuity."""
    if n == 0:
        return 1.0
    mu, sd = n * p0, math.sqrt(n * p0 * (1 - p0))
    z = (k - 0.5 - mu) / max(sd, 1e-9)
    return float(0.5 * math.erfc(z / math.sqrt(2)))


def evaluate(oos: pd.DataFrame, thresholds=(0.5, 0.53, 0.55, 0.56, 0.58, 0.6)) -> Dict:
    if oos.empty:
        return {"n": 0}
    y = oos["y_up"].values.astype(int)
    p = oos["p"].values
    pred = (p >= 0.5).astype(int)
    base = oos["base_up"].values
    res = {
        "n_days": int(len(oos)),
        "period": [str(oos.index.min().date()), str(oos.index.max().date())],
        "accuracy": float(np.mean(pred == y)),
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else None,
        "brier": float(np.mean((p - y) ** 2)),
        "logloss": _logloss(p, y),
        "logloss_baserate": _logloss(base, y),
        "up_rate": float(y.mean()),
    }
    res["skill_vs_baserate"] = res["logloss_baserate"] - res["logloss"]
    # Diebold-Mariano style test on per-day log-loss differences (model vs base-rate forecast)
    pc, bc = np.clip(p, 1e-6, 1 - 1e-6), np.clip(base, 1e-6, 1 - 1e-6)
    d = -(y * np.log(bc) + (1 - y) * np.log(1 - bc)) + (y * np.log(pc) + (1 - y) * np.log(1 - pc))
    t = d.mean() / (d.std(ddof=1) / math.sqrt(len(d))) if d.std() > 0 else 0.0
    res["skill_t_stat"] = float(t)
    res["skill_p_value"] = float(0.5 * math.erfc(t / math.sqrt(2)))
    k = int(np.sum(pred == y))
    res["p_value_vs_coinflip"] = _binom_pvalue(k, len(y), 0.5)
    maj = max(res["up_rate"], 1 - res["up_rate"])
    res["p_value_vs_majority"] = _binom_pvalue(k, len(y), maj)

    base_preds = {
        "always_up": np.ones_like(y),
        "always_train_majority": (base >= 0.5).astype(int),
        "gap_continuation": (oos["gap"].values > 0).astype(int),
        "gap_fade": (oos["gap"].values < 0).astype(int),
        "yesterday_momentum": (oos["r1"].values > 0).astype(int),
    }
    res["baselines"] = {k2: float(np.mean(v == y)) for k2, v in base_preds.items()}

    th_rows = []
    for t in thresholds:
        m = (p >= t) | (p <= 1 - t)
        if m.sum() == 0:
            continue
        d = np.where(p[m] >= 0.5, 1, -1)
        th_rows.append({"threshold": t, "coverage": float(m.mean()), "n": int(m.sum()),
                        "hit_rate": float(np.mean((p[m] >= 0.5).astype(int) == y[m])),
                        "avg_ret_bps": float(np.mean(d * oos["y_ret"].values[m]) * 1e4)})
    res["by_threshold"] = th_rows
    yr = oos.assign(hit=(pred == y)).groupby(oos.index.year)["hit"].agg(["mean", "count"])
    res["by_year"] = {int(i): {"accuracy": float(r["mean"]), "n": int(r["count"])} for i, r in yr.iterrows()}
    # Calibration table (deciles)
    q = pd.qcut(p, 5, duplicates="drop")
    cal = pd.DataFrame({"p": p, "y": y}).groupby(q, observed=True).mean()
    res["calibration"] = [{"pred": float(a), "actual": float(b)} for a, b in zip(cal["p"], cal["y"])]
    return res


# ----------------------------------------------------------------------------- registry
def train_and_register(F: pd.DataFrame, cfg: EngineConfig, specs: Optional[List[ModelSpec]] = None,
                       recent_days: int = 500, promote_margin: float = 0.0005, name: str = "champion") -> Dict:
    """Run walk-forward for each candidate, pick the best on recent OOS log-loss, compare with
    the current champion and promote only if better. Then fit the winner on all data."""
    specs = specs or CANDIDATES
    REGISTRY.mkdir(parents=True, exist_ok=True)
    results = {}
    for spec in specs:
        oos = walk_forward(F, cfg, spec)
        if oos.empty:
            continue
        recent = oos.iloc[-recent_days:]
        results[spec.name] = {"spec": spec.__dict__, "oos": oos, "metrics": evaluate(oos),
                              "recent": evaluate(recent)}
        log.info("%s: acc=%.4f ll=%.5f base=%.5f", spec.name, results[spec.name]["metrics"]["accuracy"],
                 results[spec.name]["metrics"]["logloss"], results[spec.name]["metrics"]["logloss_baserate"])
    if not results:
        raise RuntimeError("Not enough data to train. Run `python -m nifty_engine update-data` first.")
    best = min(results, key=lambda k: results[k]["recent"]["logloss"])

    champ_file = REGISTRY / f"{name}.json"
    champion = json.loads(champ_file.read_text()) if champ_file.exists() else None
    promote = True
    if champion and champion.get("spec_name") in results and champion["spec_name"] != best:
        cur = results[champion["spec_name"]]["recent"]["logloss"]
        promote = results[best]["recent"]["logloss"] < cur - promote_margin
        if not promote:
            best = champion["spec_name"]

    r = results[best]
    spec = ModelSpec(**{**r["spec"], "members": tuple(r["spec"].get("members", ()))})
    cols = feature_columns(F)
    full = F.dropna(subset=["y_ret"])
    full = full[full["y_ret"].abs() >= cfg.deadzone_bps / 1e4]
    from .zoo import set_context
    set_context(F[cols])
    model = make_model(spec).fit(full[cols], full["y_up"].astype(int))
    oos = r["oos"]
    cal = PlattCalibrator().fit(oos["p_raw"].values, oos["y_up"].values.astype(int))
    has_edge = bool(r["recent"]["skill_vs_baserate"] > 0 and r["metrics"]["skill_vs_baserate"] > 0
                    and r["metrics"]["skill_p_value"] < 0.05)

    version = dt.datetime.now().strftime("%Y%m%d-%H%M%S") + ("" if name == "champion" else f"-{name}")
    vdir = REGISTRY / version
    vdir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "calibrator": cal, "target": cfg.target}, vdir / "model.joblib")
    oos.to_csv(vdir / "oos_predictions.csv")
    meta = {
        "version": version, "spec_name": best, "target": cfg.target, "has_edge": has_edge,
        "trained_on": [str(full.index.min().date()), str(full.index.max().date())],
        "n_train": int(len(full)), "features": model.cols, "explain": model.explain(),
        "metrics": r["metrics"], "recent_metrics": r["recent"],
        "candidates": {k: {"accuracy": v["metrics"]["accuracy"], "logloss": v["metrics"]["logloss"],
                           "recent_logloss": v["recent"]["logloss"]} for k, v in results.items()},
        "promoted": promote,
    }
    (vdir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    champ_file.write_text(json.dumps({"version": version, "spec_name": best, "has_edge": has_edge}, indent=2))
    return meta


def load_champion(name: str = "champion"):
    champ_file = REGISTRY / f"{name}.json"
    if not champ_file.exists():
        return None, None
    c = json.loads(champ_file.read_text())
    vdir = REGISTRY / c["version"]
    bundle = joblib.load(vdir / "model.joblib")
    meta = json.loads((vdir / "meta.json").read_text())
    return bundle, meta


def predict_today(F: pd.DataFrame, bundle, meta) -> Dict:
    """Score the latest row of F (features for the decision day)."""
    row = F.iloc[[-1]]
    from .zoo import set_context
    set_context(F[feature_columns(F)])
    p_raw = bundle["model"].predict_proba(row[feature_columns(F)])
    p = float(bundle["calibrator"](p_raw)[0])
    return {"date": str(row.index[-1].date()), "p_up": p, "p_raw": float(p_raw[0]),
            "model_version": meta["version"], "has_edge": meta["has_edge"]}
