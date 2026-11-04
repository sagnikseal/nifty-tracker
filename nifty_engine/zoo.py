"""Model zoo + ensembles.

Members (any that can be imported are used):
  sklearn : logit, hgb, rf, et, mlp, svm
  boosters: xgb, lgbm, cat               (pip install xgboost lightgbm catboost)
  deep    : lstm, cnn, transformer         (pip install torch; uses CUDA, Apple MPS or CPU)

Sequence models look at the last `SEQ_LEN` sessions of features. They get those rows from a
*context* frame of features only (no labels). A feature row dated d is known by 09:20 on d, so a
window ending at d uses no future information.

Ensembles (built in `run_zoo`, all walk-forward / leak-free):
  mean_all  : average of every member's probability
  stack     : logistic meta-model on members' logits, fitted only on earlier OOS predictions
  best3     : average of the 3 members with the best trailing OOS log-loss (re-selected each block)
"""

from __future__ import annotations

import json
import logging
import math
import time
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .config import DATA_DIR

log = logging.getLogger(__name__)
SEQ_LEN = 20
_CONTEXT: Optional[pd.DataFrame] = None


def set_context(feats: pd.DataFrame) -> None:
    """Full feature history (sorted by date) used by sequence models to build windows."""
    global _CONTEXT
    _CONTEXT = feats.sort_index()


# ----------------------------------------------------------------------------- tabular members
def _imp_scale(model):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), model)


def _sk_members() -> Dict[str, Callable]:
    return {
        "logit": lambda: _imp_scale(LogisticRegression(C=0.05, max_iter=2000)),
        "hgb": lambda: HistGradientBoostingClassifier(max_iter=250, max_depth=3, learning_rate=0.03,
                                                      min_samples_leaf=60, l2_regularization=1.0,
                                                      early_stopping=False, random_state=7),
        "rf": lambda: make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(
            n_estimators=400, max_depth=6, min_samples_leaf=40, max_features=0.3, n_jobs=-1, random_state=7)),
        "et": lambda: make_pipeline(SimpleImputer(strategy="median"), ExtraTreesClassifier(
            n_estimators=400, max_depth=6, min_samples_leaf=40, max_features=0.3, n_jobs=-1, random_state=7)),
        "mlp": lambda: _imp_scale(MLPClassifier(hidden_layer_sizes=(32, 16), alpha=0.05, learning_rate_init=1e-3,
                                                max_iter=400, early_stopping=True, validation_fraction=0.15,
                                                random_state=7)),
        "svm": lambda: _imp_scale(SVC(C=0.3, kernel="rbf", gamma="scale", probability=True, random_state=7)),
    }


def _booster_members() -> Dict[str, Callable]:
    out = {}
    try:
        from xgboost import XGBClassifier
        out["xgb"] = lambda: XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8,
                                           colsample_bytree=0.7, reg_lambda=1.0, min_child_weight=20,
                                           eval_metric="logloss", n_jobs=-1, random_state=7, verbosity=0)
    except Exception:  # noqa: BLE001
        pass
    try:
        from lightgbm import LGBMClassifier
        out["lgbm"] = lambda: LGBMClassifier(n_estimators=300, num_leaves=8, learning_rate=0.03,
                                             min_child_samples=60, subsample=0.8, subsample_freq=1,
                                             colsample_bytree=0.7, reg_lambda=1.0, random_state=7, verbose=-1)
    except Exception:  # noqa: BLE001
        pass
    try:
        from catboost import CatBoostClassifier
        out["cat"] = lambda: CatBoostClassifier(iterations=400, depth=4, learning_rate=0.03, l2_leaf_reg=5,
                                                random_seed=7, verbose=0, thread_count=-1)
    except Exception:  # noqa: BLE001
        pass
    return out


# ----------------------------------------------------------------------------- deep members
def torch_device():
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class TorchSeqClassifier:
    """LSTM / 1-D CNN / small Transformer over the last SEQ_LEN sessions of features."""

    def __init__(self, kind: str = "lstm", hidden: int = 32, epochs: int = 40, lr: float = 1e-3,
                 weight_decay: float = 1e-3, dropout: float = 0.2, seed: int = 7):
        self.kind, self.hidden, self.epochs, self.lr = kind, hidden, epochs, lr
        self.wd, self.dropout, self.seed = weight_decay, dropout, seed

    # windows -------------------------------------------------------------
    def _windows(self, X: pd.DataFrame) -> np.ndarray:
        if _CONTEXT is None:
            raise RuntimeError("zoo.set_context(features) must be called before using sequence models")
        ctx = _CONTEXT[X.columns]
        pos = ctx.index.get_indexer(X.index)
        arr = ctx.to_numpy(dtype=np.float32)
        arr = (arr - self.mu) / self.sd
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        W = np.zeros((len(X), SEQ_LEN, arr.shape[1]), dtype=np.float32)
        for i, p in enumerate(pos):
            if p < 0:
                row = np.nan_to_num(((X.iloc[i].to_numpy(dtype=np.float32) - self.mu) / self.sd), nan=0.0)
                W[i, -1] = row
                continue
            lo = max(0, p - SEQ_LEN + 1)
            seg = arr[lo:p + 1]
            W[i, SEQ_LEN - len(seg):] = seg
        return W

    def _net(self, k: int):
        import torch.nn as nn

        h, dp = self.hidden, self.dropout

        class LSTMNet(nn.Module):
            def __init__(s):
                super().__init__()
                s.rnn = nn.LSTM(k, h, batch_first=True)
                s.head = nn.Sequential(nn.Dropout(dp), nn.Linear(h, 1))

            def forward(s, x):
                o, _ = s.rnn(x)
                return s.head(o[:, -1]).squeeze(-1)

        class CNNNet(nn.Module):
            def __init__(s):
                super().__init__()
                s.c = nn.Sequential(nn.Conv1d(k, h, 3, padding=1), nn.ReLU(), nn.Dropout(dp),
                                    nn.Conv1d(h, h, 3, padding=2, dilation=2), nn.ReLU())
                s.head = nn.Linear(h, 1)

            def forward(s, x):
                z = s.c(x.transpose(1, 2))
                return s.head(z[:, :, -5:].mean(-1)).squeeze(-1)

        class TfNet(nn.Module):
            def __init__(s):
                super().__init__()
                s.inp = nn.Linear(k, h)
                s.pos = nn.Parameter(np.float32(0.02) * __import__("torch").randn(1, SEQ_LEN, h))
                layer = nn.TransformerEncoderLayer(h, 4, 2 * h, dp, batch_first=True)
                s.enc = nn.TransformerEncoder(layer, 2)
                s.head = nn.Linear(h, 1)

            def forward(s, x):
                z = s.enc(s.inp(x) + s.pos)
                return s.head(z[:, -1]).squeeze(-1)

        return {"lstm": LSTMNet, "cnn": CNNNet, "transformer": TfNet}[self.kind]()

    def fit(self, X: pd.DataFrame, y) -> "TorchSeqClassifier":
        import torch

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        ctx = _CONTEXT[X.columns]
        hist = ctx[ctx.index <= X.index.max()].to_numpy(dtype=np.float32)
        self.mu = np.nanmean(hist, axis=0)
        sd = np.nanstd(hist, axis=0)
        self.sd = np.where((sd < 1e-8) | np.isnan(sd), 1.0, sd).astype(np.float32)
        self.mu = np.nan_to_num(self.mu).astype(np.float32)
        Xw = self._windows(X)
        yv = np.asarray(y, dtype=np.float32)
        n = len(Xw)
        cut = int(n * 0.85)                      # time-ordered validation tail for early stopping
        dev = torch_device()
        self.dev = dev
        net = self._net(Xw.shape[2]).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.wd)
        lossf = torch.nn.BCEWithLogitsLoss()
        Xt, yt = torch.tensor(Xw[:cut]), torch.tensor(yv[:cut])
        Xv, yv_t = torch.tensor(Xw[cut:]).to(dev), torch.tensor(yv[cut:]).to(dev)
        best, best_state, bad = 1e9, None, 0
        g = torch.Generator().manual_seed(self.seed)
        for ep in range(self.epochs):
            net.train()
            perm = torch.randperm(len(Xt), generator=g)
            for i in range(0, len(Xt), 128):
                b = perm[i:i + 128]
                xb, yb = Xt[b].to(dev), yt[b].to(dev)
                opt.zero_grad()
                lossf(net(xb), yb).backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
            net.eval()
            with torch.no_grad():
                vl = float(lossf(net(Xv), yv_t)) if len(Xv) else 0.0
            if vl < best - 1e-4:
                best, bad = vl, 0
                best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= 6:
                    break
        if best_state is not None:
            net.load_state_dict(best_state)
        self.net = net.cpu()
        self.dev = torch.device("cpu")
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        import torch
        self.net.eval()
        with torch.no_grad():
            p = torch.sigmoid(self.net(torch.tensor(self._windows(X)))).numpy()
        return np.vstack([1 - p, p]).T


class TabMClassifier:
    """TabM (Gorishniy et al., ICLR 2025): parameter-efficient ensemble of 32 MLPs, state of the art
    among tabular deep models. pip install tabm. Uses CUDA / Apple MPS when available."""

    def __init__(self, k: int = 32, epochs: int = 120, lr: float = 2e-3, wd: float = 3e-4, seed: int = 7):
        self.k, self.epochs, self.lr, self.wd, self.seed = k, epochs, lr, wd, seed

    def _prep(self, X: pd.DataFrame) -> np.ndarray:
        a = (X.to_numpy(dtype=np.float32) - self.mu) / self.sd
        return np.clip(np.nan_to_num(a, nan=0.0), -6, 6).astype(np.float32)

    def fit(self, X: pd.DataFrame, y) -> "TabMClassifier":
        import tabm
        import torch
        torch.manual_seed(self.seed)
        a = X.to_numpy(dtype=np.float32)
        self.mu = np.nan_to_num(np.nanmean(a, axis=0)).astype(np.float32)
        sd = np.nanstd(a, axis=0)
        self.sd = np.where((sd < 1e-8) | np.isnan(sd), 1.0, sd).astype(np.float32)
        Xa, yv = self._prep(X), np.asarray(y, dtype=np.float32)
        cut = int(len(Xa) * 0.85)
        dev = torch_device()
        net = tabm.TabM.make(n_num_features=Xa.shape[1], cat_cardinalities=None, d_out=1, k=self.k).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.wd)
        lossf = torch.nn.BCEWithLogitsLoss()
        Xt, yt = torch.tensor(Xa[:cut]), torch.tensor(yv[:cut])
        Xv, yvt = torch.tensor(Xa[cut:]).to(dev), torch.tensor(yv[cut:]).to(dev)
        best, state, bad = 1e9, None, 0
        g = torch.Generator().manual_seed(self.seed)
        for _ in range(self.epochs):
            net.train()
            perm = torch.randperm(len(Xt), generator=g)
            for i in range(0, len(Xt), 256):
                b = perm[i:i + 256]
                out = net(Xt[b].to(dev)).squeeze(-1)                     # (B, k)
                loss = lossf(out, yt[b].to(dev)[:, None].expand_as(out))  # train every member
                opt.zero_grad()
                loss.backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                pv = torch.sigmoid(net(Xv).squeeze(-1)).mean(1).clamp(1e-6, 1 - 1e-6)
                vl = float(-(yvt * pv.log() + (1 - yvt) * (1 - pv).log()).mean()) if len(Xv) else 0.0
            if vl < best - 1e-4:
                best, bad = vl, 0
                state = {k_: v.detach().cpu().clone() for k_, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= 10:
                    break
        if state is not None:
            net.load_state_dict(state)
        self.net = net.cpu()
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        import torch
        self.net.eval()
        with torch.no_grad():
            p = torch.sigmoid(self.net(torch.tensor(self._prep(X))).squeeze(-1)).mean(1).numpy()
        return np.vstack([1 - p, p]).T


class TabPFNMember:
    """TabPFN (Hollmann et al., Nature 2025): pretrained transformer for small tabular data
    (<= 10k rows, our case). Weights are gated on Hugging Face: accept the licence at
    huggingface.co/Prior-Labs and set HF_TOKEN. Fast on GPU; ~1-2 h per walk-forward on CPU."""

    def fit(self, X: pd.DataFrame, y) -> "TabPFNMember":
        from tabpfn import TabPFNClassifier
        dev = str(torch_device())
        self.m = TabPFNClassifier(device="cuda" if dev == "cuda" else ("mps" if dev == "mps" else "cpu"),
                                  n_estimators=4, random_state=7)
        self.m.fit(X.to_numpy(dtype=np.float32), np.asarray(y).astype(int))
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.m.predict_proba(X.to_numpy(dtype=np.float32))


def _deep_members() -> Dict[str, Callable]:
    try:
        import torch  # noqa: F401
    except Exception:  # noqa: BLE001
        return {}
    kinds = ["lstm", "cnn"]
    gpu = str(torch_device()) != "cpu"
    if gpu:                                # transformer is ~40+ min per walk-forward on a laptop CPU
        kinds.append("transformer")
    out = {k: (lambda k=k: TorchSeqClassifier(k)) for k in kinds}
    try:
        import tabm  # noqa: F401
        out["tabm"] = lambda: TabMClassifier()
    except Exception:  # noqa: BLE001
        pass
    import os
    try:
        import tabpfn  # noqa: F401
        if os.environ.get("HF_TOKEN") or gpu:
            out["tabpfn"] = lambda: TabPFNMember()
    except Exception:  # noqa: BLE001
        pass
    return out


def available_members(include_deep: bool = True) -> Dict[str, Callable]:
    m = {**_sk_members(), **_booster_members()}
    if include_deep:
        m.update(_deep_members())
    return m


# ----------------------------------------------------------------------------- combiners
def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def _ll(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


class ZooEnsemble:
    """Deployable ensemble: fits the chosen members on all rows; combine='mean' averages them,
    combine='stack' fits a logistic meta-model on an inner time-ordered hold-out (last 30%)."""

    def __init__(self, members: List[str], combine: str = "mean", max_missing: float = 0.2):
        self.members, self.combine, self.max_missing = list(members), combine, max_missing
        self.cols: List[str] = []

    def fit(self, X: pd.DataFrame, y) -> "ZooEnsemble":
        miss = X.isna().mean()
        self.cols = [c for c in X.columns if miss[c] <= self.max_missing and X[c].nunique() > 1]
        Xc, yv = X[self.cols], np.asarray(y).astype(int)
        reg = available_members(include_deep=True)
        self.members = [m for m in self.members if m in reg]
        self.meta = None
        if self.combine == "stack" and len(Xc) > 600:
            cut = int(len(Xc) * 0.7)
            Z = []
            for m in self.members:
                mdl = reg[m]().fit(Xc.iloc[:cut], yv[:cut])
                Z.append(_logit(mdl.predict_proba(Xc.iloc[cut:])[:, 1]))
            self.meta = LogisticRegression(C=1.0).fit(np.vstack(Z).T, yv[cut:])
        self.models = [reg[m]().fit(Xc, yv) for m in self.members]
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        Xc = X.reindex(columns=self.cols)
        ps = [m.predict_proba(Xc)[:, 1] for m in self.models]
        if self.meta is not None:
            p = self.meta.predict_proba(np.vstack([_logit(q) for q in ps]).T)[:, 1]
        else:
            p = np.mean(ps, axis=0)
        return np.clip(p, 1e-4, 1 - 1e-4)

    def explain(self) -> Dict[str, float]:
        """Logistic coefficients if a logit member exists, else averaged tree importances."""
        for m in self.models:
            if hasattr(m, "named_steps") and "logisticregression" in m.named_steps:
                coef = m.named_steps["logisticregression"].coef_[0]
                return dict(sorted(zip(self.cols, map(float, coef)), key=lambda kv: -abs(kv[1])))
        imps = []
        for m in self.models:
            est = m.steps[-1][1] if hasattr(m, "steps") else m
            fi = getattr(est, "feature_importances_", None)
            if fi is not None and len(fi) == len(self.cols):
                fi = np.asarray(fi, dtype=float)
                imps.append(fi / max(fi.sum(), 1e-12))
        if not imps:
            return {}
        avg = np.mean(imps, axis=0)
        return dict(sorted(zip(self.cols, map(float, avg)), key=lambda kv: -kv[1]))


# ----------------------------------------------------------------------------- research runner
def run_zoo(F: pd.DataFrame, cfg, members: Optional[List[str]] = None, include_deep: bool = True,
            out_dir=None) -> Dict:
    """Walk-forward every member (cached per member so long runs can resume), then build
    mean / stacked / trailing-best ensembles from the members' OOS predictions only."""
    from .features import feature_columns
    from .model import PlattCalibrator, evaluate, walk_forward, ModelSpec

    out_dir = out_dir or (DATA_DIR / "zoo")
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = feature_columns(F)
    set_context(F[cols])
    reg = available_members(include_deep)
    members = [m for m in (members or list(reg)) if m in reg or (out_dir / f"oos_{m}.csv").exists()]
    oos: Dict[str, pd.DataFrame] = {}
    timing = {}
    for m in members:
        f = out_dir / f"oos_{m}.csv"
        if f.exists():  # cached (e.g. trained on another machine): reuse
            oos[m] = pd.read_csv(f, index_col=0, parse_dates=True)
            continue
        t0 = time.time()
        o = walk_forward(F, cfg, ModelSpec(f"zoo_{m}"), model_factory=lambda m=m: ZooEnsemble([m]))
        timing[m] = round(time.time() - t0, 1)
        o.to_csv(f)
        oos[m] = o
        log.info("zoo member %s done in %ss: acc %.4f", m, timing[m], float(((o.p >= .5) == o.y_up).mean()))
    base = oos[members[0]]
    P = pd.DataFrame({m: oos[m]["p_raw"] for m in members}).dropna()
    y = base.loc[P.index, "y_up"].astype(int)
    blocks = np.array_split(np.arange(len(P)), max(1, len(P) // 21))
    ens = {"mean_all": [], "stack": [], "best3": []}
    for b in blocks:
        hist = P.index[: b[0]]
        Pb = P.iloc[b]
        ens["mean_all"].append(Pb.mean(axis=1))
        if len(hist) >= 250:
            Z = np.vstack([_logit(P.loc[hist, m].values) for m in members]).T
            meta = LogisticRegression(C=0.5).fit(Z, y.loc[hist].values)
            ens["stack"].append(pd.Series(meta.predict_proba(np.vstack([_logit(Pb[m].values) for m in members]).T)[:, 1], index=Pb.index))
            recent = hist[-500:]
            scores = {m: _ll(P.loc[recent, m].values, y.loc[recent].values) for m in members}
            top = sorted(scores, key=scores.get)[:3]
            ens["best3"].append(Pb[top].mean(axis=1))
        else:
            ens["stack"].append(Pb.mean(axis=1))
            ens["best3"].append(Pb.mean(axis=1))
    results = {}
    for name, parts in ens.items():
        p_raw = pd.concat(parts).loc[P.index]
        # leak-free calibration on earlier predictions
        p_cal, past = [], []
        for b in blocks:
            idx = P.index[b]
            if b[0] >= 250:
                cal = PlattCalibrator().fit(p_raw.iloc[: b[0]].values, y.iloc[: b[0]].values)
                p_cal.append(pd.Series(cal(p_raw.loc[idx].values), index=idx))
            else:
                p_cal.append(p_raw.loc[idx])
        o = base.loc[P.index].copy()
        o["p_raw"], o["p"] = p_raw, pd.concat(p_cal).loc[P.index]
        o.to_csv(out_dir / f"oos_{name}.csv")
        oos[name] = o
    for name, o in oos.items():
        m = evaluate(o)
        conf = [r for r in m["by_threshold"] if abs(r["threshold"] - 0.56) < 1e-9]
        rec = o[o.index >= "2022-01-01"]
        results[name] = {"accuracy": round(m["accuracy"], 4), "auc": round(m["auc"], 4),
                         "logloss": round(m["logloss"], 5), "skill_p": round(m["skill_p_value"], 4),
                         "acc_2022plus": round(float(((rec.p >= .5) == rec.y_up).mean()), 4),
                         "conf56_acc": round(conf[0]["hit_rate"], 4) if conf else None,
                         "conf56_n": conf[0]["n"] if conf else 0,
                         "fit_seconds": timing.get(name)}
    (out_dir / "zoo_metrics.json").write_text(json.dumps(results, indent=2))
    return {"metrics": results, "oos": oos}


def promote_best3(F: pd.DataFrame, cfg, zoo_result: Dict, recent_days: int = 500, margin: float = 0.0005) -> Dict:
    """Deploy the zoo if its trailing-best-3 ensemble beats the current champion on recent OOS
    log-loss. The deployed model = mean of the 3 members with the best recent OOS log-loss,
    refitted on all data, calibrated on the best3 OOS predictions."""
    import datetime as dt

    import joblib

    from .features import feature_columns
    from .model import REGISTRY, PlattCalibrator, evaluate, load_champion

    oos = zoo_result["oos"]
    members = [m for m in oos if m not in ("mean_all", "stack", "best3")]
    y = oos["best3"]["y_up"].astype(int)
    recent = oos["best3"].index[-recent_days:]
    scores = {m: _ll(oos[m].loc[recent, "p_raw"].values, y.loc[recent].values) for m in members}
    top3 = sorted(scores, key=scores.get)[:3]
    b3 = oos["best3"]
    b3_recent = _ll(b3.loc[recent, "p"].values, y.loc[recent].values)
    _, champ_meta = load_champion()
    champ_recent = None
    if champ_meta:
        f = REGISTRY / champ_meta["version"] / "oos_predictions.csv"
        if f.exists():
            co = pd.read_csv(f, index_col=0, parse_dates=True)
            common = co.index.intersection(recent)
            if len(common) > 100:
                champ_recent = _ll(co.loc[common, "p"].values, co.loc[common, "y_up"].values)
                b3_recent = _ll(b3.loc[common, "p"].values, y.loc[common].values)
    decision = {"top3": top3, "member_recent_logloss": {k: round(v, 5) for k, v in scores.items()},
                "best3_recent_logloss": round(b3_recent, 5),
                "champion_recent_logloss": None if champ_recent is None else round(champ_recent, 5)}
    if champ_recent is not None and b3_recent >= champ_recent - margin:
        decision["promoted"] = False
        return decision
    cols = feature_columns(F)
    set_context(F[cols])
    full = F.dropna(subset=["y_ret"])
    full = full[full["y_ret"].abs() >= cfg.deadzone_bps / 1e4]
    model = ZooEnsemble(top3, "mean").fit(full[cols], full["y_up"].astype(int))
    cal = PlattCalibrator().fit(b3["p_raw"].values, y.values)
    met, rec_met = evaluate(b3), evaluate(b3.loc[recent])
    has_edge = bool(met["skill_vs_baserate"] > 0 and rec_met["skill_vs_baserate"] > 0 and met["skill_p_value"] < 0.05)
    version = dt.datetime.now().strftime("%Y%m%d-%H%M%S") + "-zoo"
    vdir = REGISTRY / version
    vdir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "calibrator": cal, "target": cfg.target}, vdir / "model.joblib")
    b3.to_csv(vdir / "oos_predictions.csv")
    meta = {"version": version, "spec_name": "zoo_best3", "members": top3, "target": cfg.target,
            "has_edge": has_edge, "feature_groups": getattr(cfg, "feature_groups", "base"), "trained_on": [str(full.index.min().date()), str(full.index.max().date())],
            "n_train": int(len(full)), "features": model.cols, "explain": model.explain(),
            "metrics": met, "recent_metrics": rec_met, "zoo_decision": decision, "promoted": True}
    (vdir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    (REGISTRY / "champion.json").write_text(json.dumps({"version": version, "spec_name": "zoo_best3",
                                                        "has_edge": has_edge}, indent=2))
    decision.update({"promoted": True, "version": version, "has_edge": has_edge})
    return decision
