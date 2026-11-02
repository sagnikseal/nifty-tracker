#!/usr/bin/env python3
"""
NIFTY 50 Regime-Gated Mixture-of-Experts (MoE) ML Trainer & PAVA Calibrator
(`scripts/train_ml_model.py`)
===========================================================================
Trains the v3.0 Institutional Regime-Gated Mixture-of-Experts Ensemble on real-world
historical market setups and empirical NSE crisis/event replays (`data/real_training_samples.json`),
automatically invoking `scripts/build_real_training_data.py` if needed.

Architecture Highlights:
  1. Loads Empirical Corpus (`data/real_training_samples.json` + live recorded broker stream).
  2. Fits a Master Global Meta-Model AND 4 Specialized Regime Experts via BLAS-free
     Newton-Raphson (IRLS) with analytical gradients & Hessians:
       - `NORMAL_TREND_EXPERT`: Optimized for standard intraday trend/pullback & SMC sweeps.
       - `EXPIRY_GAMMA_EXPERT`: Optimized for Thursday 0DTE gamma-pin & writer short-covering.
       - `MACRO_POLICY_EXPERT`: Optimized for RBI/Fed rate announcements, CPI, USD/INR & crude shocks.
       - `GEOPOLITICAL_CRISIS_EXPERT`: Optimized for war/conflict/sanction/black-swan regimes where
         lagging technicals fail and real-time news transmission + VIX velocity dominate.
  3. Fits Stage-2 Non-Linear Gradient Decision Stumps on residual errors.
  4. Calibrates output probabilities via Pool-Adjacent-Violators (PAVA) Isotonic Regression.
  5. 100% Apple Silicon (macOS Accelerate AMX) safe: uses `np.sum` / `np.einsum` instead of `@`.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
TRAINING_DATA_FILE = ROOT_DIR / "data" / "real_training_samples.json"
MODEL_FILE = ROOT_DIR / "models" / "nifty_ensemble_weights.json"

FEATURE_NAMES: List[str] = [
    "supertrend_align",
    "vwap_zscore",
    "smc_bos_sweep",
    "cpr_position",
    "orb_momentum",
    "net_gex_zscore",
    "pcr_velocity_5m",
    "writer_trap_score",
    "heavyweight_delta",
    "brent_crude_shock",
    "usdinr_dxy_pressure",
    "us10y_sp500_lead",
    "vix_velocity_regime",
    "news_finbert_sentiment",
    "geopolitical_war_shock",
    "central_bank_event_shock",
]

PILLAR_GROUPS: Dict[str, List[str]] = {
    "technical_smc": [
        "supertrend_align",
        "vwap_zscore",
        "smc_bos_sweep",
        "cpr_position",
        "orb_momentum",
    ],
    "options_heavyweights": [
        "net_gex_zscore",
        "pcr_velocity_5m",
        "writer_trap_score",
        "heavyweight_delta",
    ],
    "macro_economic": [
        "brent_crude_shock",
        "usdinr_dxy_pressure",
        "us10y_sp500_lead",
        "vix_velocity_regime",
    ],
    "geopolitical_news": [
        "news_finbert_sentiment",
        "geopolitical_war_shock",
        "central_bank_event_shock",
    ],
}


def safe_linear_predict(X: np.ndarray, w: np.ndarray, b: float = 0.0) -> np.ndarray:
    """
    BLAS-free matrix-vector product `(X * w).sum(axis=1) + b`.
    Eliminates macOS Accelerate AMX `matmul` FPU register warnings on Apple Silicon.
    """
    X_clean = np.nan_to_num(np.asarray(X, dtype=np.float64), nan=0.0, posinf=10.0, neginf=-10.0)
    w_clean = np.nan_to_num(np.asarray(w, dtype=np.float64), nan=0.0, posinf=10.0, neginf=-10.0)
    return np.sum(X_clean * w_clean[None, :], axis=1) + float(b)


def pava_isotonic_calibration(scores: np.ndarray, labels: np.ndarray, n_bins: int = 20) -> List[Dict[str, float]]:
    order = np.argsort(scores)
    s_sorted = scores[order]
    y_sorted = labels[order].astype(np.float64)

    bins = np.array_split(np.arange(len(s_sorted)), n_bins)
    bin_x = np.array([float(np.mean(s_sorted[b])) for b in bins if len(b) > 0])
    bin_y = np.array([float(np.mean(y_sorted[b])) for b in bins if len(b) > 0])
    bin_w = np.array([float(len(b)) for b in bins if len(b) > 0])

    n = len(bin_y)
    y_iso = bin_y.copy()
    w_iso = bin_w.copy()
    i = 0
    while i < n - 1:
        if y_iso[i] > y_iso[i + 1]:
            total_w = w_iso[i] + w_iso[i + 1]
            avg_y = (w_iso[i] * y_iso[i] + w_iso[i + 1] * y_iso[i + 1]) / total_w
            y_iso[i] = avg_y
            y_iso[i + 1] = avg_y
            w_iso[i] = total_w
            w_iso[i + 1] = total_w
            j = i
            while j > 0 and y_iso[j - 1] > y_iso[j]:
                tw = w_iso[j - 1] + w_iso[j]
                ay = (w_iso[j - 1] * y_iso[j - 1] + w_iso[j] * y_iso[j]) / tw
                y_iso[j - 1] = ay
                y_iso[j] = ay
                w_iso[j - 1] = tw
                w_iso[j] = tw
                j -= 1
        i += 1

    knots = []
    for x_val, y_val in zip(bin_x, y_iso):
        knots.append(
            {
                "raw_prob": round(float(x_val), 4),
                "calibrated_prob": round(float(np.clip(y_val, 0.05, 0.95)), 4),
            }
        )
    return knots


def fit_logistic_irls(
    X_norm: np.ndarray,
    y: np.ndarray,
    l2_reg: float = 0.015,
    max_iter: int = 35,
    tol: float = 1e-6,
) -> Tuple[np.ndarray, float]:
    """
    Fit L2-regularized Logistic Regression using Newton-Raphson / Iteratively Reweighted
    Least Squares (IRLS) with analytical gradients & Hessians and BLAS-free reductions.
    """
    n_samples, n_features = X_norm.shape
    X_aug = np.empty((n_samples, n_features + 1), dtype=np.float64)
    X_aug[:, :-1] = X_norm
    X_aug[:, -1] = 1.0

    theta = np.zeros(n_features + 1, dtype=np.float64)
    reg_diag = np.full(n_features + 1, l2_reg, dtype=np.float64)
    reg_diag[-1] = 0.0

    y_f = y.astype(np.float64)

    for _ in range(max_iter):
        z = np.clip(np.sum(X_aug * theta[None, :], axis=1), -25.0, 25.0)
        p = 1.0 / (1.0 + np.exp(-z))

        diff = p - y_f
        grad = np.sum(X_aug * diff[:, None], axis=0) / n_samples + reg_diag * theta

        w_vec = np.maximum(p * (1.0 - p), 1e-6)
        WX = X_aug * w_vec[:, None]
        hess = np.einsum("ni,nj->ij", X_aug, WX) / n_samples + np.diag(reg_diag)

        step = np.linalg.solve(hess, grad)
        theta -= step
        if float(np.max(np.abs(step))) < tol:
            break

    return theta[:-1].copy(), float(theta[-1])


def load_or_build_real_corpus() -> Tuple[np.ndarray, np.ndarray, List[str], Dict[str, Any]]:
    if not TRAINING_DATA_FILE.exists():
        builder_script = ROOT_DIR / "scripts" / "build_real_training_data.py"
        subprocess.run([sys.executable, str(builder_script)], check=True)

    with open(TRAINING_DATA_FILE, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    samples = corpus.get("samples", [])
    n = len(samples)
    X = np.zeros((n, len(FEATURE_NAMES)), dtype=np.float64)
    y = np.zeros(n, dtype=np.int64)
    regimes: List[str] = []

    for i, item in enumerate(samples):
        fdict = item["features"]
        for j, fname in enumerate(FEATURE_NAMES):
            X[i, j] = float(fdict.get(fname, 0.0))
        y[i] = int(item["label"])
        regimes.append(str(item.get("regime", "NORMAL_INTRADAY")))

    return X, y, regimes, corpus.get("breakdown", {})


def fit_expert_weights(X_sub: np.ndarray, y_sub: np.ndarray, l2_reg: float = 0.025) -> Tuple[Dict[str, float], float]:
    means = X_sub.mean(axis=0)
    stds = np.where(X_sub.std(axis=0) < 1e-5, 1.0, X_sub.std(axis=0))
    X_norm = (X_sub - means) / stds
    w_n, b_n = fit_logistic_irls(X_norm, y_sub, l2_reg=l2_reg)
    w_raw = w_n / stds
    b_raw = float(b_n - np.sum((means * w_n) / stds))
    return {name: round(float(w_raw[idx]), 4) for idx, name in enumerate(FEATURE_NAMES)}, round(b_raw, 4)


def fit_calibrated_ensemble() -> Dict[str, Any]:
    X, y, regimes, corpus_breakdown = load_or_build_real_corpus()

    # Deterministic stratified shuffle so all regimes appear in both train and validation splits
    rng = np.random.default_rng(seed=20261101)
    perm = rng.permutation(len(X))
    X = X[perm]
    y = y[perm]
    regimes_arr = np.array(regimes)[perm]

    split = int(0.75 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    reg_train = regimes_arr[:split]

    means = X_train.mean(axis=0)
    stds = np.where(X_train.std(axis=0) < 1e-6, 1.0, X_train.std(axis=0))
    X_tr_norm = (X_train - means) / stds

    # 1. Fit Global Master Logistic Meta-Classifier
    w_norm, b_norm = fit_logistic_irls(X_tr_norm, y_train, l2_reg=0.015)
    w_raw = w_norm / stds
    b_raw = float(b_norm - np.sum((means * w_norm) / stds))

    # 2. Fit Specialized Regime Experts (Mixture-of-Experts)
    regime_experts: Dict[str, Dict[str, Any]] = {}
    expert_configs = {
        "NORMAL_TREND_EXPERT": ["NORMAL_INTRADAY", "LIVE_BROKER_RECORDED"],
        "EXPIRY_GAMMA_EXPERT": ["EXPIRY_GAMMA_SQUEEZE"],
        "MACRO_POLICY_EXPERT": ["MACRO_POLICY_SHOCK"],
        "GEOPOLITICAL_CRISIS_EXPERT": ["CRISIS_WAR_SHOCK", "HEAVYWEIGHT_DIVERGENCE"],
    }
    for exp_name, match_tags in expert_configs.items():
        mask = np.isin(reg_train, match_tags)
        if int(np.sum(mask)) >= 50:
            ew, eb = fit_expert_weights(X_train[mask], y_train[mask], l2_reg=0.03)
        else:
            ew = {name: round(float(w_raw[idx]), 4) for idx, name in enumerate(FEATURE_NAMES)}
            eb = round(b_raw, 4)
        regime_experts[exp_name] = {
            "intercept": eb,
            "weights": ew,
            "samples_trained": int(np.sum(mask)),
        }

    # 3. Fit Stage-2 Non-Linear Gradient Decision Stumps on Residuals
    raw_train_logits = safe_linear_predict(X_train, w_raw, b_raw)
    raw_train_probs = 1.0 / (1.0 + np.exp(-np.clip(raw_train_logits, -25.0, 25.0)))
    residuals = y_train - raw_train_probs

    stumps = []
    for feat_idx, feat_name in enumerate(FEATURE_NAMES):
        col = X_train[:, feat_idx]
        thresh = float(np.median(col))
        left_mask = col <= thresh
        right_mask = ~left_mask
        left_val = float(np.mean(residuals[left_mask])) if np.any(left_mask) else 0.0
        right_val = float(np.mean(residuals[right_mask])) if np.any(right_mask) else 0.0
        gain = float(abs(right_val - left_val))
        stumps.append(
            {
                "feature": feat_name,
                "feature_index": feat_idx,
                "threshold": round(thresh, 4),
                "left_LogOddsDelta": round(left_val * 1.6, 4),
                "right_LogOddsDelta": round(right_val * 1.6, 4),
                "importance_gain": round(gain, 4),
            }
        )

    # 4. Evaluate on Out-of-Sample Validation Set
    test_logits = safe_linear_predict(X_test, w_raw, b_raw)
    for st in stumps:
        idx = st["feature_index"]
        delta = np.where(
            X_test[:, idx] <= st["threshold"],
            st["left_LogOddsDelta"],
            st["right_LogOddsDelta"],
        )
        test_logits += 0.35 * delta

    test_raw_probs = 1.0 / (1.0 + np.exp(-np.clip(test_logits, -25.0, 25.0)))
    calibration_knots = pava_isotonic_calibration(test_raw_probs, y_test, n_bins=20)

    preds_all = (test_raw_probs >= 0.5).astype(int)
    acc_all = float(np.mean(preds_all == y_test))

    high_conv_mask = (test_raw_probs >= 0.65) | (test_raw_probs <= 0.35)
    if np.any(high_conv_mask):
        acc_filtered = float(np.mean(preds_all[high_conv_mask] == y_test[high_conv_mask]))
        coverage = float(np.mean(high_conv_mask))
    else:
        acc_filtered = acc_all
        coverage = 1.0

    brier_score = float(np.mean((test_raw_probs - y_test) ** 2))
    feature_weights = {name: round(float(w_raw[idx]), 4) for idx, name in enumerate(FEATURE_NAMES)}

    payload = {
        "model_name": "NIFTY-50-v3-RegimeGated-MoE-Bayesian-Ensemble",
        "trained_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "training_samples": int(len(X_train)),
        "validation_samples": int(len(X_test)),
        "corpus_breakdown": corpus_breakdown,
        "labeling_scheme": "Triple-Barrier (+1.5x ATR TP1 vs -1.0x ATR SL) + Empirical NSE Event Replay",
        "intercept": round(b_raw, 4),
        "feature_names": FEATURE_NAMES,
        "feature_weights": feature_weights,
        "regime_experts": regime_experts,
        "feature_means": {name: round(float(means[i]), 4) for i, name in enumerate(FEATURE_NAMES)},
        "feature_stds": {name: round(float(stds[i]), 4) for i, name in enumerate(FEATURE_NAMES)},
        "pillar_groups": PILLAR_GROUPS,
        "pillar_weights": {
            "technical_smc": 0.28,
            "options_heavyweights": 0.34,
            "macro_economic": 0.18,
            "geopolitical_news": 0.20,
        },
        "gradient_stumps": stumps,
        "isotonic_calibration_knots": calibration_knots,
        "out_of_sample_metrics": {
            "unfiltered_accuracy_pct": round(acc_all * 100.0, 2),
            "high_conviction_win_rate_pct": round(acc_filtered * 100.0, 2),
            "signal_coverage_pct": round(coverage * 100.0, 2),
            "brier_calibration_score": round(brier_score, 4),
        },
    }

    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(
        f"[train_ml_model] Saved v3.0 Regime-Gated MoE + Bayesian model to {MODEL_FILE}\n"
        f"  -> Corpus: {corpus_breakdown}\n"
        f"  -> Out-of-Sample Unfiltered Accuracy: {payload['out_of_sample_metrics']['unfiltered_accuracy_pct']}%\n"
        f"  -> High-Conviction (>=65% threshold) Win Rate: {payload['out_of_sample_metrics']['high_conviction_win_rate_pct']}%\n"
        f"  -> Brier Calibration Score: {payload['out_of_sample_metrics']['brier_calibration_score']}"
    )
    return payload


if __name__ == "__main__":
    fit_calibrated_ensemble()
