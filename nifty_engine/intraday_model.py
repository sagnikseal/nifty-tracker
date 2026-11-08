"""Intraday movement model: at every 5-minute bar close, predict NIFTY's direction
  * over the next 30 minutes          (`y30`,  drives the "hold another 30 minutes" rule), and
  * from now to the 15:15 exit         (`yclose`, drives entries).

Timing rules (identical in backtest and live):
  * a 5-minute bar that starts at s closes at s+5; the decision at T uses only bars that have
    closed by T, the previous sessions, and information known before the open (daily model
    probability, overnight global moves, pre-open news);
  * labels: close of the bar ending at T+30 (or 15:15) versus the close at T.

Evaluation is out-of-sample walk-forward by whole trading days (monthly refits, expanding
window, calibration fitted only on earlier out-of-sample predictions). Rows within one day are
strongly dependent, so every significance test aggregates to one number per day first
(day-clustered), which keeps the p-values honest.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import math
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .calendar import IST, days_to_expiry, is_expiry_day
from .config import DATA_DIR, EngineConfig

log = logging.getLogger(__name__)
REGISTRY = DATA_DIR / "models" / "intraday"
REPORTS = DATA_DIR / "reports"
OPEN_T = dt.time(9, 15)
EXIT_BAR = dt.time(15, 10)          # the bar 15:10-15:15; its close is the 15:15 exit price
LAST_DECISION = dt.time(14, 45)     # T + 30 minutes must end by 15:15
HORIZONS = {"30m": "y30", "close": "yclose"}
DEFAULT_MEMBERS = ("cat", "xgb")
# Removed after testing: L2 logistic regression and LightGBM. Walk-forward on 942 days (2023-2026) and the
# pre-registered ablation, they had no log-loss skill over the base rate (30-min accuracy ~50.8%, holdout ~49.9%).

FEATURE_LABELS = {
    "t_min": "Minutes since open", "dow": "Day of week", "is_expiry": "Expiry day", "dte": "Days to expiry",
    "gap": "Opening gap", "r_open_T": "Move since open", "r_pc_T": "Move vs previous close",
    "r_5m": "Last 5-min move", "r_15m": "Last 15-min move", "r_30m": "Last 30-min move", "r_60m": "Last 60-min move",
    "rv_sofar": "Realised volatility today", "rv_1h": "Realised volatility, last hour", "rv_rel": "Today's volatility vs normal",
    "range_pos": "Position in today's range", "range_pct": "Today's range size",
    "dist_twap": "Distance from today's average price", "dist_ema12": "Distance from 1-hour average",
    "orb_ready": "Opening range complete", "orb_break": "Opening-range breakout", "orb_dist": "Distance from opening range",
    "gap_fill": "Share of gap filled", "frac_above_open": "Time spent above open", "up_bars": "Share of rising bars",
    "streak": "Run of same-direction bars", "rsi14": "RSI (5-min)", "r_first30": "First 30-min move",
    "pdh_dist": "Distance to previous high", "pdl_dist": "Distance to previous low", "pivot_dist": "Distance to pivot",
    "prev_last60": "Previous day's last-hour move", "prev_oc": "Previous day open-to-close",
    "vix_level": "India VIX", "vix_chg_T": "VIX change vs yesterday", "vix_chg_open_T": "VIX change since open",
    "vix_r_30m": "VIX change, last 30 min", "p_daily": "Daily model probability",
    "nw_dir_30m": "News tone, last 30 min", "nw_shock_30m": "News shock, last 30 min", "nw_n_30m": "Headlines, last 30 min",
    "nw_dir_2h": "News tone, last 2 hours", "nw_dir_pre": "Pre-open news tone", "nw_impact_30m": "Learned news impact",
}


# ----------------------------------------------------------------------------- data
def load_bars(name: str = "nifty_5m") -> pd.DataFrame:
    p = DATA_DIR / "intraday" / f"{name}.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p, index_col=0)
    df.index = pd.to_datetime(df.index, utc=True).tz_convert(IST)
    df = df[(df.index.time >= OPEN_T) & (df.index.time <= dt.time(15, 29))]
    return df[~df.index.duplicated(keep="last")].sort_index()


def to_5m(df1: pd.DataFrame) -> pd.DataFrame:
    """Resample 1-minute bars (start-stamped) to 5-minute bars aligned to 09:15."""
    if df1.empty:
        return df1
    out = []
    for _, g in df1.groupby(df1.index.date):
        r = g.resample("5min", origin="start_day", offset="15min", label="left", closed="left").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
        out.append(r)
    return pd.concat(out) if out else pd.DataFrame()


# ----------------------------------------------------------------------------- features
def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def build_frame(n5: pd.DataFrame, v5: Optional[pd.DataFrame] = None, daily_close: Optional[pd.Series] = None,
                p_daily: Optional[pd.Series] = None, news_fn=None, last_only: bool = False,
                min_bars: int = 70) -> pd.DataFrame:
    """One row per (session, decision time T). Index: decision timestamp (IST).

    n5: 5-minute NIFTY bars (start-stamped) for several sessions (live: the last few + today).
    v5: 5-minute India VIX bars. daily_close: official daily closes (previous close for the gap).
    p_daily: daily-model P(close > open) keyed by session date. news_fn(T) -> dict of news features.
    last_only: only the latest decision of the last session (live)."""
    if n5.empty:
        return pd.DataFrame()
    n5 = n5.sort_index()
    rsi = _rsi(n5["close"])
    ema12 = n5["close"].ewm(span=12, adjust=False).mean()
    vix_by_day = {d: g for d, g in v5.groupby(v5.index.date)} if v5 is not None and len(v5) else {}
    days = sorted(set(n5.index.date))
    groups = {d: g for d, g in n5.groupby(n5.index.date)}
    rows: List[Dict] = []
    prev = None                                   # previous full session's summary
    prev_vix = None
    for i, d in enumerate(days):
        g = groups[d]
        live_day = last_only and i == len(days) - 1
        complete = len(g) >= min_bars
        if not complete and not live_day:
            if len(g) >= min_bars // 2:          # still update context from short sessions
                prev = _summary(g, prev)
            continue
        vg = vix_by_day.get(d)
        if prev is not None and (not last_only or live_day):
            pc = prev["close"]
            if daily_close is not None:
                ts_prev = pd.Timestamp(prev["date"])
                if ts_prev in daily_close.index and not pd.isna(daily_close.loc[ts_prev]):
                    pc = float(daily_close.loc[ts_prev])
            rows += _day_rows(d, g, pc, prev, vg, prev_vix, rsi, ema12, p_daily, news_fn, last_only)
        prev = _summary(g, prev)
        if vg is not None and len(vg):
            prev_vix = float(vg["close"].iloc[-1])
    if not rows:
        return pd.DataFrame()
    X = pd.DataFrame(rows).set_index("ts").sort_index()
    return X.replace([np.inf, -np.inf], np.nan)


def _summary(g: pd.DataFrame, prev: Optional[Dict]) -> Dict:
    c = g["close"]
    last60 = c[c.index.time >= dt.time(14, 25)]
    return {"date": g.index[0].date(), "open": float(g["open"].iloc[0]), "high": float(g["high"].max()),
            "low": float(g["low"].min()), "close": float(c.iloc[-1]),
            "last60": float(np.log(c.iloc[-1] / last60.iloc[0])) if len(last60) > 1 else 0.0,
            "rv": float(np.log(c).diff().std() * math.sqrt(75)) if len(c) > 5 else np.nan,
            "rv_hist": ((prev or {}).get("rv_hist", []) + [float(np.log(c).diff().std() * math.sqrt(75))])[-22:]}


def _day_rows(d, g, pc, prev, vg, prev_vix, rsi, ema12, p_daily, news_fn, last_only) -> List[Dict]:
    o = float(g["open"].iloc[0])
    c = g["close"].values.astype(float)
    h = g["high"].values.astype(float)
    lo_ = g["low"].values.astype(float)
    times = [t.time() for t in g.index]
    lr = np.diff(np.log(c), prepend=np.log(o))
    n = len(g)
    exit_idx = next((k for k, t in enumerate(times) if t == EXIT_BAR), None)
    H, L, C = prev["high"], prev["low"], prev["close"]
    pivot = (H + L + C) / 3
    atr = max(H - L, 1.0)
    rv_hist = [x for x in prev.get("rv_hist", []) if x == x]
    rv_norm = float(np.median(rv_hist)) if rv_hist else np.nan
    pdaily = float(p_daily.get(pd.Timestamp(d), np.nan)) if p_daily is not None else np.nan
    vix_close = vg["close"].reindex(g.index).ffill().values.astype(float) if vg is not None and len(vg) else None
    vix_open = float(vg["open"].iloc[0]) if vg is not None and len(vg) else np.nan
    orb_h = float(np.max(h[:6])) if n >= 6 else np.nan
    orb_l = float(np.min(lo_[:6])) if n >= 6 else np.nan
    p945 = c[5] if n >= 6 else np.nan
    out = []
    ks = range(n) if not last_only else [n - 1]
    for k in ks:
        T = g.index[k] + pd.Timedelta(minutes=5)
        if T.time() > LAST_DECISION and not last_only:
            break
        cT = c[k]
        hi, lo = float(np.max(h[:k + 1])), float(np.min(lo_[:k + 1]))
        rr = lr[1:k + 1]
        f = {"ts": T, "date": pd.Timestamp(d), "t_min": (T.hour * 60 + T.minute) - (9 * 60 + 15),
             "dow": T.weekday(), "is_expiry": int(is_expiry_day(d)), "dte": days_to_expiry(d),
             "gap": math.log(o / pc), "r_open_T": math.log(cT / o), "r_pc_T": math.log(cT / pc),
             "r_5m": lr[k] if k >= 1 else math.log(cT / o),
             "r_15m": math.log(cT / c[k - 3]) if k >= 3 else math.log(cT / o),
             "r_30m": math.log(cT / c[k - 6]) if k >= 6 else math.log(cT / o),
             "r_60m": math.log(cT / c[k - 12]) if k >= 12 else math.log(cT / o),
             "rv_sofar": float(np.std(rr) * math.sqrt(75)) if len(rr) > 2 else np.nan,
             "rv_1h": float(np.std(rr[-12:]) * math.sqrt(75)) if len(rr) > 4 else np.nan,
             "range_pos": (cT - lo) / (hi - lo) if hi > lo else 0.5, "range_pct": (hi - lo) / o,
             "dist_twap": math.log(cT / float(np.mean(c[:k + 1]))),
             "dist_ema12": math.log(cT / float(ema12.iloc[ema12.index.get_loc(g.index[k])])),
             "orb_ready": int(k >= 5),
             "orb_break": (1.0 if cT > orb_h else (-1.0 if cT < orb_l else 0.0)) if k >= 5 else 0.0,
             "orb_dist": ((cT - (orb_h + orb_l) / 2) / max(orb_h - orb_l, 1.0)) if k >= 5 else 0.0,
             "gap_fill": (cT - o) / (pc - o) if abs(pc - o) > 1 else 0.0,
             "frac_above_open": float(np.mean(c[:k + 1] > o)),
             "up_bars": float(np.mean(c[:k + 1] > g["open"].values[:k + 1])),
             "rsi14": float(rsi.iloc[rsi.index.get_loc(g.index[k])]),
             "r_first30": math.log(p945 / pc) if k >= 5 else np.nan,
             "pdh_dist": (cT - H) / atr, "pdl_dist": (cT - L) / atr, "pivot_dist": (cT - pivot) / atr,
             "prev_last60": prev.get("last60", 0.0), "prev_oc": math.log(prev["close"] / prev["open"]),
             "p_daily": pdaily}
        sgn = np.sign(lr[1:k + 1])
        run = 0
        for s in sgn[::-1]:
            if s == 0 or (run and np.sign(run) != s):
                break
            run += int(s)
        f["streak"] = float(run)
        f["rv_rel"] = f["rv_sofar"] / rv_norm if rv_norm and rv_norm == rv_norm and f["rv_sofar"] == f["rv_sofar"] else np.nan
        if vix_close is not None and not np.isnan(vix_close[k]):
            f["vix_level"] = vix_close[k]
            f["vix_chg_T"] = math.log(vix_close[k] / prev_vix) if prev_vix else np.nan
            f["vix_chg_open_T"] = math.log(vix_close[k] / vix_open) if vix_open == vix_open and vix_open > 0 else np.nan
            f["vix_r_30m"] = math.log(vix_close[k] / vix_close[k - 6]) if k >= 6 and vix_close[k - 6] > 0 else np.nan
        if news_fn is not None:
            try:
                f.update(news_fn(T.to_pydatetime()))
            except Exception as e:  # noqa: BLE001
                log.debug("news features failed: %s", e)
        # labels
        if k + 6 < n and (exit_idx is None or k + 6 <= exit_idx):
            f["y30_ret"] = math.log(c[k + 6] / cT)
            f["y30"] = float(c[k + 6] > cT)
        if exit_idx is not None and exit_idx > k:
            f["yclose_ret"] = math.log(c[exit_idx] / cT)
            f["yclose"] = float(c[exit_idx] > cT)
        f["px_T"] = cT
        out.append(f)
    return out


def feature_columns(X: pd.DataFrame) -> List[str]:
    return [c for c in X.columns if c not in ("date",) and not c.startswith(("y30", "yclose", "px_"))]


# ----------------------------------------------------------------------------- extra feature groups
INTRADAY_GROUPS = ("tech", "tech_mtf", "candles", "structure", "banknifty", "breadth", "futures", "chain",
             "global_intraday", "daily_ctx", "events")
STOCKS = ["hdfcbank", "reliance", "icicibank", "infy", "bhartiartl", "lt", "tcs", "itc", "sbin", "axisbank"]


def load_series() -> Dict[str, object]:
    """Every stored intraday series the feature groups can use (missing ones are simply absent)."""
    from .data.global_markets import load_hourly
    s: Dict[str, object] = {"nifty": load_bars("nifty_5m"), "vix": load_bars("indiavix_5m"),
                            "banknifty": load_bars("banknifty_5m"), "niftyfut": load_bars("niftyfut_5m")}
    s["stocks"] = {k: v for k in STOCKS if not (v := load_bars(f"{k}_5m")).empty}
    hourly = {}
    for name in ("es", "nq", "usdinr", "crude", "gold_f"):
        try:
            hourly[name] = load_hourly(name)
        except Exception:  # noqa: BLE001
            pass
    s["hourly"] = {k: v for k, v in hourly.items() if v is not None and len(v)}
    return s


def add_groups(X: pd.DataFrame, groups, series: Dict[str, object], daily_feats: Optional[pd.DataFrame] = None,
               chain: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Join feature groups onto a base frame indexed by decision time T. Identical in backtest and live."""
    from .analytics import cross_asset as ca
    groups = set(groups)
    dec = X.index
    parts = []
    n5 = series["nifty"]
    tech_groups = [g for g in ("tech", "tech_mtf", "candles", "structure") if g in groups]
    if tech_groups:
        from .analytics.patterns import tech_features
        T = tech_features(n5[["open", "high", "low", "close"]], tech_groups)
        T = T.reindex(dec - pd.Timedelta(minutes=5))
        T.index = dec
        parts.append(T)
    if "banknifty" in groups and len(series.get("banknifty", [])):
        parts.append(ca.banknifty_features(dec, n5, series["banknifty"]))
    if "breadth" in groups and series.get("stocks"):
        parts.append(ca.breadth_features(dec, n5, series["stocks"]))
    if "futures" in groups and len(series.get("niftyfut", [])):
        parts.append(ca.futures_features(dec, n5, series["niftyfut"]))
    if "global_intraday" in groups and series.get("hourly"):
        parts.append(ca.global_intraday_features(dec, series["hourly"]))
    if "daily_ctx" in groups and daily_feats is not None:
        parts.append(ca.daily_context(dec, daily_feats))
    if "events" in groups:
        try:
            parts.append(ca.events_features(dec))
        except Exception as e:  # noqa: BLE001
            log.info("events group skipped: %s", e)
    if "chain" in groups and chain is not None and len(chain):
        parts.append(chain.reindex(dec))
    if not parts:
        return X
    A = pd.concat(parts, axis=1)
    A = A.loc[:, ~A.columns.duplicated()]
    return X.join(A.drop(columns=[c for c in A.columns if c in X.columns])).replace([np.inf, -np.inf], np.nan)


GROUP_PREFIX = {"tech": ("t5_",), "tech_mtf": ("m15_", "m60_", "mtf_"), "candles": ("cdl_", "ha_"),
                "structure": ("sw_", "don", "round"), "banknifty": ("bn_",), "breadth": ("br_",), "futures": ("fut_",),
                "chain": ("oc_",), "global_intraday": ("gl_",), "daily_ctx": ("dc_",), "events": ("evt_",),
                "news": ("nw_",), "tsfm": ("tsfm_",)}
# Pre-registered on 2026-11-08 before any run: nested feature sets, compared on identical out-of-sample rows.
ABLATION_SETS = {
    "A0_base": [],
    "A1_+technicals": ["tech", "tech_mtf", "candles", "structure"],
    "A2_+cross_asset": ["tech", "tech_mtf", "candles", "structure", "banknifty", "breadth", "daily_ctx", "events"],
    "A3_+futures_chain_global": ["tech", "tech_mtf", "candles", "structure", "banknifty", "breadth", "daily_ctx", "events",
                                 "futures", "chain", "global_intraday"],
}


def columns_for(X: pd.DataFrame, groups) -> List[str]:
    """Base columns + the columns of the given feature groups."""
    every = tuple(p for ps in GROUP_PREFIX.values() for p in ps)
    base = [c for c in feature_columns(X) if not c.startswith(every)]
    extra = [c for c in feature_columns(X) if any(c.startswith(p) for g in groups for p in GROUP_PREFIX.get(g, ()))]
    return base + extra


def ablation(X: pd.DataFrame, members=DEFAULT_MEMBERS, refit_days: int = 63, sets=None) -> Dict:
    """Walk-forward every feature set; compare consecutive sets on identical rows (day-clustered)."""
    sets = sets or ABLATION_SETS
    out: Dict = {"sets": {}, "comparisons": {}, "members": list(members), "refit_days": refit_days}
    oos: Dict = {}
    for h, label in HORIZONS.items():
        for name, groups in sets.items():
            cols = columns_for(X, groups)
            o = walk_forward(X, label, members=members, refit_days=refit_days, cols=cols, max_missing=0.9)
            oos[(h, name)] = o
            r = evaluate(o)
            r["n_features"] = len(cols)
            r["holdout"] = evaluate(o[o["date"] >= pd.Timestamp(HOLDOUT_START)])
            out["sets"][f"{h}|{name}"] = r
            log.info("ablation %s %s: acc=%.4f skill=%.5f p=%.3g", h, name, r["accuracy"], r["logloss_skill"], r["p_value"])
        names = list(sets)
        for a, b in zip(names[:-1], names[1:]):
            A, B = oos[(h, a)], oos[(h, b)]
            common = A.index.intersection(B.index)
            if "futures" in sets[b] and "futures" not in sets[a]:
                rec = X.loc[common].filter(like="fut_basis").notna().any(axis=1)
                common = common[rec.reindex(common).fillna(False).values]
            A, B = A.loc[common], B.loc[common]
            d = _ll(A["p"].values, A["y"].values) - _ll(B["p"].values, B["y"].values)
            t = day_clustered_test(d, A["date"].values)
            out["comparisons"][f"{h}|{a} -> {b}"] = {"rows": int(len(common)), "days": t["days"],
                                                    "delta_logloss": float(d.mean()), "t": t["t"], "p": t["p"],
                                                    "acc_a": float(np.mean((A["p"] >= 0.5) == A["y"])),
                                                    "acc_b": float(np.mean((B["p"] >= 0.5) == B["y"]))}
    write_ablation_report(out, REPORTS / "INTRADAY_ABLATION_REPORT.md")
    return out


def write_ablation_report(out: Dict, path) -> None:
    L = ["# Which data and features help the movement model? (pre-registered ablation)", "",
         f"Walk-forward, members {out['members']} chosen monthly by trailing out-of-sample log-loss, refit every "
         f"{out['refit_days']} trading days. Feature sets were fixed before the first run. 'Skill' = log-loss improvement "
         "over the base rate (positive = better than always predicting the historical up-rate). Comparisons use identical "
         "rows; the futures/chain/global comparison uses only the days where that data exists (Sept 2024 onwards).", "",
         "| Horizon | Feature set | Features | Days | Accuracy | AUC | Skill | p (day-clustered) | Holdout acc | Holdout skill |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for k, r in out["sets"].items():
        h, name = k.split("|")
        ho = r.get("holdout") or {}
        L.append(f"| {h} | {name} | {r['n_features']} | {r['days']} | {r['accuracy']:.2%} | {r['auc']:.3f} | "
                 f"{r['logloss_skill']:+.5f} | {r['p_value']:.3g} | {ho.get('accuracy', float('nan')):.2%} | "
                 f"{ho.get('logloss_skill', float('nan')):+.5f} |")
    L += ["", "| Comparison | Rows | Days | Accuracy before → after | Log-loss gain | t | p |", "|---|---|---|---|---|---|---|"]
    for k, c in out["comparisons"].items():
        L.append(f"| {k} | {c['rows']:,} | {c['days']} | {c['acc_a']:.2%} → {c['acc_b']:.2%} | {c['delta_logloss']:+.5f} | "
                 f"{c['t']:.2f} | {c['p']:.3g} |")
    L += ["", "A group earns its place only with a positive log-loss gain at p < 0.05 on identical rows. With several "
          "comparisons, treat p between 0.01 and 0.05 as weak evidence."]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L), encoding="utf-8")


def build_training_frame_groups(cfg: EngineConfig, groups=INTRADAY_GROUPS, with_news: bool = True) -> pd.DataFrame:
    X = build_training_frame(cfg, with_news=with_news)
    series = load_series()
    daily = None
    if "daily_ctx" in groups:
        from .data import store
        from .features import build_daily_features
        daily = build_daily_features(store.load_all(), "oc", groups=["base"])
    chain = None
    if "chain" in groups:
        from .analytics import chain_intraday
        chain = chain_intraday.load()
        if chain.empty or chain.index.max() < X.index.max() - pd.Timedelta(days=14):
            chain = chain_intraday.build(series["nifty"]["close"], X.index)
    return add_groups(X, groups, series, daily, chain)


# ----------------------------------------------------------------------------- model
class IntradayEnsemble:
    """Gradient-boosted tree members (CatBoost, XGBoost), probabilities averaged. Sequence networks and
    foundation-model features live in intraday_zoo / intraday_stack."""

    def __init__(self, members=DEFAULT_MEMBERS, max_missing: float = 0.3, seed: int = 7):
        self.members = tuple(members)
        self.max_missing = max_missing
        self.seed = seed
        self.cols: List[str] = []
        self.models: Dict[str, object] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "IntradayEnsemble":
        miss = X.isna().mean()
        self.cols = [c for c in X.columns if miss[c] <= self.max_missing and X[c].nunique() > 1]
        Xc = X[self.cols]
        self.models = {}
        for m in self.members:
            if m == "cat":
                from catboost import CatBoostClassifier
                est = CatBoostClassifier(iterations=300, depth=4, learning_rate=0.05, l2_leaf_reg=8,
                                         random_seed=self.seed, verbose=False, allow_writing_files=False)
                est.fit(Xc, y)
            elif m == "xgb":
                from xgboost import XGBClassifier
                est = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8,
                                    colsample_bytree=0.8, min_child_weight=50, reg_lambda=5.0,
                                    random_state=self.seed, verbosity=0)
                est.fit(Xc, y)
            else:
                raise ValueError(f"unknown member {m!r} (available: cat, xgb)")
            self.models[m] = est
        return self

    def member_proba(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        Xc = X.reindex(columns=self.cols)
        return {m: est.predict_proba(Xc)[:, 1] for m, est in self.models.items()}

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        ps = list(self.member_proba(X).values())
        return np.clip(np.mean(ps, axis=0), 1e-4, 1 - 1e-4)

    def contributions(self, x: pd.DataFrame, top: int = 8) -> List[Dict]:
        """Signed contribution of each feature to the log-odds for one row (exact tree SHAP values from the
        XGBoost member, else CatBoost)."""
        xc = x.reindex(columns=self.cols)
        contrib = None
        if "xgb" in self.models:
            import xgboost as xgb
            contrib = self.models["xgb"].get_booster().predict(xgb.DMatrix(xc), pred_contribs=True)[0][:-1]
        elif "cat" in self.models:
            from catboost import Pool
            contrib = self.models["cat"].get_feature_importance(type="ShapValues", data=Pool(xc))[0][:-1]
        if contrib is None:
            return []
        order = np.argsort(-np.abs(contrib))[:top]
        return [{"feature": self.cols[i], "label": FEATURE_LABELS.get(self.cols[i], self.cols[i]),
                 "value": float(xc.iloc[0, i]) if not pd.isna(xc.iloc[0, i]) else None,
                 "contribution": float(contrib[i])} for i in order]


class Platt:
    def __init__(self):
        self.a, self.b, self.fitted = 1.0, 0.0, False

    def fit(self, p, y) -> "Platt":
        p, y = np.asarray(p, float), np.asarray(y, int)
        if len(p) < 2000 or len(np.unique(y)) < 2:
            return self
        from sklearn.linear_model import LogisticRegression
        z = np.log(p / (1 - p)).reshape(-1, 1)
        lr = LogisticRegression(C=1.0).fit(z, y)
        self.a, self.b, self.fitted = float(lr.coef_[0][0]), float(lr.intercept_[0]), True
        return self

    def __call__(self, p):
        p = np.asarray(p, float)
        if not self.fitted:
            return p
        return 1 / (1 + np.exp(-(self.a * np.log(p / (1 - p)) + self.b)))


def walk_forward(X: pd.DataFrame, label: str, members=DEFAULT_MEMBERS, warmup_days: int = 250,
                 refit_days: int = 21, deadzone: float = 0.0001, max_train_days: Optional[int] = None,
                 cols: Optional[List[str]] = None, combine: str = "vote", trail_rows: int = 8000,
                 max_missing: float = 0.3) -> pd.DataFrame:
    """Expanding-window walk-forward by trading day. Returns one OOS row per decision.

    combine="vote" (default): every member is calibrated separately on its own earlier OOS predictions;
    each month only the members whose log-loss over the last `trail_rows` OOS predictions beats the base
    rate get a vote, and the prediction is the average of their probabilities (a soft majority vote).
    No qualified member -> the base rate (no trade). Uses only the past, so it is itself out-of-sample.
    combine="mean": plain average of every member, then calibration."""
    cols = cols or feature_columns(X)
    data = X.dropna(subset=[label]).copy()
    days = np.array(sorted(data["date"].unique()))
    out, past_p, past_y = [], [], []
    past_m: Dict[str, List[float]] = {m: [] for m in members}
    i = warmup_days
    ret_col = label + "_ret"
    while i < len(days):
        test_days = days[i:i + refit_days]
        tr_days = days[max(0, i - max_train_days):i] if max_train_days else days[:i]
        tr = data[data["date"].isin(tr_days)]
        tr = tr[tr[ret_col].abs() >= deadzone]
        te = data[data["date"].isin(test_days)]
        model = IntradayEnsemble(members, max_missing=max_missing).fit(tr[cols], tr[label].astype(int))
        mp = model.member_proba(te[cols])
        base = float(tr[label].mean())
        p_raw = np.clip(np.mean(list(mp.values()), axis=0), 1e-4, 1 - 1e-4)
        chosen = "mean"
        if combine == "vote" and len(past_y) >= 2000:
            yy = np.array(past_y[-trail_rows:])
            base_ll = float(_ll(np.full(len(yy), np.mean(past_y)), yy).mean())
            voters = []
            for m in members:
                pm = np.array(past_m[m])
                older = len(pm) > trail_rows + 2000      # calibrate on older rows, judge on the trailing window
                cal = Platt().fit(pm[:-trail_rows] if older else pm, past_y[:len(pm) - trail_rows] if older else past_y)
                if float(_ll(cal(pm[-trail_rows:]), yy).mean()) < base_ll:
                    voters.append(m)
            chosen = "+".join(voters) if voters else "base"
            if voters:
                p_cal = np.mean([Platt().fit(past_m[m], past_y)(np.clip(mp[m], 1e-4, 1 - 1e-4)) for m in voters], axis=0)
            else:
                p_cal = np.full(len(te), base)
        else:
            p_cal = Platt().fit(past_p, past_y)(p_raw)
        o = pd.DataFrame({"date": te["date"].values, "p_raw": p_raw, "p": p_cal, "y": te[label].values,
                          "ret": te[ret_col].values, "base": base, "t_min": te["t_min"].values,
                          "r_30m": te["r_30m"].values, "p_daily": te["p_daily"].values if "p_daily" in te else np.nan,
                          "chosen": chosen}, index=te.index)
        for m, v in mp.items():
            o[f"p_{m}"] = v
            past_m[m] += np.clip(v, 1e-4, 1 - 1e-4).tolist()
        out.append(o)
        past_p += p_raw.tolist()
        past_y += te[label].astype(int).tolist()
        i += refit_days
        log.info("%s walk-forward: through %s (%d/%d days) using %s", label, str(test_days[-1])[:10],
                 min(i, len(days)), len(days), chosen)
    return pd.concat(out) if out else pd.DataFrame()


# ----------------------------------------------------------------------------- evaluation
def _ll(p, y):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def _ncdf(x):
    return 0.5 * math.erfc(-x / math.sqrt(2))


def day_clustered_test(per_row: np.ndarray, days: np.ndarray) -> Dict[str, float]:
    """One-sided t-test on the per-day means of `per_row` (positive = good)."""
    s = pd.Series(per_row).groupby(days).mean()
    n = len(s)
    if n < 10 or s.std(ddof=1) == 0:
        return {"days": n, "t": float("nan"), "p": float("nan")}
    t = s.mean() / (s.std(ddof=1) / math.sqrt(n))
    return {"days": n, "t": float(t), "p": float(1 - _ncdf(t))}


def evaluate(oos: pd.DataFrame, thresholds=(0.55, 0.6, 0.65, 0.7, 0.75, 0.8)) -> Dict:
    if oos.empty:
        return {"n": 0}
    y = oos["y"].values.astype(int)
    p = oos["p"].values
    days = oos["date"].values
    pred = (p >= 0.5).astype(int)
    from sklearn.metrics import roc_auc_score
    ll_m, ll_b = _ll(p, y), _ll(oos["base"].values, y)
    res = {"n": int(len(oos)), "days": int(len(set(days))),
           "period": [str(pd.Timestamp(min(days)).date()), str(pd.Timestamp(max(days)).date())],
           "base_rate": float(y.mean()), "accuracy": float(np.mean(pred == y)),
           "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else None,
           "logloss": float(ll_m.mean()), "logloss_base": float(ll_b.mean()),
           "brier": float(np.mean((p - y) ** 2))}
    res["logloss_skill"] = res["logloss_base"] - res["logloss"]
    sk = day_clustered_test(ll_b - ll_m, days)
    res["skill_t"], res["p_value"] = sk["t"], sk["p"]
    acc = day_clustered_test((pred == y).astype(float) - 0.5, days)
    res["accuracy_vs_coin_p"] = acc["p"]
    maj = int(y.mean() >= 0.5)
    res["accuracy_vs_majority_p"] = day_clustered_test((pred == y).astype(float) - (y == maj).astype(float), days)["p"]
    res["baselines"] = {"always_up": float(np.mean(y == 1)), "always_majority": float(np.mean(y == maj)),
                        "momentum_30m": float(np.mean((oos["r_30m"].values > 0).astype(int) == y)),
                        "mean_reversion_30m": float(np.mean((oos["r_30m"].values < 0).astype(int) == y))}
    if "p_daily" in oos and oos["p_daily"].notna().any():
        m = oos["p_daily"].notna().values
        res["baselines"]["daily_model_sign"] = float(np.mean((oos["p_daily"].values[m] >= 0.5).astype(int) == y[m]))
    rows = []
    for t in thresholds:
        m = (p >= t) | (p <= 1 - t)
        if m.sum() == 0:
            rows.append({"threshold": t, "coverage": 0.0, "n": 0, "hit_rate": None, "avg_ret_bps": None, "days": 0})
            continue
        d = np.where(p[m] >= 0.5, 1, -1)
        rows.append({"threshold": t, "coverage": float(m.mean()), "n": int(m.sum()),
                     "hit_rate": float(np.mean((p[m] >= 0.5).astype(int) == y[m])),
                     "avg_ret_bps": float(np.mean(d * oos["ret"].values[m]) * 1e4), "days": int(len(set(days[m])))})
    res["by_threshold"] = rows
    res["share_above_80"] = float(np.mean((p >= 0.8) | (p <= 0.2)))
    res["p_quantiles"] = {q: float(np.quantile(p, q)) for q in (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)}
    q = pd.qcut(p, 10, duplicates="drop")
    cal = pd.DataFrame({"p": p, "y": y}).groupby(q, observed=True).mean()
    res["calibration"] = [{"pred": float(a), "actual": float(b)} for a, b in zip(cal["p"], cal["y"])]
    yrs = pd.DatetimeIndex(days).year
    res["by_year"] = {int(k): {"accuracy": float(g["hit"].mean()), "n": int(len(g))}
                      for k, g in pd.DataFrame({"hit": pred == y, "yr": yrs}).groupby("yr")}
    tb = pd.cut(oos["t_min"], [-1, 30, 90, 180, 270, 400], labels=["09:20-09:45", "09:45-10:45", "10:45-12:15",
                                                                     "12:15-13:45", "13:45-15:15"])
    res["by_time"] = {str(k): {"accuracy": float(g["hit"].mean()), "n": int(len(g))}
                      for k, g in pd.DataFrame({"hit": pred == y, "tb": tb}).groupby("tb", observed=True)}
    res["members"] = {c[2:]: float(np.mean((oos[c].values >= 0.5).astype(int) == y))
                      for c in oos.columns if c.startswith("p_") and c not in ("p_raw", "p_daily")}
    return res


# ----------------------------------------------------------------------------- pipeline
def daily_prior() -> Optional[pd.Series]:
    """Out-of-sample daily-model probabilities from the champion's walk-forward (no look-ahead)."""
    from .model import REGISTRY as DREG
    champ = DREG / "champion.json"
    for p in ([DREG / json.loads(champ.read_text())["version"] / "oos_predictions.csv"] if champ.exists() else []) + \
             [REPORTS / "oos_predictions.csv"]:
        if p.exists():
            s = pd.read_csv(p, index_col=0, parse_dates=True)["p"]
            s.index = pd.DatetimeIndex(s.index).normalize()
            return s
    return None


def build_training_frame(cfg: EngineConfig, with_news: bool = True) -> pd.DataFrame:
    from .data import store
    n5 = load_bars("nifty_5m")
    if n5.empty:
        n1 = load_bars("nifty_1m")
        n5 = to_5m(n1)
    if n5.empty:
        raise RuntimeError("No intraday bars. Run: python -m nifty_engine fetch-intraday")
    v5 = load_bars("indiavix_5m")
    nf = store.load("nifty")
    news_fn = None
    if with_news:
        try:
            from .semantic.news_impact import NewsFeatureFn
            news_fn = NewsFeatureFn.from_store(cfg)
        except Exception as e:  # noqa: BLE001
            log.info("no news features: %s", e)
    return build_frame(n5, v5, nf["close"] if len(nf) else None, daily_prior(), news_fn)


HOLDOUT_START = "2026-05-01"   # final 6 months: reported separately as an untouched test period


class Chosen:
    """The deployed predictor: the members that had a vote in the last walk-forward month ("a+b"), each
    calibrated on its own OOS history, averaged. "base" = no member beat the base rate (no trade)."""

    def __init__(self, ensemble: IntradayEnsemble, member: str, calibrators: Dict[str, "Platt"], base: float):
        self.ensemble, self.member, self.calibrators, self.base = ensemble, member, calibrators, base

    @property
    def voters(self) -> List[str]:
        return [] if self.member in ("base", "mean") else self.member.split("+")

    def predict(self, x: pd.DataFrame) -> float:
        if self.member == "base":
            return self.base
        mp = self.ensemble.member_proba(x)
        if self.member == "mean":
            return float(self.calibrators["mean"](np.mean(list(mp.values()), axis=0))[0])
        return float(np.mean([self.calibrators[m](np.clip(mp[m], 1e-4, 1 - 1e-4))[0] for m in self.voters]))

    def agreement(self, x: pd.DataFrame) -> Optional[float]:
        """Share of voting members on the same side as the vote (1.0 = unanimous)."""
        if not self.voters:
            return None
        mp = self.ensemble.member_proba(x)
        sides = [self.calibrators[m](np.clip(mp[m], 1e-4, 1 - 1e-4))[0] >= 0.5 for m in self.voters]
        up = self.predict(x) >= 0.5
        return float(np.mean([s == up for s in sides]))


def train(cfg: EngineConfig, members=DEFAULT_MEMBERS, quick: bool = False, combine: str = "vote",
          groups=None, X: Optional[pd.DataFrame] = None) -> Dict:
    """Walk-forward both horizons, evaluate (full period, development, holdout), fit final models
    on all data, save + report. `groups`: extra feature groups to add (see INTRADAY_GROUPS); the deployed model
    records them so the live runtime computes exactly the same inputs."""
    groups = list(groups or [])
    if X is None:
        X = build_training_frame_groups(cfg, groups) if groups else build_training_frame(cfg)
    cols = columns_for(X, groups) if groups else feature_columns(X)
    max_missing = 0.9 if groups else 0.3
    res, oos_all, finals = {}, {}, {}
    hs = pd.Timestamp(HOLDOUT_START)
    for h, label in HORIZONS.items():
        oos = walk_forward(X, label, members=members, refit_days=63 if quick else 21, cols=cols, combine=combine,
                           max_missing=max_missing)
        oos_all[h] = oos
        res[h] = evaluate(oos)
        res[h]["development"] = evaluate(oos[oos["date"] < hs])
        res[h]["holdout"] = evaluate(oos[oos["date"] >= hs])
        res[h]["chosen_history"] = oos.groupby(oos["date"].dt.strftime("%Y"))["chosen"].agg(
            lambda s: s.value_counts().to_dict()).to_dict()
        data = X.dropna(subset=[label])
        data = data[data[label + "_ret"].abs() >= 0.0001]
        ens = IntradayEnsemble(members, max_missing=max_missing).fit(data[cols], data[label].astype(int))
        member = str(oos["chosen"].iloc[-1])
        cals = {m: Platt().fit(oos[f"p_{m}"].values, oos["y"].values) for m in members}
        cals["mean"] = Platt().fit(oos["p_raw"].values, oos["y"].values)
        finals[h] = {"model": ens, "predictor": Chosen(ens, member, cals, float(data[label].mean())), "member": member}
        log.info("intraday %s: acc=%.4f auc=%.4f skill=%.5f p=%.3g | holdout acc=%s | deployed member=%s", h,
                 res[h]["accuracy"], res[h]["auc"] or 0, res[h]["logloss_skill"], res[h]["p_value"],
                 res[h]["holdout"].get("accuracy"), member)
    version = "intraday-" + dt.datetime.now().strftime("%Y%m%d-%H%M")
    vdir = REGISTRY / version
    vdir.mkdir(parents=True, exist_ok=True)
    import joblib
    joblib.dump({"models": finals, "cols": cols}, vdir / "model.joblib")
    for h, oos in oos_all.items():
        oos.to_csv(vdir / f"oos_{h}.csv")
    meta = {"version": version, "members": list(members), "combine": combine, "features": cols, "groups": groups,
            "metrics": res,
            "deployed_member": {h: f["member"] for h, f in finals.items()}, "holdout_start": HOLDOUT_START,
            "has_edge": {h: bool(r.get("p_value") == r.get("p_value") and r["p_value"] < 0.05 and r["logloss_skill"] > 0
                                 and r["holdout"].get("logloss_skill", -1) > 0)
                         for h, r in res.items()},
            "trained_on": [str(X["date"].min().date()), str(X["date"].max().date())], "rows": int(len(X))}
    (vdir / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    (REGISTRY / "champion.json").write_text(json.dumps({"version": version}))
    write_report(meta, REPORTS / "INTRADAY_REPORT.md")
    return meta


def load_champion():
    f = REGISTRY / "champion.json"
    if not f.exists():
        return None, None
    v = json.loads(f.read_text())["version"]
    import joblib
    # Safe: model.joblib is written by `train()` on this machine into the local store, never downloaded.
    return joblib.load(REGISTRY / v / "model.joblib"), json.loads((REGISTRY / v / "meta.json").read_text())


def load_oos(h: str = "30m") -> pd.DataFrame:
    f = REGISTRY / "champion.json"
    if not f.exists():
        return pd.DataFrame()
    v = json.loads(f.read_text())["version"]
    p = REGISTRY / v / f"oos_{h}.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p, index_col=0, parse_dates=["date"])
    df.index = pd.to_datetime(df.index, utc=True).tz_convert(IST)
    return df


def predict_rows(bundle, X: pd.DataFrame) -> Dict[str, Dict]:
    """Calibrated predictions for the last row of X for each horizon + top drivers."""
    out = {}
    x = X.iloc[[-1]]
    for h, b in bundle["models"].items():
        pr = b["predictor"]
        out[h] = {"p": pr.predict(x), "member": b.get("member"), "drivers": b["model"].contributions(x),
                  "voters": getattr(pr, "voters", None),
                  "agreement": pr.agreement(x) if hasattr(pr, "agreement") else None}
    return out


# ----------------------------------------------------------------------------- report
def _pct(x, nd=1):
    return "—" if x is None or x != x else f"{x * 100:.{nd}f}%"


def write_report(meta: Dict, path) -> None:
    L = [f"# Intraday movement model: walk-forward report",
         f"_Version `{meta['version']}`, trained on {meta['trained_on'][0]} to {meta['trained_on'][1]} "
         f"({meta['rows']:,} decision points, one every 5 minutes). Members: {', '.join(meta['members'])}._", "",
         "Every number below is out-of-sample: each month is predicted by a model trained only on earlier days, "
         "and probabilities are calibrated only on earlier predictions. Significance tests aggregate to one value "
         "per day, because decisions within a day are not independent.", ""]
    for h, r in meta["metrics"].items():
        name = "next 30 minutes" if h == "30m" else "now to 15:15"
        if not r.get("n"):
            continue
        edge = meta["has_edge"].get(h)
        L += [f"## Horizon: {name}", "",
              f"* Period {r['period'][0]} to {r['period'][1]}: {r['days']} days, {r['n']:,} predictions. "
              f"Up rate {_pct(r['base_rate'])}.",
              f"* Accuracy **{_pct(r['accuracy'], 2)}** | AUC {r['auc']:.3f} | log-loss {r['logloss']:.5f} vs base rate "
              f"{r['logloss_base']:.5f} (skill {r['logloss_skill']:+.5f}, day-clustered t = {r['skill_t']:.2f}, "
              f"p = {r['p_value']:.3g}) → **{'significant edge' if edge else 'no proven edge'}**",
              f"* Accuracy vs coin flip p = {r['accuracy_vs_coin_p']:.3g}; vs always-majority p = {r['accuracy_vs_majority_p']:.3g}",
              f"* Share of predictions at or beyond 80% / 20%: **{_pct(r['share_above_80'], 2)}** "
              f"(how often a hold-if-above-80% rule can fire)",
              f"* Deployed predictor: `{meta.get('deployed_member', {}).get(h, '?')}` (members with a vote: each month, "
              f"only members whose trailing out-of-sample log-loss beats the base rate vote, and their probabilities "
              f"are averaged; 'base' = nobody qualified, so no trade. History by year: {r.get('chosen_history', {})})"]
        for part, title in (("development", f"Development (before {meta.get('holdout_start')})"),
                            ("holdout", f"Holdout ({meta.get('holdout_start')} onwards, untouched while building the model)")):
            q = r.get(part) or {}
            if q.get("n"):
                L.append(f"* {title}: {q['days']} days, accuracy {_pct(q['accuracy'], 2)}, AUC {q['auc']:.3f}, "
                         f"log-loss skill {q['logloss_skill']:+.5f} (p = {q['p_value']:.3g})")
        L += ["",
              "| Baseline | Accuracy |", "|---|---|"]
        L += [f"| {k} | {_pct(v, 2)} |" for k, v in r["baselines"].items()]
        L += ["", "| Member | Accuracy |", "|---|---|"] + [f"| {k} | {_pct(v, 2)} |" for k, v in r["members"].items()]
        L += ["", "| Confidence ≥ | Coverage | Predictions | Days | Hit rate | Avg signed move (bps) |", "|---|---|---|---|---|---|"]
        for t in r["by_threshold"]:
            hr = _pct(t["hit_rate"], 2) if t["hit_rate"] is not None else "—"
            bps = "—" if t["avg_ret_bps"] is None else "%.2f" % t["avg_ret_bps"]
            L.append(f"| {t['threshold']:.2f} | {_pct(t['coverage'], 2)} | {t['n']:,} | {t['days']} | {hr} | {bps} |")
        L += ["", "| Predicted P(up) | Actual up rate |", "|---|---|"] + \
             [f"| {_pct(c['pred'])} | {_pct(c['actual'])} |" for c in r["calibration"]]
        L += ["", "| Year | Accuracy | n |", "|---|---|---|"] + \
             [f"| {k} | {_pct(v['accuracy'], 2)} | {v['n']:,} |" for k, v in r["by_year"].items()]
        L += ["", "| Time of day | Accuracy | n |", "|---|---|---|"] + \
             [f"| {k} | {_pct(v['accuracy'], 2)} | {v['n']:,} |" for k, v in r["by_time"].items()]
        L.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L), encoding="utf-8")
