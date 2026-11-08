"""How did NIFTY actually move after the LLM read a headline? (event study + learned reactions)

For every scored headline first seen during market hours (09:15-15:00 IST) the engine measures
NIFTY's move over the next 5/15/30/60 minutes from the first 1-minute bar after `first_seen`
(never the publisher's timestamp), together with the 15 minutes before it. It then learns a
small, heavily regularised model:

    expected 30-minute move (bps) = f(LLM direction x magnitude x relevance, surprise, confidence,
                                      transmission channel, time of day, prior drift)

The model is evaluated walk-forward (each month predicted by a fit on earlier months only), and
its coefficients are stored as plain JSON. The learned expected impact is used in three places:
  1. as a feature of the intraday movement model (`nw_impact_30m`),
  2. in the dashboard next to each headline,
  3. as a reaction rule (exit on a strong adverse headline), which is only switched on once the
     pre-registered test in the report passes.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import math
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..calendar import IST
from ..config import DATA_DIR, EngineConfig

log = logging.getLogger(__name__)
MODEL_FILE = DATA_DIR / "models" / "news_impact.json"
REPORT = DATA_DIR / "reports" / "NEWS_IMPACT_REPORT.md"
HORIZONS = (5, 15, 30, 60)
CHANNELS = ["rates", "inflation", "fx", "crude_energy", "geopolitics", "flows", "earnings",
            "policy_regulation", "global_equities", "commodities", "domestic_macro", "other"]
# Pre-registered pass rule for switching the reaction rule on (fixed before any data was seen):
PASS_RULE = {"min_events": 300, "min_days": 40, "oos_ic_min": 0.05, "oos_ic_p_max": 0.05}


# ----------------------------------------------------------------------------- data
def load_1m() -> pd.DataFrame:
    p = DATA_DIR / "intraday" / "nifty_1m.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p, index_col=0, usecols=[0, 1, 2, 3, 4])
    df.index = pd.to_datetime(df.index, utc=True).tz_convert(IST)
    return df[~df.index.duplicated(keep="last")].sort_index()


def _x(row: Dict) -> Dict[str, float]:
    """Design vector for one headline (also used live)."""
    d = float(row.get("nifty_direction") or 0.0)
    mag = float(row.get("magnitude") or 0.0)
    rel = float(row.get("india_relevance") or 0.0)
    signed = d * mag * rel
    x = {"signed": signed, "signed_surprise": signed * float(row.get("surprise") or 0.0),
         "signed_conf": signed * float(row.get("confidence") or 0.0)}
    ch = row.get("channel") or "other"
    for c in CHANNELS:
        x[f"ch_{c}"] = signed if ch == c else 0.0
    t = row.get("t_min")
    x["signed_early"] = signed if t is not None and t < 60 else 0.0
    return x


def build_events(cfg: EngineConfig, S: Optional[pd.DataFrame] = None, bars: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """One row per in-session headline with forward NIFTY moves (bps)."""
    if S is None:
        from .news_llm import scored_frame
        S = scored_frame(cfg)
    if S is None or S.empty:
        return pd.DataFrame()
    bars = load_1m() if bars is None else bars
    if bars.empty:
        return pd.DataFrame()
    close = bars["close"]
    idx = close.index
    S = S.copy()
    S["first_seen"] = pd.to_datetime(S["first_seen"])
    tt = S["first_seen"].dt.tz_convert(IST)
    m = (tt.dt.day_of_week < 5) & (tt.dt.time >= dt.time(9, 15)) & (tt.dt.time <= dt.time(15, 0))
    S = S[m]
    rows = []
    for r in S.to_dict("records"):
        t0 = r["first_seen"]
        pub = r.get("published")
        if pub and isinstance(pub, str):
            try:     # stale: the publisher's time is well before we first saw it -> timing unknown, skip
                if (t0 - pd.Timestamp(pub).tz_convert(IST)) > pd.Timedelta(minutes=30):
                    continue
            except Exception:  # noqa: BLE001
                pass
        k = idx.searchsorted(t0, side="right")          # first bar that STARTS after the headline
        if k >= len(idx) or idx[k].date() != t0.date() or (idx[k] - t0) > pd.Timedelta(minutes=3):
            continue
        p0 = float(close.iloc[k - 1]) if k > 0 and idx[k - 1].date() == t0.date() else float(bars["open"].iloc[k])
        kb = idx.searchsorted(t0 - pd.Timedelta(minutes=15))
        pre = math.log(p0 / float(close.iloc[kb])) * 1e4 if kb < k and idx[kb].date() == t0.date() else 0.0
        ev = {"id": r.get("id"), "title": r.get("title"), "source": r.get("source"), "first_seen": t0,
              "date": pd.Timestamp(t0.date()), "t_min": (t0.hour * 60 + t0.minute) - 555, "pre15_bps": pre,
              "scorer": r.get("scorer")}
        for c in ("india_relevance", "nifty_direction", "magnitude", "surprise", "confidence", "channel", "horizon"):
            ev[c] = r.get(c)
        for h in HORIZONS:
            j = idx.searchsorted(t0 + pd.Timedelta(minutes=h), side="right") - 1
            ev[f"fwd{h}_bps"] = (math.log(float(close.iloc[j]) / p0) * 1e4
                                 if j >= k and idx[j].date() == t0.date() else np.nan)
        rows.append(ev)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- model
class ImpactModel:
    """Ridge regression of the forward move on the headline design vector (JSON-serialisable)."""

    def __init__(self, alpha: float = 50.0, horizon: int = 30):
        self.alpha, self.horizon = alpha, horizon
        self.cols: List[str] = []
        self.coef: List[float] = []
        self.intercept, self.resid_sd, self.n = 0.0, 25.0, 0
        self.validated = False
        self.stats: Dict = {}

    def fit(self, ev: pd.DataFrame) -> "ImpactModel":
        y = ev[f"fwd{self.horizon}_bps"].astype(float)
        ok = y.notna()
        X = pd.DataFrame([_x(r) for r in ev[ok].to_dict("records")])
        y = y[ok].values
        self.cols = list(X.columns)
        self.n = int(len(y))
        if self.n < 20:
            self.coef = [0.0] * len(self.cols)
            return self
        from sklearn.linear_model import Ridge
        r = Ridge(alpha=self.alpha, fit_intercept=False).fit(X.values, y)   # no intercept: no news => no drift
        self.coef = [float(c) for c in r.coef_]
        self.resid_sd = float(np.std(y - r.predict(X.values)) or 25.0)
        return self

    def predict_bps(self, row: Dict) -> float:
        if not self.cols:
            return 0.0
        x = _x(row)
        return float(sum(c * x.get(k, 0.0) for k, c in zip(self.cols, self.coef)))

    def p_up(self, row: Dict) -> float:
        z = self.predict_bps(row) / max(self.resid_sd, 1e-6)
        return 0.5 * math.erfc(-z / math.sqrt(2))

    def to_json(self) -> Dict:
        return {"alpha": self.alpha, "horizon": self.horizon, "cols": self.cols, "coef": self.coef,
                "resid_sd": self.resid_sd, "n": self.n, "validated": self.validated, "stats": self.stats}

    @classmethod
    def from_json(cls, js: Dict) -> "ImpactModel":
        m = cls(js.get("alpha", 50.0), js.get("horizon", 30))
        m.cols, m.coef = js.get("cols", []), js.get("coef", [])
        m.resid_sd, m.n = js.get("resid_sd", 25.0), js.get("n", 0)
        m.validated, m.stats = bool(js.get("validated")), js.get("stats", {})
        return m


def load_model() -> Optional[ImpactModel]:
    if not MODEL_FILE.exists():
        return None
    try:
        return ImpactModel.from_json(json.loads(MODEL_FILE.read_text(encoding="utf-8")))
    except Exception:  # noqa: BLE001
        return None


def _spearman(a, b) -> float:
    s = pd.DataFrame({"a": a, "b": b}).dropna()
    return float(s["a"].rank().corr(s["b"].rank())) if len(s) > 5 else float("nan")


def _day_t(x: np.ndarray, days: np.ndarray) -> Dict[str, float]:
    s = pd.Series(x).groupby(days).mean().dropna()
    if len(s) < 5 or s.std(ddof=1) == 0:
        return {"t": float("nan"), "p": float("nan"), "days": int(len(s))}
    t = s.mean() / (s.std(ddof=1) / math.sqrt(len(s)))
    return {"t": float(t), "p": float(0.5 * math.erfc(t / math.sqrt(2))), "days": int(len(s))}


def study(cfg: EngineConfig, ev: Optional[pd.DataFrame] = None, horizon: int = 30) -> Dict:
    """Event study + walk-forward evaluation; fits and saves the impact model; writes the report."""
    ev = build_events(cfg) if ev is None else ev
    out: Dict = {"events": int(len(ev)), "horizon": horizon, "pass_rule": PASS_RULE}
    if ev.empty:
        out["status"] = "INSUFFICIENT_DATA"
        _write_report(out, ev)
        return out
    ev = ev.sort_values("first_seen").reset_index(drop=True)
    signed = (ev["nifty_direction"].astype(float) * ev["magnitude"].astype(float) * ev["india_relevance"].astype(float))
    out["days"] = int(ev["date"].nunique())
    out["by_scorer"] = ev["scorer"].value_counts().to_dict()
    raw = {}
    for h in HORIZONS:
        y = ev[f"fwd{h}_bps"].astype(float)
        strong = (ev["india_relevance"].astype(float) >= 0.5) & (ev["nifty_direction"].astype(float).abs() >= 0.3) & y.notna()
        hit = (np.sign(ev.loc[strong, "nifty_direction"].astype(float)) == np.sign(y[strong]))
        signed_ret = np.sign(ev.loc[strong, "nifty_direction"].astype(float)) * y[strong]
        raw[h] = {"ic": _spearman(signed, y), "n": int(y.notna().sum()), "strong_n": int(strong.sum()),
                  "strong_hit_rate": float(hit.mean()) if strong.sum() else None,
                  "strong_signed_bps": float(signed_ret.mean()) if strong.sum() else None,
                  "strong_t": _day_t(signed_ret.values, ev.loc[strong, "date"].values) if strong.sum() else None}
    out["raw_llm_signal"] = raw
    by_ch = {}
    yh = ev[f"fwd{horizon}_bps"].astype(float)
    for ch, g in ev.groupby("channel"):
        s = signed[g.index]
        by_ch[str(ch)] = {"n": int(len(g)), "ic": _spearman(s, yh[g.index]),
                          "avg_signed_bps": float((np.sign(s) * yh[g.index]).mean()) if len(g) else None}
    out["by_channel"] = by_ch
    # walk-forward (monthly, expanding) out-of-sample predictions
    months = ev["first_seen"].dt.tz_convert(IST).dt.strftime("%Y-%m")
    um = sorted(months.unique())
    preds = pd.Series(np.nan, index=ev.index)
    for i in range(1, len(um)):
        tr, te = ev[months < um[i]], ev[months == um[i]]
        if tr[f"fwd{horizon}_bps"].notna().sum() < 50:
            continue
        mdl = ImpactModel(horizon=horizon).fit(tr)
        preds[te.index] = [mdl.predict_bps(r) for r in te.to_dict("records")]
    ok = preds.notna() & yh.notna()
    oos = {"n": int(ok.sum()), "days": int(ev.loc[ok, "date"].nunique())}
    if ok.sum() >= 30:
        oos["ic"] = _spearman(preds[ok], yh[ok])
        signed_oos = np.sign(preds[ok]) * yh[ok]
        oos["signed_bps"] = float(signed_oos.mean())
        oos["hit_rate"] = float((np.sign(preds[ok]) == np.sign(yh[ok])).mean())
        tt = _day_t(signed_oos.values, ev.loc[ok, "date"].values)
        oos["t"], oos["p"] = tt["t"], tt["p"]
        # IC significance (day-clustered): rank product per day
        rk = (preds[ok].rank() - preds[ok].rank().mean()) * (yh[ok].rank() - yh[ok].rank().mean())
        ict = _day_t(rk.values, ev.loc[ok, "date"].values)
        oos["ic_p"] = ict["p"]
    out["oos"] = oos
    model = ImpactModel(horizon=horizon).fit(ev)
    passed = (out["events"] >= PASS_RULE["min_events"] and out["days"] >= PASS_RULE["min_days"]
              and oos.get("ic", -1) >= PASS_RULE["oos_ic_min"] and oos.get("ic_p", 1) <= PASS_RULE["oos_ic_p_max"])
    model.validated = bool(passed)
    model.stats = {"events": out["events"], "days": out["days"], "oos": oos}
    out["status"] = "PASS" if passed else ("INSUFFICIENT_DATA" if out["events"] < PASS_RULE["min_events"]
                                            or out["days"] < PASS_RULE["min_days"] else "FAIL")
    out["coefficients"] = dict(zip(model.cols, model.coef))
    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    MODEL_FILE.write_text(json.dumps(model.to_json(), indent=1), encoding="utf-8")
    _write_report(out, ev)
    return out


def _write_report(out: Dict, ev: pd.DataFrame) -> None:
    f = lambda v, nd=3: "—" if v is None or (isinstance(v, float) and v != v) else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))  # noqa: E731
    L = ["# News impact: how NIFTY moved after the LLM read each headline", "",
         f"Status: **{out.get('status')}**. Events: {out.get('events', 0)} in-session headlines over "
         f"{out.get('days', 0)} days. Pre-registered pass rule (fixed in code before testing): "
         f"≥ {PASS_RULE['min_events']} events on ≥ {PASS_RULE['min_days']} days, out-of-sample rank IC ≥ "
         f"{PASS_RULE['oos_ic_min']} with day-clustered p ≤ {PASS_RULE['oos_ic_p_max']}.", "",
         "Until it passes, learned impacts are shown in the dashboard as information only and the "
         "adverse-headline exit rule stays off.", ""]
    if out.get("raw_llm_signal"):
        L += ["## The LLM's raw reading vs what happened", "",
              "| Horizon | Events | Rank IC (signed score vs move) | Strong calls | Strong-call hit rate | Avg move in called direction (bps) | t (day-clustered) |",
              "|---|---|---|---|---|---|---|"]
        for h, r in out["raw_llm_signal"].items():
            st = r.get("strong_t") or {}
            L.append(f"| {h} min | {r['n']} | {f(r['ic'])} | {r['strong_n']} | {f(r['strong_hit_rate'])} | "
                     f"{f(r['strong_signed_bps'], 2)} | {f(st.get('t'), 2)} |")
        L += ["", "Strong call = India relevance ≥ 0.5 and |direction| ≥ 0.3.", ""]
    if out.get("oos"):
        o = out["oos"]
        L += ["## Learned reaction model (walk-forward, 30 minutes)", "",
              f"* Out-of-sample predictions: {o.get('n')} on {o.get('days')} days",
              f"* Rank IC {f(o.get('ic'))} (day-clustered p = {f(o.get('ic_p'))}); hit rate {f(o.get('hit_rate'))}; "
              f"avg move in predicted direction {f(o.get('signed_bps'), 2)} bps (t = {f(o.get('t'), 2)})", ""]
    if out.get("by_channel"):
        L += ["| Channel | Events | Rank IC | Avg move in called direction (bps) |", "|---|---|---|---|"]
        for ch, r in sorted(out["by_channel"].items(), key=lambda kv: -kv[1]["n"]):
            L.append(f"| {ch} | {r['n']} | {f(r['ic'])} | {f(r['avg_signed_bps'], 2)} |")
        L.append("")
    if out.get("coefficients"):
        L += ["## Coefficients (bps of 30-minute move per unit of signed score)", "", "| Term | Coefficient |", "|---|---|"]
        L += [f"| {k} | {v:+.2f} |" for k, v in out["coefficients"].items() if abs(v) > 1e-9]
        L.append("")
    if out.get("events", 0) == 0:
        L += ["No scored in-session headlines with matching 1-minute NIFTY bars yet. Run the engine during "
              "market hours (it captures and scores news every minute), or `python -m nifty_engine news-backfill` "
              "for ~3 months of GDELT history, then `python -m nifty_engine news-impact`."]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(L), encoding="utf-8")


# ----------------------------------------------------------------------------- features for the movement model
class NewsFeatureFn:
    """Callable T -> news features known at T (only headlines with first_seen <= T)."""

    def __init__(self, S: pd.DataFrame, model: Optional[ImpactModel] = None):
        S = S.copy()
        S["first_seen"] = pd.to_datetime(S["first_seen"])
        S = S.sort_values("first_seen").reset_index(drop=True)
        # all comparisons in naive UTC nanoseconds (both sides converted the same way)
        self.ts = S["first_seen"].dt.tz_convert("UTC").dt.tz_localize(None).to_numpy().astype("datetime64[ns]")
        rel = S["india_relevance"].astype(float).values
        self.rel = rel
        self.signed = (S["nifty_direction"].astype(float) * S["magnitude"].astype(float)).values
        self.wt = rel * (0.25 + S["confidence"].astype(float).values)
        self.shock = rel * S["magnitude"].astype(float).values * (0.5 + 0.5 * S["surprise"].astype(float).values)
        recs = S.to_dict("records")
        self.impact = np.array([model.predict_bps(r) for r in recs]) if model is not None else np.zeros(len(S))

    @classmethod
    def from_store(cls, cfg: EngineConfig) -> "NewsFeatureFn":
        from .news_llm import scored_frame
        S = scored_frame(cfg)
        if S is None or S.empty:
            raise RuntimeError("no scored headlines")
        return cls(S, load_model())

    @staticmethod
    def _utc(t) -> np.datetime64:
        return np.datetime64(pd.Timestamp(t).tz_convert("UTC").tz_localize(None), "ns")

    def _win(self, a: dt.datetime, b: dt.datetime):
        i, j = np.searchsorted(self.ts, self._utc(a), side="right"), np.searchsorted(self.ts, self._utc(b), side="right")
        return slice(i, j)

    def __call__(self, T: dt.datetime) -> Dict[str, float]:
        if len(self.ts) == 0 or self._utc(T) < self.ts[0]:
            return {}
        out = {}
        for name, mins in (("30m", 30), ("2h", 120)):
            s = self._win(T - dt.timedelta(minutes=mins), T)
            w = self.wt[s]
            out[f"nw_n_{name}"] = float(np.sum(self.rel[s] >= 0.5))
            out[f"nw_dir_{name}"] = float(np.sum(w * self.signed[s]) / w.sum()) if w.sum() > 0 else 0.0
            if name == "30m":
                out["nw_shock_30m"] = float(np.max(self.shock[s] * np.sign(self.signed[s]), initial=0.0)
                                            if s.stop > s.start else 0.0)
                out["nw_impact_30m"] = float(np.sum(self.impact[s]))
        day = pd.Timestamp(T).tz_convert(IST).date()
        pre_a = dt.datetime.combine(day - dt.timedelta(days=3 if day.weekday() == 0 else 1), dt.time(15, 30), tzinfo=IST)
        s = self._win(pre_a, dt.datetime.combine(day, dt.time(9, 10), tzinfo=IST))
        w = self.wt[s]
        out["nw_dir_pre"] = float(np.sum(w * self.signed[s]) / w.sum()) if w.sum() > 0 else 0.0
        return out
