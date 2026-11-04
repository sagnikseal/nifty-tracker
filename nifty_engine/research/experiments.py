"""Runs every pre-registered experiment in config/experiments.json and writes
store/reports/RESEARCH_REPORT.md (+ research.json). Nothing is skipped silently: an experiment
without enough data is reported as INSUFFICIENT_DATA with what to download or record.

    python -m nifty_engine research            # everything the local data allows
    python -m nifty_engine research --only P1,S4
    python -m nifty_engine research --quick    # logistic model only (fast; used by CI)
"""

from __future__ import annotations

import copy
import datetime as dt
import json
import logging
import math
import traceback
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..calendar import IST, next_weekly_expiry
from ..config import DATA_DIR, ROOT, EngineConfig
from ..data import store
from ..data.expired_options import OUT as OPT_DIR
from ..features import feature_columns
from ..model import CANDIDATES, ModelSpec, evaluate, walk_forward
from ..paper.charges import order_charges
from ..paper.sim import rates_from
from ..strategies.base import Context, TradePlan
from ..strategies.books import BOOKS, IronCondor, SMCIntraday
from . import stats
from .option_replay import simulate

log = logging.getLogger(__name__)
REPORTS = DATA_DIR / "reports"
REGISTRY_FILE = ROOT / "config" / "experiments.json"


def registry() -> Dict:
    return json.loads(REGISTRY_FILE.read_text(encoding="utf-8"))


class Lab:
    """Caches data, frames and walk-forward predictions shared by experiments."""

    def __init__(self, cfg: EngineConfig, quick: bool = False):
        self.cfg = cfg
        self.quick = quick
        self.raw = store.load_all()
        self.spec = CANDIDATES[1] if quick else CANDIDATES[0]
        self._frames: Dict = {}
        self._oos: Dict = {}
        self._n5 = None
        self._daily = None

    # ------------------------------------------------------------------ predictions
    def frame(self, groups, target: str = "oc") -> pd.DataFrame:
        from ..pipeline import build_frame
        key = (tuple(sorted(groups)), target)
        if key not in self._frames:
            self._frames[key] = build_frame(self.cfg, self.raw, list(groups), target=target)
        return self._frames[key]

    def oos(self, groups=("base",), target: str = "oc") -> pd.DataFrame:
        key = (tuple(sorted(groups)), target)
        if key not in self._oos:
            c = copy.deepcopy(self.cfg)
            c.target = target
            self._oos[key] = walk_forward(self.frame(groups, target), c, self.spec)
        return self._oos[key]

    # ------------------------------------------------------------------ intraday + options
    @property
    def n5(self) -> pd.DataFrame:
        if self._n5 is None:
            try:
                from ..intraday import load_5m
                self._n5 = load_5m("nifty_5m")
            except Exception:  # noqa: BLE001
                self._n5 = pd.DataFrame()
        return self._n5

    def real_expiries(self) -> List[dt.date]:
        if not OPT_DIR.exists():
            return []
        out = []
        for f in OPT_DIR.glob("20*.csv"):
            try:
                out.append(dt.date.fromisoformat(f.stem))
            except ValueError:
                continue
        return sorted(out)

    def expiries(self, start: dt.date, end: dt.date) -> List[dt.date]:
        real = self.real_expiries()
        gen, d = [], start
        while d <= end + dt.timedelta(days=70):
            e = next_weekly_expiry(d)
            gen.append(e)
            d = e + dt.timedelta(days=1)
        gen = [g for g in gen if all(abs((g - r).days) > 3 for r in real)]
        return sorted(set(real) | set(gen))

    @property
    def daily(self) -> pd.DataFrame:
        """Per-session context used by the books (all causal for that session's decisions)."""
        if self._daily is None:
            from ..analytics.regime import regime_features
            from ..data.events import event_features
            nf = self.raw["nifty"]
            vix = self.raw["indiavix"]["close"] if len(self.raw.get("indiavix", [])) else None
            D = regime_features(nf, vix, lag=1)
            D = D.join(event_features(nf.index))
            D["gap"] = np.log(nf["open"] / nf["close"].shift(1))
            D["px_vix_prev"] = vix.reindex(nf.index).ffill(limit=3).shift(1) if vix is not None else np.nan
            base = self.oos(("base",))
            D["p"] = base["p"].reindex(D.index)
            try:
                rng = self.oos(("base",), target="range")
                D["p_range"] = rng["p"].reindex(D.index)
            except Exception:  # noqa: BLE001
                D["p_range"] = np.nan
            try:
                from ..semantic.news_llm import daily_news_frame
                nw = daily_news_frame(self.cfg, nf.index)
                if len(nw):
                    D = D.join(nw)
            except Exception:  # noqa: BLE001
                pass
            self._daily = D
        return self._daily

    def contexts(self, at: dt.time):
        n5 = self.n5
        if n5.empty:
            return
        sessions = sorted(set(n5.index.date))
        exps = self.expiries(sessions[0], sessions[-1])
        D = self.daily
        ev_types = sorted({c[3:] for c in D.columns if c.startswith("ev_") and not c.endswith("_next")
                           and c not in ("ev_any", "ev_any_next")})
        briefs = {}
        try:
            from ..semantic.market_brief import load_briefs
            b = load_briefs()
            briefs = {d.date(): {"brief": r.to_dict()} for d, r in b.iterrows()}
        except Exception:  # noqa: BLE001
            pass
        by_month: Dict = {}
        for d in sessions:
            by_month.setdefault((d.year, d.month), []).append(d)
        bars_at = n5[n5.index.time == at]
        open_by_day = {ts.date(): float(r["open"]) for ts, r in bars_at.iterrows()}
        for i, d in enumerate(sessions):
            if d not in open_by_day or pd.Timestamp(d) not in D.index:
                continue
            r = D.loc[pd.Timestamp(d)]
            vix = r.get("px_vix_prev")
            if vix != vix or vix is None:
                continue
            month = by_month[(d.year, d.month)]
            k = month.index(d)
            yield Context(cfg=self.cfg, day=d, now=at, spot=open_by_day[d], vix=float(vix),
                          expiries=[e for e in exps if e >= d][:6],
                          p_up=None if r.get("p") != r.get("p") else float(r["p"]),
                          p_range=None if r.get("p_range") != r.get("p_range") else float(r["p_range"]),
                          regime={c: float(r[c]) for c in ("har_vol_fc", "vrp_har", "vix_pct_1y", "har_vol_pct", "er20")
                                  if c in r and r[c] == r[c]},
                          events_today=[t for t in ev_types if r.get(f"ev_{t}") == 1],
                          events_next=[t for t in ev_types if r.get(f"ev_{t}_next") == 1],
                          gap=None if r.get("gap") != r.get("gap") else float(r["gap"]),
                          sessions_left_in_month=len(month) - k if (d.year, d.month) != (sessions[-1].year, sessions[-1].month) else None,
                          session_of_month=k + 1, next_sessions=sessions[i + 1:i + 30], brief=briefs.get(d))

    def vix_daily(self) -> pd.Series:
        return self.raw["indiavix"]["close"]


# ============================================================================== experiments
def _status_feature(res: Dict, rule: Dict) -> str:
    if res.get("n", 0) < rule["min_days"]:
        return "INSUFFICIENT_DATA"
    return "PASS" if res.get("delta_logloss", 0) > 0 and res.get("p", 1) < rule["max_p"] else "FAIL"


def exp_benchmark_base(lab: Lab, e: Dict, reg: Dict) -> Dict:
    oos = lab.oos(("base",))
    m = evaluate(oos)
    keep = {k: m.get(k) for k in ("n_days", "period", "accuracy", "auc", "logloss", "logloss_baserate",
                                  "skill_p_value", "baselines")}
    keep["by_threshold"] = m.get("by_threshold")
    keep["accuracy_coverage"] = stats.accuracy_coverage(oos["p"].values, oos["y_up"].values)
    return {"status": "REPORTED", "metrics": keep}


def _group_cols(lab: Lab, group: str) -> List[str]:
    base = set(feature_columns(lab.frame(("base",))))
    return [c for c in feature_columns(lab.frame(("base", group))) if c not in base]


def exp_feature_group(lab: Lab, e: Dict, reg: Dict) -> Dict:
    g = e["group"]
    cols = _group_cols(lab, g)
    if not cols:
        return {"status": "INSUFFICIENT_DATA", "note": f"no '{g}' columns: run `update-data` (needs the v5 series)"}
    a, b = lab.oos(("base",)), lab.oos(("base", g))
    F = lab.frame(("base", g))
    avail = F[cols].notna().any(axis=1)
    j = a[["p", "y_up"]].join(b[["p"]].rename(columns={"p": "p_b"}), how="inner")
    j = j[avail.reindex(j.index).fillna(False).astype(bool)]
    res = stats.dm_logloss(j["p"].values, j["p_b"].values, j["y_up"].values.astype(int))
    rule = reg["rules"]["feature_group"]
    res["columns"] = cols
    return {"status": _status_feature(res, rule), "metrics": res}


def stack_test(base: pd.DataFrame, X: pd.DataFrame, min_train: int = 120, refit: int = 20) -> pd.DataFrame:
    """Walk-forward 2nd stage: logit(p_base) + group features -> p. Same days, no look-ahead."""
    j = base[["p", "y_up"]].join(X, how="inner")
    j = j[X.reindex(j.index).notna().any(axis=1)].sort_index()
    if len(j) <= min_train:
        return pd.DataFrame()
    j["z"] = np.log(j["p"] / (1 - j["p"]))
    feats = ["z"] + list(X.columns)
    out, model = [], None
    for i in range(min_train, len(j)):
        if model is None or (i - min_train) % refit == 0:
            tr = j.iloc[:i]
            model = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
                                  LogisticRegression(C=0.2, max_iter=1000)).fit(tr[feats], tr["y_up"].astype(int))
        out.append({"date": j.index[i], "p": float(j["p"].iloc[i]),
                    "p_b": float(model.predict_proba(j[feats].iloc[[i]])[0, 1]), "y_up": float(j["y_up"].iloc[i])})
    return pd.DataFrame(out).set_index("date")


def exp_short_group(lab: Lab, e: Dict, reg: Dict) -> Dict:
    g = e["group"]
    F = lab.frame(("base", g))
    cols = _group_cols(lab, g)
    hints = {"overnight_futures": "run `update-data` daily (stores hourly ES/NQ/crude/BTC bars; Yahoo keeps ~2 years)",
             "options_hist": "run `fetch-intraday`, `fetch-expired-options --mode full`, then `build-option-features`",
             "news": "run `news-backfill` once, then the scheduler collects news daily (`news-capture`, `news-score`)"}
    if not cols:
        return {"status": "INSUFFICIENT_DATA", "note": hints.get(g, "")}
    t = stack_test(lab.oos(("base",)), F[cols])
    if t.empty:
        n = int(F[cols].notna().any(axis=1).sum())
        return {"status": "INSUFFICIENT_DATA", "note": f"{n} days with data; {hints.get(g, '')}"}
    res = stats.dm_logloss(t["p"].values, t["p_b"].values, t["y_up"].values.astype(int))
    res["columns"] = cols
    rule = reg["rules"]["short_history_group"]
    st = "INSUFFICIENT_DATA" if res["n"] < rule["min_days"] else (
        "PASS" if res["delta_logloss"] > 0 and res["p"] < rule["max_p"] else "FAIL")
    return {"status": st, "metrics": res}


def _book_trades(lab: Lab, book, allow_modeled: bool = True) -> pd.DataFrame:
    plans: List[TradePlan] = []
    for at in book.times:
        for ctx in lab.contexts(at):
            p = book.propose(ctx)
            if p:
                plans.append(p)
    return simulate(plans, lab.cfg, lab.n5, lab.vix_daily(), allow_modeled=allow_modeled)


def _smc_plans(lab: Lab) -> List[TradePlan]:
    from ..analytics.smc import smc_features, smc_setups
    from ..analytics.technical import prev_day_levels
    n5 = lab.n5
    lv = prev_day_levels(n5)
    S = smc_features(n5, k=3, ext_high=lv["pdh"], ext_low=lv["pdl"])
    sess = n5.index.to_series().dt.date
    st = smc_setups(n5, S, session=sess, rr=SMCIntraday.rr, start_time=SMCIntraday.window[0],
                    end_time=SMCIntraday.window[1])
    exps = lab.expiries(n5.index[0].date(), n5.index[-1].date())
    plans = []
    for _, s in st.iterrows():
        d = s["ts_entry"].date()
        ex = [e for e in exps if (e - d).days >= 1]
        if not ex:
            continue
        a = round(s["entry"] / 50) * 50
        k = a - 50 if s["direction"] > 0 else a + 50
        from ..options import Leg
        plans.append(TradePlan("smc_intraday", d, s["ts_entry"].time(), int(s["direction"]), "itm_buy",
                               [Leg("CE" if s["direction"] > 0 else "PE", float(k), +1)], ex[0], stop_pct=60.0,
                               underlying_stop=float(s["stop"]), underlying_target=float(s["target"]),
                               meta={"risk_pts": float(s["risk_pts"])}))
    return plans, st


def _strategy_status(summ: Dict, dsr: Dict, rule: Dict) -> str:
    if summ.get("trades", 0) < rule["min_trades"]:
        return "INSUFFICIENT_DATA"
    pos = summ["net_per_trade"] > 0
    if pos and summ["t_stat"] >= rule["min_t"] and dsr.get("dsr", 0) >= rule["min_dsr"] and \
            summ.get("real_price_share", 0) >= rule["min_real_share"]:
        return "PASS"
    return "PROMISING" if pos and summ["t_stat"] >= 1.0 else "FAIL"


def exp_strategy(lab: Lab, e: Dict, reg: Dict) -> Dict:
    if lab.n5.empty:
        return {"status": "INSUFFICIENT_DATA", "note": "needs 5-minute NIFTY bars: `python -m nifty_engine fetch-intraday`"}
    name = e["book"]
    extra = {}
    if name == "smc_intraday":
        plans, setups = _smc_plans(lab)
        T = simulate(plans, lab.cfg, lab.n5, lab.vix_daily())
        extra["setups"] = int(len(setups))
    elif name == "iron_condor":
        variants = {}
        for k in (1.0, 1.25, 1.5):
            c = copy.deepcopy(lab.cfg)
            c.condor_k = k
            sub = Lab.__new__(Lab)
            sub.__dict__.update(lab.__dict__)
            sub.cfg = c
            variants[k] = _book_trades(sub, IronCondor())
        T = variants[1.25]
        months = {k: (v.assign(m=v["date"].str[:7]).groupby("m")["net_pnl"].sum() if len(v) else pd.Series(dtype=float))
                  for k, v in variants.items()}
        M = pd.DataFrame(months)
        extra["pbo"] = stats.pbo_cscv(M) if len(M) >= 10 else {"pbo": None, "note": "fewer than 10 months"}
        extra["variants"] = {str(k): stats.trade_summary(v, lab.cfg.book_capital) for k, v in variants.items()}
    else:
        T = _book_trades(lab, BOOKS[name], allow_modeled=(name != "iron_fly_0dte"))
    summ = stats.trade_summary(T, lab.cfg.book_capital)
    lab.__dict__.setdefault("_book_results", {})[name] = T
    return {"status": "PENDING_DSR", "metrics": summ, "extra": extra, "_trades": T}


def exp_tom_futures(lab: Lab, e: Dict, reg: Dict) -> Dict:
    nf = lab.raw["nifty"]
    if nf.empty:
        return {"status": "INSUFFICIENT_DATA"}
    rates = rates_from(lab.cfg)
    q = lab.cfg.lot_size
    rows = []
    months = nf.groupby([nf.index.year, nf.index.month])
    keys = list(months.groups.keys())
    for i in range(len(keys) - 1):
        cur, nxt = months.get_group(keys[i]), months.get_group(keys[i + 1])
        if len(cur) < 3 or len(nxt) < 3:
            continue
        en, ex = float(cur["close"].iloc[-2]), float(nxt["close"].iloc[2])
        en_px, ex_px = en + lab.cfg.futures_slippage_pts, ex - lab.cfg.futures_slippage_pts
        ch = order_charges("future", "BUY", en_px * q, rates)["total"] + order_charges("future", "SELL", ex_px * q, rates)["total"]
        gross = (ex_px - en_px) * q
        rows.append({"date": str(cur.index[-2].date()), "gross_pnl": gross, "charges": ch, "net_pnl": gross - ch,
                     "ret_pct": (ex / en - 1) * 100})
    T = pd.DataFrame(rows)
    s = stats.trade_summary(T, 500_000)
    s["mean_ret_pct"] = round(float(T["ret_pct"].mean()), 3)
    s["by_era_net"] = {era: round(float(T[(T["date"] >= a) & (T["date"] < b)]["net_pnl"].sum()), 0)
                       for era, (a, b) in {"2007-13": ("2007", "2014"), "2014-19": ("2014", "2020"), "2020-26": ("2020", "2027")}.items()}
    return {"status": "REPORTED", "metrics": s}


def exp_smc_index(lab: Lab, e: Dict, reg: Dict) -> Dict:
    if lab.n5.empty:
        return {"status": "INSUFFICIENT_DATA", "note": "needs `fetch-intraday`"}
    _, st = _smc_plans(lab)
    n5 = lab.n5
    R = []
    for _, s in st.iterrows():
        d = s["ts_entry"].date()
        b = n5[(n5.index >= s["ts_entry"]) & (n5.index.date == d) & (n5.index.time <= dt.time(15, 15))]
        if b.empty:
            continue
        r_mult, dirn = None, s["direction"]
        for _, x in b.iterrows():
            if (x["low"] <= s["stop"] if dirn > 0 else x["high"] >= s["stop"]):
                r_mult = -1.0          # stop first when both touch in one bar (conservative)
                break
            if (x["high"] >= s["target"] if dirn > 0 else x["low"] <= s["target"]):
                r_mult = SMCIntraday.rr
                break
        if r_mult is None:
            r_mult = (float(b["close"].iloc[-1]) - s["entry"]) * dirn / s["risk_pts"]
        R.append(r_mult)
    R = np.array(R)
    if len(R) == 0:
        return {"status": "INSUFFICIENT_DATA", "note": "no setups found"}
    return {"status": "REPORTED", "metrics": {"setups": int(len(R)), "mean_R": round(float(R.mean()), 3),
                                              "t_stat": round(stats.t_stat(R), 2), "win_rate": round(float((R > 0).mean()), 3),
                                              "R_95ci": [round(x, 3) for x in stats.bootstrap_mean_ci(R)]}}


def exp_meta(lab: Lab, e: Dict, reg: Dict) -> Dict:
    from ..meta.stacker import add_trade_context, evaluate_meta, walk_forward_meta
    res = getattr(lab, "_book_results", {})
    frames = [t for t in res.values() if t is not None and len(t)]
    if not frames:
        return {"status": "INSUFFICIENT_DATA", "note": "no strategy trades in this run"}
    T = pd.concat(frames, ignore_index=True)
    rule = reg["rules"]["meta"]
    if len(T) < rule["min_trades"]:
        return {"status": "INSUFFICIENT_DATA", "metrics": {"trades": int(len(T))}}
    T = add_trade_context(T, lab.daily)
    T = walk_forward_meta(T)
    m = evaluate_meta(T)
    ok = m.get("kept_share", 0) >= rule["min_kept"] and (m.get("net_per_trade_kept") or -1e9) > m.get("net_per_trade_all", 0)
    return {"status": "PASS" if ok else "FAIL", "metrics": m}


def exp_forward_brief(lab: Lab, e: Dict, reg: Dict) -> Dict:
    from ..semantic.market_brief import load_briefs
    b = load_briefs()
    rule = reg["rules"]["forward"]
    if b.empty:
        return {"status": "INSUFFICIENT_DATA", "note": "no briefs yet: the scheduler writes one every morning (needs Ollama)"}
    nf = lab.raw["nifty"]
    y = (nf["close"] > nf["open"]).astype(float)
    j = b[["p_up"]].join(y.rename("y_up"), how="inner").join(lab.oos(("base",))["p"], how="left")
    j = j.dropna(subset=["y_up"])
    out = {"days": int(len(j)), "llm_accuracy": round(float(((j["p_up"] >= 0.5) == (j["y_up"] == 1)).mean()), 3) if len(j) else None}
    jj = j.dropna(subset=["p"])
    if len(jj) >= 10:
        out["vs_model"] = stats.dm_logloss(jj["p"].values, jj["p_up"].values, jj["y_up"].values.astype(int))
    st = "INSUFFICIENT_DATA" if len(j) < rule["min_days"] else (
        "PASS" if out.get("vs_model", {}).get("p", 1) < 0.10 else "FAIL")
    return {"status": st, "metrics": out}


def exp_forward_live(lab: Lab, e: Dict, reg: Dict) -> Dict:
    from ..data.recorder import load_day, recorded_days
    days = recorded_days()
    rule = reg["rules"]["forward"]
    rows = []
    for d in days:
        m = load_day(d, "chain_metrics")
        if m.empty:
            continue
        at = m[m["ts"].dt.time <= dt.time(9, 20)]
        if at.empty:
            at = m.iloc[[0]]
        r = at.iloc[-1]
        rows.append({"date": pd.Timestamp(d), **{f"lv_{k}": r[k] for k in r.index if k.startswith("near_") and
                                                 isinstance(r[k], (int, float, np.floating)) and k not in ("near_spot", "near_atm")}})
    if len(rows) < rule["min_days"]:
        return {"status": "INSUFFICIENT_DATA", "metrics": {"recorded_sessions": len(days), "usable": len(rows)},
                "note": "keep `python -m nifty_engine run` (or `record`) going during market hours"}
    X = pd.DataFrame(rows).set_index("date")
    t = stack_test(lab.oos(("base",)), X, min_train=40)
    res = stats.dm_logloss(t["p"].values, t["p_b"].values, t["y_up"].values.astype(int)) if len(t) else {"n": 0}
    return {"status": "PASS" if res.get("p", 1) < 0.10 and res.get("delta_logloss", 0) > 0 else "FAIL", "metrics": res}


RUNNERS = {"benchmark": None, "feature_group": exp_feature_group, "short_history_group": exp_short_group,
           "strategy": exp_strategy, "meta": exp_meta, "forward": None}
BENCH = {"P0": exp_benchmark_base, "S7b": exp_tom_futures, "S8b": exp_smc_index}
FORWARD = {"F1": exp_forward_brief, "F2": exp_forward_live}


def run_all(cfg: EngineConfig, only: Optional[List[str]] = None, quick: bool = False) -> Dict:
    reg = registry()
    lab = Lab(cfg, quick=quick)
    results: Dict[str, Dict] = {}
    for e in reg["experiments"]:
        if only and e["id"] not in only:
            continue
        fn = BENCH.get(e["id"]) or FORWARD.get(e["id"]) or RUNNERS.get(e["kind"])
        log.info("experiment %s: %s", e["id"], e["name"])
        try:
            r = fn(lab, e, reg) if fn else {"status": "SKIPPED"}
        except Exception as ex:  # noqa: BLE001
            r = {"status": "ERROR", "error": str(ex)[:300], "trace": traceback.format_exc()[-1500:]}
        results[e["id"]] = {**e, **r}
    # deflated Sharpe across every strategy trial run (variants count as trials)
    strat = {k: v for k, v in results.items() if v.get("status") == "PENDING_DSR"}
    trials = sum(len(v.get("extra", {}).get("variants", {})) or 1 for v in strat.values())
    srs = [stats.sharpe(v["_trades"]["net_pnl"].values) for v in strat.values() if len(v["_trades"]) > 2]
    sr_var = float(np.var(srs, ddof=1)) if len(srs) > 1 else None
    for k, v in strat.items():
        T = v.pop("_trades")
        d = stats.deflated_sharpe(T["net_pnl"].values, trials, sr_var) if len(T) > 2 else {"dsr": float("nan")}
        v["metrics"]["deflated_sharpe"] = {kk: (round(x, 4) if isinstance(x, float) else x) for kk, x in d.items()}
        v["status"] = _strategy_status(v["metrics"], d, reg["rules"]["strategy"])
        if len(T):
            REPORTS.mkdir(parents=True, exist_ok=True)
            T.to_csv(REPORTS / f"trades_{v['book']}.csv", index=False)
    for v in results.values():
        v.pop("_trades", None)
    prev_file = REPORTS / "research.json"
    if only and prev_file.exists():      # a partial run updates the earlier report instead of replacing it
        prev = json.loads(prev_file.read_text(encoding="utf-8"))
        merged = {**prev.get("results", {}), **results}
        order = [e["id"] for e in reg["experiments"]]
        results = {k: merged[k] for k in order if k in merged}
        trials = max(trials, int(prev.get("n_strategy_trials", 0)))
    out = {"generated": dt.datetime.now(IST).isoformat(), "quick": quick, "n_strategy_trials": trials,
           "results": results}
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "research.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    (REPORTS / "RESEARCH_REPORT.md").write_text(render_markdown(out, reg), encoding="utf-8")
    return out


def _fmt_metrics(r: Dict) -> str:
    m = r.get("metrics") or {}
    k = r.get("kind")
    if k in ("feature_group", "short_history_group") or r.get("id") in ("F1", "F2"):
        if "delta_logloss" in m:
            return (f"n={m.get('n')}, log-loss gain {m.get('delta_logloss', float('nan')):+.5f}, DM t={m.get('t', float('nan')):.2f}, "
                    f"p={m.get('p', float('nan')):.3f}, acc {m.get('acc_a', float('nan')) * 100:.1f}% -> {m.get('acc_b', float('nan')) * 100:.1f}%")
        if "vs_model" in m:
            v = m["vs_model"]
            return f"days={m.get('days')}, LLM acc {m.get('llm_accuracy')}, vs model p={v.get('p', float('nan')):.3f}"
    if k == "strategy" and m.get("trades"):
        d = m.get("deflated_sharpe", {})
        return (f"{m['trades']} trades, win {m['win_rate'] * 100:.0f}%, net/trade Rs {m['net_per_trade']:,.0f} "
                f"(95% CI {m['net_per_trade_95ci'][0]:,.0f} to {m['net_per_trade_95ci'][1]:,.0f}), t={m['t_stat']}, "
                f"DSR={d.get('dsr', float('nan')):.2f}, total Rs {m['net_total']:,.0f}, max DD {m['max_drawdown_pct']}%, "
                f"real prices {m.get('real_price_share', 0) * 100:.0f}%")
    if r.get("id") == "P0" and m:
        return (f"{m.get('n_days')} days, accuracy {m.get('accuracy', 0) * 100:.1f}%, AUC {m.get('auc') or 0:.3f}, "
                f"skill p={m.get('skill_p_value', float('nan')):.3f}")
    if m:
        return ", ".join(f"{a}={b}" for a, b in list(m.items())[:6] if not isinstance(b, (dict, list)))
    return r.get("note") or r.get("error") or ""


def render_markdown(out: Dict, reg: Dict) -> str:
    L = ["# NIFTY v5 research report", "",
         f"_Generated {out['generated'][:16]} IST{' (quick mode: logistic model only)' if out.get('quick') else ''}. "
         f"Rules were fixed in `config/experiments.json` on {reg.get('registered')} before any test ran._", "",
         f"Strategy trials counted for the deflated Sharpe: **{out['n_strategy_trials']}**.", "",
         "| ID | Experiment | Status | Result |", "|---|---|---|---|"]
    for k, r in out["results"].items():
        L.append(f"| {k} | {r['name']} | **{r['status']}** | {_fmt_metrics(r)} |")
    L += ["", "## Rules", ""]
    L += [f"* **{k}**: {v['text']}" for k, v in reg["rules"].items()]
    L += ["", "## What the statuses mean", "",
          "* **PASS**: met the pre-registered bar. It is a candidate for paper trading or for `feature_groups`, not yet for real money.",
          "* **PROMISING**: positive but not statistically convincing. Keep paper trading it.",
          "* **FAIL**: did not beat the bar. Reported so the number of trials stays honest.",
          "* **INSUFFICIENT_DATA**: download or record more (the note says what).", "",
          "Per-trade files: `store/reports/trades_<book>.csv`. Full numbers: `store/reports/research.json`."]
    details = [r for r in out["results"].values() if r.get("extra")]
    if details:
        L += ["", "## Variant details", ""]
        for r in details:
            L.append(f"### {r['id']} {r['name']}")
            L.append("```json\n" + json.dumps(r["extra"], indent=1, default=str) + "\n```")
    return "\n".join(L)
