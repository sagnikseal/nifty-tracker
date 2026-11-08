"""Replay the ATM rules (strategies/atm_rules.py) on REAL 1-minute prices of the expired weekly
NIFTY options, driven by the intraday model's OUT-OF-SAMPLE walk-forward predictions.

Fill model (conservative):
  * entry: the option's 1-minute bar starting at the decision time T, at its open plus a modelled
    half-spread plus slippage;
  * exits: sell at the trigger level (or the bar's open if the price gapped through it) minus the
    half-spread and slippage; when the stop and the target fall in the same 1-minute bar, the stop is
    assumed to have come first;
  * full NSE charges (2026 STT) on both orders.
Every trade records which rule closed it. A pre-registered grid of variants is reported in full,
with the deflated Sharpe ratio across all of them.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..calendar import IST
from ..config import DATA_DIR, EngineConfig
from ..options import half_spread
from ..research.stats import deflated_sharpe, pbo_cscv, trade_summary
from ..strategies.atm_rules import ATMRules, PositionState, on_bar_close, on_price
from .charges import order_charges
from .sim import rates_from

log = logging.getLogger(__name__)
OPT_DIR = DATA_DIR / "options_hist_1m"
REPORTS = DATA_DIR / "reports"

# Pre-registered variants (fixed before the first run; every one is reported).
GRID = [
    {"name": f"thr{thr:.2f}_{scope}", "entry_threshold": thr, "against_scope": scope}
    for thr in (0.55, 0.60, 0.65) for scope in ("in_profit", "always", "after_target")
] + [
    {"name": "thr0.60_no_extension", "entry_threshold": 0.60, "against_scope": "in_profit", "extend_prob": 1.01},
    {"name": "thr0.60_time_exit_only", "entry_threshold": 0.60, "against_scope": "in_profit",
     "target_inr": 1e12, "stop_inr": 1e12, "against_bars": 10_000},
]

# Entries from the daily model at 09:20 (its proven out-of-sample edge, engine threshold 0.56), to test
# the exit rules themselves on real option prices. Pre-registered on 2026-11-08 after the intraday
# model was found to abstain, before any of these were run.
GRID_DAILY = [
    {"name": "daily_in_profit", "entry_threshold": 0.56, "against_scope": "in_profit"},
    {"name": "daily_always", "entry_threshold": 0.56, "against_scope": "always"},
    {"name": "daily_after_target", "entry_threshold": 0.56, "against_scope": "after_target"},
    {"name": "daily_stop2000_only", "entry_threshold": 0.56, "target_inr": 1e12, "against_bars": 10_000},
    {"name": "daily_time_exit_only", "entry_threshold": 0.56, "target_inr": 1e12, "stop_inr": 1e12,
     "against_bars": 10_000},
]


def expiries() -> List[dt.date]:
    return sorted(dt.date.fromisoformat(p.stem) for p in OPT_DIR.glob("*.csv"))


def load_expiry(exp: dt.date) -> pd.DataFrame:
    f = OPT_DIR / f"{exp.isoformat()}.csv"
    if not f.exists():
        return pd.DataFrame()
    df = pd.read_csv(f, usecols=["ts", "open", "high", "low", "close", "strike", "type"])
    df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert(IST)
    return df


def _exit_px(mid: float, slip: float) -> float:
    return max(0.05, mid - half_spread(mid) - slip)


def replay_day(rules: ATMRules, cfg: EngineConfig, d: dt.date, preds: pd.DataFrame, spot5: pd.Series,
               opt: pd.DataFrame, exp: dt.date, rates) -> Optional[Dict]:
    """preds: rows for day d indexed by decision time T with columns p_close, p30."""
    slip = cfg.option_slippage_pts
    for T, r in preds.iterrows():
        direction = rules.entry_direction(r["p_close"], T.to_pydatetime(), 0)
        if direction == 0:
            continue
        s_prev = T - pd.Timedelta(minutes=5)
        if s_prev not in spot5.index:
            continue
        S = float(spot5.loc[s_prev])
        strike = round(S / 50) * 50
        typ = "CE" if direction > 0 else "PE"
        bars = opt[(opt["strike"] == strike) & (opt["type"] == typ) & (opt["ts"].dt.date == d)].set_index("ts").sort_index()
        bars = bars[bars.index >= T]
        if len(bars) < 5 or bars.index[0] - T > pd.Timedelta(minutes=2):
            return {"date": d.isoformat(), "skipped": "no_option_bars", "strike": strike, "type": typ}
        mid0 = float(bars["open"].iloc[0])
        entry = mid0 + half_spread(mid0) + slip
        st = PositionState(direction, entry, rules.qty, T.to_pydatetime(), S)
        p30_series = preds["p30"]
        reason, exit_mid, exit_ts = None, None, None
        exit_t = dt.time(*map(int, rules.exit_time.split(":")))
        for ts, b in bars.iterrows():
            now = ts.to_pydatetime()
            if now.time() >= exit_t:
                reason, exit_mid, exit_ts = "time", float(b["open"]), ts
                break
            # latest 30-minute prediction available at this minute (made at the last 5-min close <= now)
            avail = p30_series[p30_series.index <= ts]
            p30 = float(avail.iloc[-1]) if len(avail) else None
            # 5-minute bar close happens at minutes divisible by 5 (bar [ts-5, ts) just closed)
            if ts.minute % 5 == 0 and ts > T:
                s5 = ts - pd.Timedelta(minutes=5)
                if s5 in spot5.index:
                    rr = on_bar_close(rules, st, float(spot5.loc[s5]), _exit_px(float(b["open"]), slip), now)
                    if rr:
                        reason, exit_mid, exit_ts = rr, float(b["open"]), ts
                        break
            # stop first (worst case inside the bar), then target / extension
            lo_px = _exit_px(float(b["low"]), slip)
            if st.pnl(lo_px) <= -rules.stop_inr:
                stop_mid = entry - rules.stop_inr / rules.qty + half_spread(entry) + slip
                reason, exit_mid, exit_ts = "stop", min(stop_mid, float(b["open"])), ts
                break
            hi_px = _exit_px(float(b["high"]), slip)
            if not st.target_hit and st.pnl(hi_px) >= rules.target_inr:
                tgt_mid = entry + rules.target_inr / rules.qty + half_spread(entry) + slip
                fill_mid = max(tgt_mid, float(b["open"]))
                rr = on_price(rules, st, _exit_px(fill_mid, slip), now, p30)
                if rr:
                    reason, exit_mid, exit_ts = rr, fill_mid, ts
                    break
                continue
            rr = on_price(rules, st, _exit_px(float(b["close"]), slip), now, p30)
            if rr:
                reason, exit_mid, exit_ts = rr, float(b["close"]), ts
                break
        if reason is None:
            reason, exit_mid, exit_ts = "time", float(bars["close"].iloc[-1]), bars.index[-1]
        exit_px = _exit_px(exit_mid, slip)
        q = rules.qty
        ch = order_charges("option", "BUY", entry * q, rates)["total"] + order_charges("option", "SELL", exit_px * q, rates)["total"]
        gross = (exit_px - entry) * q
        return {"date": d.isoformat(), "entry_ts": T.isoformat(), "exit_ts": exit_ts.isoformat(),
                "direction": direction, "type": typ, "strike": strike, "expiry": exp.isoformat(), "spot": round(S, 2),
                "p_close": round(float(r["p_close"]), 4), "p30_at_entry": None if pd.isna(r["p30"]) else round(float(r["p30"]), 4),
                "entry": round(entry, 2), "exit": round(exit_px, 2), "premium_outlay": round(entry * q, 0),
                "exit_reason": reason, "extensions": st.extensions, "target_hit": st.target_hit,
                "peak_pnl": round(st.peak_pnl, 0), "gross_pnl": round(gross, 2), "charges": round(ch, 2),
                "net_pnl": round(gross - ch, 2), "price_source": "real"}
    return None


def run(cfg: EngineConfig, variants: Optional[List[Dict]] = None, start: Optional[str] = None,
        entry: str = "intraday", tag: str = "") -> Dict:
    """entry="intraday": scan the intraday model's P(up by 15:15) through the entry window.
    entry="daily": enter at 09:20 only, from the daily model's out-of-sample P(close > open)."""
    from ..intraday_model import load_bars, load_oos
    o30, oc = load_oos("30m"), load_oos("close")
    if oc.empty:
        raise RuntimeError("No intraday out-of-sample predictions. Run: python -m nifty_engine intraday-train")
    P = pd.DataFrame({"p_close": oc["p"], "p30": o30["p"].reindex(oc.index)})
    P["date"] = [t.date() for t in P.index]
    if entry == "daily":
        first = ~P["date"].duplicated()
        P["p_close"] = np.where(first & (P.index.time == dt.time(9, 20)), oc["p_daily"].values, np.nan)
        variants = variants or GRID_DAILY
    n5 = load_bars("nifty_5m")
    spot5 = n5["close"]
    exps = expiries()
    if not exps:
        raise RuntimeError("No real 1-minute option prices in store/options_hist_1m")
    days = sorted(set(P["date"]))
    days = [d for d in days if exps[0] - dt.timedelta(days=10) <= d <= exps[-1]]
    if start:
        days = [d for d in days if d >= dt.date.fromisoformat(start)]
    day2exp = {}
    for d in days:
        nxt = [e for e in exps if (e - d).days >= 1]
        if nxt:
            day2exp[d] = nxt[0]
    rates = rates_from(cfg)
    base = ATMRules.from_cfg(cfg)
    variants = variants or GRID
    cache: Dict[dt.date, pd.DataFrame] = {}
    results = {}
    by_day = {d: g for d, g in P.groupby("date")}
    for v in variants:
        rules = ATMRules(**{**base.__dict__, **{k: val for k, val in v.items() if k != "name"}})
        rows, skipped = [], {}
        for d in days:
            if d not in day2exp or d not in by_day:
                continue
            exp = day2exp[d]
            if exp not in cache:
                cache[exp] = load_expiry(exp)
            res = replay_day(rules, cfg, d, by_day[d], spot5, cache[exp], exp, rates)
            if res is None:
                continue
            if "skipped" in res:
                skipped[res["skipped"]] = skipped.get(res["skipped"], 0) + 1
                continue
            rows.append(res)
        T = pd.DataFrame(rows)
        summ = trade_summary(T, capital=cfg.book_capital) if len(T) else {"trades": 0}
        if len(T):
            summ["exit_reasons"] = T["exit_reason"].value_counts().to_dict()
            summ["avg_premium_outlay"] = round(float(T["premium_outlay"].mean()), 0)
            summ["max_premium_outlay"] = round(float(T["premium_outlay"].max()), 0)
            summ["extensions_total"] = int(T["extensions"].sum())
            summ["target_hit_rate"] = round(float(T["target_hit"].mean()), 3)
            summ["direction_long_share"] = round(float((T["direction"] > 0).mean()), 3)
        summ["skipped"] = skipped
        summ["days_considered"] = len([d for d in days if d in day2exp])
        results[v["name"]] = {"variant": v, "summary": summ, "trades": T}
        log.info("ATM replay %s: %s", v["name"], {k: summ.get(k) for k in ("trades", "win_rate", "net_total", "net_per_trade")})
    # deflated Sharpe across all variants (per-trade net P&L)
    srs = [np.mean(r["trades"]["net_pnl"]) / np.std(r["trades"]["net_pnl"], ddof=1)
           for r in results.values() if len(r["trades"]) > 2 and np.std(r["trades"]["net_pnl"], ddof=1) > 0]
    var_sr = float(np.var(srs, ddof=1)) if len(srs) > 1 else None
    for r in results.values():
        if len(r["trades"]) > 2:
            r["summary"]["dsr"] = deflated_sharpe(r["trades"]["net_pnl"].values, len(results), var_sr)
    M = pd.DataFrame({k: r["trades"].assign(m=r["trades"]["date"].str[:7]).groupby("m")["net_pnl"].sum()
                      for k, r in results.items() if len(r["trades"])})
    pbo = pbo_cscv(M, S=min(10, max(2, len(M) // 2 * 2))) if len(M) >= 8 else {"pbo": float("nan"), "splits": 0}
    out = {"variants": {k: r["summary"] for k, r in results.items()}, "pbo": pbo, "entry": entry,
           "period": [str(days[0]), str(days[-1])] if days else None, "n_trials": len(results)}
    REPORTS.mkdir(parents=True, exist_ok=True)
    for k, r in results.items():
        if len(r["trades"]):
            r["trades"].to_csv(REPORTS / f"atm_trades_{k}.csv", index=False)
    (REPORTS / f"atm_replay{tag}.json").write_text(json.dumps(out, indent=1, default=str))
    return out


def run_all(cfg: EngineConfig) -> Dict:
    """Both pre-registered grids -> one report."""
    a = run(cfg, entry="intraday")
    b = run(cfg, entry="daily", tag="_daily")
    write_report([a, b], ATMRules.from_cfg(cfg), REPORTS / "ATM_REPLAY_REPORT.md")
    return {"intraday": a, "daily": b}


def _row(k: str, s: Dict, bold: bool) -> str:
    if not s.get("trades"):
        return f"| {k} | 0 | — | — | — | — | — | — | — | — | — | — | skipped: {s.get('skipped') or 'none'} |"
    dsr = (s.get("dsr") or {}).get("dsr")
    dsr_s = "—" if dsr is None or dsr != dsr else "%.2f" % dsr
    reasons = ", ".join("%s: %s" % (a, b) for a, b in s.get("exit_reasons", {}).items())
    name = "**%s**" % k if bold else k
    return (f"| {name} | {s['trades']} | {s['win_rate']:.1%} | {s['net_total']:,.0f} | {s['net_per_trade']:,.0f} | "
            f"{s['net_per_trade_95ci'][0]:,.0f} to {s['net_per_trade_95ci'][1]:,.0f} | {s['t_stat']:.2f} | {dsr_s} | "
            f"{s['max_drawdown_pct']:.1f}% | {s.get('avg_premium_outlay', 0):,.0f} | {s.get('target_hit_rate', 0):.1%} | "
            f"{s.get('extensions_total', 0)} | {reasons} |")


def write_report(outs: List[Dict], base: ATMRules, path: Path) -> None:
    head = ("| Variant | Trades | Win rate | Net total (Rs) | Net/trade (Rs) | 95% CI | t | DSR | Max DD | "
            "Avg premium (Rs) | Target hit | Extensions | Exit reasons |")
    L = ["# Your ATM rules, replayed on real option prices", "",
         "Real 1-minute prices of the expired weekly NIFTY options (nearest expiry with at least one day left, "
         "at-the-money strike at entry). Every probability used is out-of-sample. Full NSE charges, a modelled "
         "bid-ask spread and 1 point of slippage per order; when the stop and the target fall inside the same "
         "minute, the stop is assumed first. At most one trade per day.", "",
         f"Rules tested: target Rs {base.target_inr:,.0f}; stop Rs {base.stop_inr:,.0f}; hold another "
         f"{base.extend_minutes} min while P(move continues) > {base.extend_prob:.0%} (re-checked, repeatable); exit "
         f"after {base.against_bars} consecutive 5-minute closes against the trade; time exit {base.exit_time}. "
         "`in_profit` / `always` / `after_target` = when the 2-closes-against rule applies.", ""]
    for out in outs:
        if not out.get("period"):
            continue
        title = ("A. Entries from the intraday model (P(up by 15:15) scanned 09:20-13:30)" if out["entry"] == "intraday"
                 else "B. Entries from the daily model at 09:20 (its proven out-of-sample edge), testing the exit rules")
        pbo = out["pbo"].get("pbo", float("nan"))
        L += [f"## {title}", "", f"Period {out['period'][0]} to {out['period'][1]}. {out['n_trials']} pre-registered "
              f"variants, all shown. DSR = deflated Sharpe across them; PBO = "
              f"{'—' if pbo != pbo else '%.2f' % pbo} ({out['pbo'].get('splits', 0)} splits).", "", head,
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        default = f"thr{base.entry_threshold:.2f}_{base.against_scope}" if out["entry"] == "intraday" else "daily_in_profit"
        L += [_row(k, s, k == default) for k, s in out["variants"].items()]
        L.append("")
    L += ["How to read this: a variant is only worth paper-trading with real conviction if its net per trade is "
          "positive with a 95% confidence interval above zero, t ≥ 2 and DSR ≥ 0.95. Win rate alone means little: "
          "a Rs 5,000 target and a Rs 2,000 stop make wins and losses very different sizes."]
    path.write_text("\n".join(L), encoding="utf-8")
