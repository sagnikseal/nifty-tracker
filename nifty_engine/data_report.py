"""`python -m nifty_engine data-report`: what data this machine has, what is missing, how to get it.
Writes store/reports/DATA_REPORT.md (also shown in the dashboard's Reports tab)."""

from __future__ import annotations

import datetime as dt
import json
from typing import Dict, List

import pandas as pd

from .config import DATA_DIR, EngineConfig


def _csv_range(p, index_col=0) -> Dict:
    if not p.exists():
        return {"exists": False}
    try:
        df = pd.read_csv(p, usecols=[0])
        s = pd.to_datetime(df.iloc[:, 0], utc=True, errors="coerce").dropna().dt.tz_convert("Asia/Kolkata")
        return {"exists": True, "rows": int(len(df)), "first": str(s.min())[:16], "last": str(s.max())[:16]}
    except Exception as e:  # noqa: BLE001
        return {"exists": True, "error": str(e)[:80]}


def inventory(cfg: EngineConfig) -> Dict:
    from .data.providers import YAHOO_SYMBOLS
    inv: Dict = {"generated": dt.datetime.now().isoformat(timespec="minutes")}
    inv["daily"] = {n: _csv_range(DATA_DIR / "raw" / f"{n}.csv") for n in YAHOO_SYMBOLS}
    inv["hourly"] = {p.stem: _csv_range(p) for p in sorted((DATA_DIR / "raw_1h").glob("*.csv"))} if (DATA_DIR / "raw_1h").exists() else {}
    inv["intraday"] = {p.stem: _csv_range(p) for p in sorted((DATA_DIR / "intraday").glob("*.csv"))}
    for name in ("options_hist", "options_hist_1m"):
        fs = sorted((DATA_DIR / name).glob("*.csv"))
        inv[name] = {"expiries": len(fs), "first": fs[0].stem if fs else None, "last": fs[-1].stem if fs else None}
    live = DATA_DIR / "live"
    inv["recorded_sessions"] = sorted(p.name for p in live.iterdir() if p.is_dir()) if live.exists() else []
    heads = DATA_DIR / "news" / "headlines"
    n_heads, days = 0, []
    if heads.exists():
        for f in sorted(heads.glob("*.jsonl")):
            n_heads += sum(1 for _ in f.open(encoding="utf-8"))
            days.append(f.stem)
    sc = DATA_DIR / "news" / "scores.jsonl"
    n_llm = 0
    if sc.exists():
        for line in sc.open(encoding="utf-8"):
            try:
                n_llm += int("llm" in json.loads(line))
            except ValueError:
                pass
    inv["news"] = {"headlines": n_heads, "days": len(days), "first": days[0] if days else None,
                   "last": days[-1] if days else None, "llm_scored": n_llm}
    inv["briefs"] = len(list((DATA_DIR / "briefs").glob("*.json"))) if (DATA_DIR / "briefs").exists() else 0
    inv["agent_decision_days"] = len(list((DATA_DIR / "agent").glob("*.jsonl"))) if (DATA_DIR / "agent").exists() else 0
    poi = DATA_DIR / "nse" / "participant_oi"
    inv["participant_oi_files"] = len(list(poi.glob("*.csv"))) if poi.exists() else 0
    return inv


MISSING: List[Dict] = [
    {"what": "Live option-chain and order-book history (OI, IV, depth, PCR, GEX)",
     "why": "Positioning data is the most promising intraday predictor and cannot be downloaded later",
     "how": "Keep the engine running on market days with UPSTOX_ANALYTICS_TOKEN set (recorder: chain every 30 s, depth every 5 s). 60 sessions unlock experiment F2."},
    {"what": "Historical news with first-seen timestamps during market hours",
     "why": "The news-impact model needs >= 300 in-session events on >= 40 days to pass its test",
     "how": "`python -m nifty_engine news-backfill` (GDELT, ~3 months) once; then the engine collects every minute. Re-run `news-impact` weekly."},
    {"what": "GIFT Nifty (pre-open index futures in GIFT City)",
     "why": "Best single pre-open signal for NIFTY's gap",
     "how": "Not on free feeds or Upstox. The engine uses overnight ES/NQ futures as a proxy; a paid NSE IX feed would add it."},
    {"what": "FII/DII daily flows and participant-wise open interest",
     "why": "Explains multi-day trend; participant OI shows how institutions are positioned in index futures/options",
     "how": "NSE publishes daily CSVs but forbids scraping. Save fao_participant_oi_DDMMYYYY.csv by hand into store/nse/participant_oi/ (loader exists)."},
    {"what": "NIFTY futures (volume, OI, basis) and option-chain positioning before Sept/Oct 2024",
     "why": "Upstox serves expired contracts only about 2 years back, so these groups have ~2 years of history vs ~5 for prices",
     "how": "Nothing to buy cheaply; history grows by itself (the GitHub collector and the engine store every session)."},
    {"what": "Option strikes far from the money (full chain) in history",
     "why": "Walls/max pain from strikes beyond ATM +/- ~5 are not in the downloaded candles",
     "how": "`python -m nifty_engine fetch-expired-options --mode full` (hours, Upstox Plus) widens to ATM +/- 800 points."},
    {"what": "Expiry-day (0DTE) option prices for past weeks",
     "why": "Tests of expiry-day strategies",
     "how": "Upstox may not serve expired expiry-day bars; the recorder collects them going forward."},
]


def write(cfg: EngineConfig) -> Dict:
    inv = inventory(cfg)
    L = ["# Data report: what the engine has, what is missing, how to get it", "",
         f"_Generated {inv['generated']} on this machine._", "", "## What we have", "",
         "| Dataset | Rows / items | From | To |", "|---|---|---|---|"]
    for n, r in inv["intraday"].items():
        L.append(f"| intraday {n} | {r.get('rows', '—'):,} | {r.get('first', '—')} | {r.get('last', '—')} |"
                 if isinstance(r.get("rows"), int) else f"| intraday {n} | — | — | — |")
    for k in ("options_hist_1m", "options_hist"):
        r = inv[k]
        L.append(f"| real expired-option prices ({'1-min' if k.endswith('1m') else '5-min'}) | {r['expiries']} weekly expiries | {r['first']} | {r['last']} |")
    have_daily = {n: r for n, r in inv["daily"].items() if r.get("exists")}
    if have_daily:
        firsts = [r["first"] for r in have_daily.values() if r.get("first")]
        lasts = [r["last"] for r in have_daily.values() if r.get("last")]
        L.append(f"| daily series (NIFTY, VIX, global markets, ADRs) | {len(have_daily)} of {len(inv['daily'])} series | "
                 f"{min(firsts)[:10] if firsts else '—'} | {max(lasts)[:10] if lasts else '—'} |")
    if inv["hourly"]:
        L.append(f"| hourly overnight futures (ES, NQ, crude, gold, BTC, USDINR) | {len(inv['hourly'])} series | — | — |")
    n = inv["news"]
    L.append(f"| news headlines (first-seen stamped) | {n['headlines']:,} on {n['days']} days, {n['llm_scored']:,} LLM-scored | {n['first'] or '—'} | {n['last'] or '—'} |")
    L.append(f"| recorded live sessions (option chain + depth) | {len(inv['recorded_sessions'])} sessions | "
             f"{inv['recorded_sessions'][0] if inv['recorded_sessions'] else '—'} | {inv['recorded_sessions'][-1] if inv['recorded_sessions'] else '—'} |")
    L.append(f"| LLM pre-open briefs / agent decision days | {inv['briefs']} / {inv['agent_decision_days']} | — | — |")
    L.append(f"| NSE participant-wise OI files | {inv['participant_oi_files']} | — | — |")
    missing_daily = [n for n, r in inv["daily"].items() if not r.get("exists")]
    if missing_daily:
        L += ["", f"Missing daily series (run `python -m nifty_engine update-data`): {', '.join(missing_daily)}"]
    L += ["", "## Features engineered from this data", "",
          "* **Movement model (every 5 minutes):** moves over 5/15/30/60 minutes and since the open/previous close; "
          "opening gap and how much of it has filled; realised volatility today and in the last hour, relative to "
          "normal; position in the day's range; distance from the day's average price and a 1-hour average; opening-"
          "range breakout (09:15-09:45); first-30-minute move (intraday momentum); share of rising bars and runs; "
          "RSI on 5-minute bars; distance to the previous day's high, low and pivot; previous day's last-hour move; "
          "India VIX level and its change since yesterday, since the open and over 30 minutes; time of day, day of "
          "week, expiry day and days to expiry; the daily model's out-of-sample probability; and, where news exists, "
          "LLM news tone/shock over 30 minutes and 2 hours, pre-open tone and the learned news impact.",
          "* **Daily model:** see `docs/RESEARCH.md` (returns, volatility, global markets with strict timing, Asia "
          "opening gaps, VIX regime, calendar; optional SMC, structure, regime, events, ADR, overnight futures, news, "
          "historical option-chain groups, each tested before use).", "",
          "## What is missing and how to get it", "", "| Data | Why it matters | How to get it |", "|---|---|---|"]
    L += [f"| {m['what']} | {m['why']} | {m['how']} |" for m in MISSING]
    p = DATA_DIR / "reports" / "DATA_REPORT.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(L), encoding="utf-8")
    return inv
