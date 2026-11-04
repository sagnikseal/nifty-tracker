"""Forward-collected news sentiment (broker-independent).

Why forward-collected: free *historical* headline archives with timestamps don't exist for
Indian markets (GDELT's DOC API keeps ~3 months). So the engine stores a daily sentiment
row every evening; once >= ~1 year accumulates, the model's missing-data gate admits the
feature automatically and walk-forward decides whether it actually helps.

Scoring is a transparent finance lexicon (Loughran-McDonald style). Research notes:
LLM/FinBERT scores can beat lexicons, but (a) LLMs have look-ahead and "distraction"
biases on history (Glasserman & Lin 2023) and (b) the tradable part of news is small and
fast (Lopez-Lira & Tang). Plug an LLM scorer in `score_headline` only for live use.
"""

from __future__ import annotations

import datetime as dt
import re
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from typing import Dict, List

import numpy as np
import pandas as pd
import requests

from .calendar import IST
from .config import DATA_DIR
from .data.providers import UA, GdeltProvider

NEWS_FILE = DATA_DIR / "news" / "daily_sentiment.csv"

FEEDS = {
    "et_markets": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "et_economy": "https://economictimes.indiatimes.com/news/economy/rssfeeds/1373380680.cms",
    "livemint_markets": "https://www.livemint.com/rss/markets",
    "bs_markets": "https://www.business-standard.com/rss/markets-106.rss",
    "gnews_india_mkts": "https://news.google.com/rss/search?q=Nifty+OR+Sensex+OR+FII+India+stocks&hl=en-IN&gl=IN&ceid=IN:en",
    "gnews_global": "https://news.google.com/rss/search?q=Fed+OR+treasury+yields+OR+crude+oil+OR+geopolitical&hl=en-US&gl=US&ceid=US:en",
}

POS = {"surge", "rally", "gain", "gains", "jumps", "soar", "record high", "upgrade", "beat", "beats", "inflow",
       "inflows", "rate cut", "cuts rates", "easing", "ceasefire", "truce", "boost", "strong", "optimism",
       "recovery", "rebound", "bullish", "buying", "outperform", "eases", "cools", "deal"}
NEG = {"crash", "plunge", "slump", "falls", "fall", "drops", "selloff", "sell-off", "outflow", "outflows",
       "downgrade", "miss", "misses", "rate hike", "hikes", "war", "missile", "strike", "attack", "sanction",
       "sanctions", "tariff", "tariffs", "recession", "inflation surge", "default", "bearish", "fear", "panic",
       "escalation", "tension", "tensions", "volatile", "slowdown", "weak", "loss", "losses", "probe", "ban"}


def score_headline(text: str) -> float:
    t = " " + re.sub(r"[^a-z\- ]", " ", text.lower()) + " "
    p = sum(1 for w in POS if f" {w} " in t)
    n = sum(1 for w in NEG if f" {w} " in t)
    return 0.0 if p + n == 0 else (p - n) / (p + n)


def fetch_headlines(timeout: float = 8.0) -> List[Dict]:
    out = []
    for src, url in FEEDS.items():
        try:
            r = requests.get(url, headers=UA, timeout=timeout)
            root = ET.fromstring(r.content)
            for it in root.iter("item"):
                title = (it.findtext("title") or "").strip()
                pub = it.findtext("pubDate")
                try:
                    ts = parsedate_to_datetime(pub).astimezone(IST) if pub else None
                except Exception:  # noqa: BLE001
                    ts = None
                if title:
                    out.append({"source": src, "title": title, "ts": ts})
        except Exception:  # noqa: BLE001
            continue
    return out


def collect_daily(day: dt.date | None = None) -> Dict:
    """Score headlines published in the 24h before 09:00 IST of `day` (default: tomorrow's
    session when run in the evening) and append one row to the store."""
    day = day or dt.date.today()
    heads = fetch_headlines()
    cutoff_end = dt.datetime.combine(day, dt.time(9, 0), tzinfo=IST)
    cutoff_start = cutoff_end - dt.timedelta(hours=24)
    sel = [h for h in heads if h["ts"] is None or cutoff_start <= h["ts"] <= cutoff_end]
    scores = [score_headline(h["title"]) for h in sel]
    nz = [s for s in scores if s != 0]
    row = {"date": day.isoformat(), "news_n": len(sel), "news_sent": sum(nz) / len(nz) if nz else 0.0,
           "news_neg_share": sum(1 for s in nz if s < 0) / len(nz) if nz else 0.0}
    try:
        tone = GdeltProvider().daily_tone('(India OR Nifty OR Sensex) (market OR stocks OR economy)', "7d")
        row["gdelt_tone"] = float(tone.iloc[-1]) if len(tone) else None
    except Exception:  # noqa: BLE001
        row["gdelt_tone"] = None
    NEWS_FILE.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(NEWS_FILE) if NEWS_FILE.exists() else pd.DataFrame()
    df = pd.concat([df, pd.DataFrame([row])]).drop_duplicates("date", keep="last")
    df.to_csv(NEWS_FILE, index=False)
    return row


def load_news_series() -> Dict[str, pd.Series]:
    """News features keyed by the session date they are meant for (already pre-open)."""
    if not NEWS_FILE.exists():
        return {}
    df = pd.read_csv(NEWS_FILE, parse_dates=["date"]).set_index("date").sort_index()
    out = {}
    for c in ("news_sent", "news_neg_share", "gdelt_tone"):
        if c in df and df[c].notna().sum() > 0:
            # stored for session `date`; features.py lags extras by 1 day, so shift index back
            s = df[c].astype(float)
            s.index = s.index - pd.Timedelta(days=1)
            out[c] = s
    return out


# ----------------------------------------------------------------------------- pre-open overlay
def premarket_overlay(direction: int = 0) -> Dict:
    """Pre-open news overlay used at 09:20 (UNVALIDATED until enough history exists to test it).

    Combines the legacy v3 MultiEndpointMediaEngine (15 feeds, 8 transmission channels,
    war / central-bank / crude shock scanner) with the v4 lexicon collector. Returns a
    composite score in [-1, 1], a shock intensity in [0, 1], top headlines, and a risk-gate
    decision: if a strong shock points AGAINST the model's direction, the trade is vetoed.
    """
    out: Dict = {"validated": False, "sources": []}
    composite, shock = 0.0, 0.0
    try:
        import importlib.util
        from .config import ROOT
        spec = importlib.util.spec_from_file_location("media_news_engine", ROOT / "scripts" / "media_news_engine.py")
        import sys
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod          # dataclasses need the module registered
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        res = mod.MultiEndpointMediaEngine().analyze_media_ecosystem(force_refresh=True)
        f = res.get("features", {})
        gates = res.get("non_linear_gates", {})
        shock = float(gates.get("news_shock_intensity", 0.0))
        v3 = float(np.clip(0.5 * f.get("news_finbert_sentiment", 0.0) + 0.3 * f.get("geopolitical_war_shock", 0.0)
                           + 0.2 * f.get("central_bank_event_shock", 0.0), -2, 2)) / 2
        composite += v3
        out.update({"v3_features": f, "crisis_level": res.get("crisis_level"),
                    "geopolitical_alert": res.get("geopolitical_alert"),
                    "channels": {k: v.get("stance") for k, v in res.get("channel_transmission", {}).items()},
                    "headlines": [h.get("title") for h in res.get("scored_headlines", [])[:6]],
                    "endpoints_live": res.get("endpoints_live")})
        out["sources"].append("v3_media_engine")
    except Exception as e:  # noqa: BLE001
        out["v3_error"] = str(e)[:200]
    try:
        heads = fetch_headlines()
        sc = [score_headline(h["title"]) for h in heads]
        nz = [x for x in sc if x != 0]
        lex = sum(nz) / len(nz) if nz else 0.0
        composite = composite + lex if out["sources"] else lex
        if out["sources"]:
            composite /= 2
        out["lexicon_score"] = round(lex, 3)
        out["sources"].append("v4_lexicon")
        out.setdefault("headlines", [h["title"] for h in heads[:6]])
    except Exception as e:  # noqa: BLE001
        out["lexicon_error"] = str(e)[:200]
    out["composite"] = round(float(np.clip(composite, -1, 1)), 3)
    out["shock_intensity"] = round(shock, 3)
    veto = bool(direction != 0 and shock >= 0.6 and np.sign(out["composite"]) == -direction
                and abs(out["composite"]) >= 0.25)
    out["veto"] = veto
    out["note"] = ("strong news shock against the model's direction -> trade vetoed" if veto else
                   "overlay informational; not yet validated on history")
    return out
