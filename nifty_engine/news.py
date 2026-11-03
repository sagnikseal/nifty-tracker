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
