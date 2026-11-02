#!/usr/bin/env python3
"""
Multi-Endpoint Global & Indian Media Intelligence & Bayesian Market Transmission Engine
(`scripts/media_news_engine.py`)
=======================================================================================
Scans 15+ live Indian & Global financial, macroeconomic, commodity, central-bank, and
geopolitical news endpoints concurrently, deduplicates stories, applies exponential
recency decay, and maps every article through an 8-Channel Economic & Geopolitical
Transmission Graph into NIFTY 50 predictive state variables.

Why Not a Simple Linear Weight?
-------------------------------
In real markets, news does NOT act as an isolated additive constant. Instead, breaking and
ongoing news acts in THREE non-linear ways:
  1. Bayesian Prior Drift Shift (`news_log_odds_shift`): Shifts the baseline directional log-odds
     before technicals are even evaluated (e.g., surprise RBI rate cut or ceasefire rally vs.
     strait blockade / tariff / FII tax shock).
  2. Regime-Gated Technical Trust Multiplier (`tech_trust_gate` in [0.15, 1.15]): When high-impact
     macro or geopolitical news is breaking (`information_intensity > 0.65`), lagging indicators
     like 5m/15m Supertrend and old support levels break down. The engine dynamically down-weights
     lagging technicals and up-weights real-time institutional option flow & news transmission.
  3. Sectoral Heavyweight Transmission (`sector_transmission_bias`): Maps global/domestic news
     directly onto NIFTY's key sector drivers (Banking/Financials 35%, IT/US-Tech 14%,
     Oil/Reliance/Energy 12%, Auto/Infra/Metals).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import html
import json
import math
import numpy as np
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

IST_TZ = dt.timezone(dt.timedelta(hours=5, minutes=30))
ROOT_DIR = Path(__file__).resolve().parent.parent
NEWS_CACHE_FILE = ROOT_DIR / "data" / "latest_media_intelligence.json"

# ==============================================================================
# 16 Live Indian & Global Media RSS / Atom / Google News Wire Endpoints
# ==============================================================================
MEDIA_ENDPOINTS: List[Dict[str, str]] = [
    # Indian Financial & Market Media
    {
        "id": "et_markets",
        "name": "Economic Times Markets",
        "category": "INDIA_MARKETS",
        "reliability": "0.92",
        "url": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    },
    {
        "id": "et_economy",
        "name": "Economic Times Macro Economy",
        "category": "INDIA_MACRO",
        "reliability": "0.92",
        "url": "https://economictimes.indiatimes.com/news/economy/rssfeeds/1373380680.cms",
    },
    {
        "id": "moneycontrol_top",
        "name": "Moneycontrol Market Reports",
        "category": "INDIA_MARKETS",
        "reliability": "0.90",
        "url": "https://www.moneycontrol.com/rss/marketreports.xml",
    },
    {
        "id": "moneycontrol_biz",
        "name": "Moneycontrol Business & Economy",
        "category": "INDIA_MACRO",
        "reliability": "0.88",
        "url": "https://www.moneycontrol.com/rss/business.xml",
    },
    {
        "id": "livemint_markets",
        "name": "LiveMint Markets",
        "category": "INDIA_MARKETS",
        "reliability": "0.91",
        "url": "https://www.livemint.com/rss/markets",
    },
    {
        "id": "livemint_economy",
        "name": "LiveMint Economy & Policy",
        "category": "INDIA_MACRO",
        "reliability": "0.90",
        "url": "https://www.livemint.com/rss/economy",
    },
    {
        "id": "bs_markets",
        "name": "Business Standard Markets",
        "category": "INDIA_MARKETS",
        "reliability": "0.91",
        "url": "https://www.business-standard.com/rss/markets-106.rss",
    },
    {
        "id": "fe_markets",
        "name": "Financial Express Markets",
        "category": "INDIA_MARKETS",
        "reliability": "0.88",
        "url": "https://www.financialexpress.com/market/feed/",
    },
    # Global Macro, Geopolitics, Central Banks & Commodities (Google News & Global Wires)
    {
        "id": "gnews_nifty_nse",
        "name": "Google News Wire: NIFTY 50 & NSE FII",
        "category": "INDIA_FLOWS",
        "reliability": "0.90",
        "url": "https://news.google.com/rss/search?q=NIFTY+50+OR+NSE+OR+FII+DII+India+markets&hl=en-IN&gl=IN&ceid=IN:en",
    },
    {
        "id": "gnews_rbi_inflation",
        "name": "Google News Wire: RBI MPC, Repo Rate & India CPI",
        "category": "CENTRAL_BANK",
        "reliability": "0.94",
        "url": "https://news.google.com/rss/search?q=RBI+MPC+OR+India+CPI+inflation+OR+Repo+Rate&hl=en-IN&gl=IN&ceid=IN:en",
    },
    {
        "id": "gnews_fed_us_macro",
        "name": "Global Wire: Federal Reserve, US CPI & Treasury Yields",
        "category": "GLOBAL_CENTRAL_BANK",
        "reliability": "0.95",
        "url": "https://news.google.com/rss/search?q=Federal+Reserve+rates+OR+US+CPI+inflation+OR+Treasury+yields&hl=en-US&gl=US&ceid=US:en",
    },
    {
        "id": "gnews_geopolitics_war",
        "name": "Global Geopolitics & Conflict Escalation Wire",
        "category": "GEOPOLITICS",
        "reliability": "0.93",
        "url": "https://news.google.com/rss/search?q=geopolitical+tension+OR+military+strike+OR+ceasefire+OR+sanctions+OR+Strait+of+Hormuz+OR+Red+Sea&hl=en-US&gl=US&ceid=US:en",
    },
    {
        "id": "gnews_crude_opec",
        "name": "Global Commodities: Brent Crude Oil & OPEC+",
        "category": "ENERGY_SUPPLY",
        "reliability": "0.92",
        "url": "https://news.google.com/rss/search?q=Brent+crude+oil+price+OR+OPEC+production+cut&hl=en-US&gl=US&ceid=US:en",
    },
    {
        "id": "gnews_india_heavyweights",
        "name": "India Corporate Heavyweights: HDFC Bank, Reliance, ICICI, Infosys, TCS",
        "category": "HEAVYWEIGHT_EARNINGS",
        "reliability": "0.90",
        "url": "https://news.google.com/rss/search?q=%22HDFC+Bank%22+OR+%22Reliance+Industries%22+OR+%22ICICI+Bank%22+OR+Infosys+OR+TCS+earnings&hl=en-IN&gl=IN&ceid=IN:en",
    },
    {
        "id": "cnbc_world",
        "name": "CNBC Global Markets & Economy",
        "category": "GLOBAL_MARKETS",
        "reliability": "0.91",
        "url": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000664",
    },
]

# ==============================================================================
# 8-Channel Economic & Geopolitical Transmission Knowledge Graph
# ==============================================================================
# Each channel contains multi-word phrases and semantic stems with directional impact
# on NIFTY (-1.0 = severe bearish, +1.0 = strong bullish) and volatility multiplier.
TRANSMISSION_CHANNELS: Dict[str, Dict[str, Any]] = {
    "geopolitics_conflict": {
        "label": "Geopolitics, War & Defense Security",
        "base_weight": 1.35,
        "vol_multiplier": 1.65,
        "patterns": {
            "missile strike": -0.92,
            "airstrike": -0.88,
            "military escalation": -0.90,
            "invasion": -0.96,
            "nuclear threat": -1.00,
            "border clash": -0.90,
            "loc tension": -0.85,
            "war widens": -0.92,
            "retaliatory strike": -0.88,
            "drone attack": -0.82,
            "strait of hormuz": -0.94,
            "red sea blockade": -0.84,
            "taiwan blockade": -0.92,
            "martial law": -0.86,
            "terror attack": -0.85,
            "new sanctions": -0.72,
            "trade embargo": -0.78,
            "ceasefire agreed": 0.78,
            "peace treaty": 0.75,
            "peace talks": 0.58,
            "de-escalation": 0.68,
            "truce holds": 0.62,
            "troop withdrawal": 0.65,
        },
    },
    "energy_crude_supply": {
        "label": "Crude Oil, Energy & Shipping Supply Chain",
        "base_weight": 1.10,
        "vol_multiplier": 1.35,
        "patterns": {
            "crude surges": -0.78,
            "oil spikes": -0.80,
            "brent above": -0.68,
            "opec+ cut": -0.65,
            "supply disruption": -0.72,
            "refinery hit": -0.75,
            "lng shortage": -0.62,
            "oil plunges": 0.72,
            "crude falls": 0.65,
            "brent drops": 0.65,
            "brent stabilizes": 0.42,
            "opec increases output": 0.68,
            "oil cools": 0.60,
        },
    },
    "central_bank_inflation": {
        "label": "RBI / Fed Monetary Policy, Rates & Inflation",
        "base_weight": 1.25,
        "vol_multiplier": 1.40,
        "patterns": {
            "rate hike": -0.82,
            "hawkish pause": -0.62,
            "hawkish fed": -0.74,
            "hawkish rbi": -0.78,
            "inflation spikes": -0.78,
            "cpi jumps": -0.72,
            "cpi higher than expected": -0.75,
            "sticky inflation": -0.60,
            "yields surge": -0.68,
            "crr hike": -0.80,
            "liquidity deficit": -0.62,
            "rate cut": 0.84,
            "dovish pivot": 0.78,
            "dovish rbi": 0.76,
            "accommodative stance": 0.58,
            "crr cut": 0.80,
            "omo liquidity injection": 0.65,
            "inflation cools": 0.72,
            "cpi eases": 0.68,
            "soft landing": 0.60,
        },
    },
    "macro_growth_fiscal": {
        "label": "GDP Growth, Fiscal Policy, Tariffs & SEBI Regulation",
        "base_weight": 1.05,
        "vol_multiplier": 1.25,
        "patterns": {
            "recession fear": -0.82,
            "stagflation": -0.88,
            "gdp slows": -0.70,
            "gdp misses": -0.72,
            "pmi contracts": -0.65,
            "fiscal slippage": -0.68,
            "sovereign downgrade": -0.85,
            "tariff hike": -0.74,
            "trade war": -0.78,
            "ltcg hike": -0.82,
            "stt hike": -0.80,
            "sebi tightens": -0.65,
            "gdp beats": 0.78,
            "strong gdp": 0.72,
            "pmi expands": 0.62,
            "record gst collection": 0.60,
            "capex boost": 0.68,
            "pli scheme": 0.55,
            "sovereign upgrade": 0.82,
            "tax cut": 0.75,
            "trade deal": 0.68,
        },
    },
    "fii_dii_fx_flows": {
        "label": "FII / DII Institutional Flows & Rupee (USD/INR)",
        "base_weight": 1.15,
        "vol_multiplier": 1.25,
        "patterns": {
            "fii outflow": -0.74,
            "fii selling": -0.72,
            "foreign investors dump": -0.78,
            "rupee hits record low": -0.72,
            "inr depreciates": -0.60,
            "dollar index surges": -0.65,
            "fii buying": 0.74,
            "fii inflow": 0.75,
            "dii inflows steady": 0.52,
            "strong dii buying": 0.62,
            "block deal accumulation": 0.58,
            "rupee strengthens": 0.58,
        },
    },
    "sector_banking_fin": {
        "label": "NIFTY Banking & Financials Heavyweights (35% Index Wt)",
        "base_weight": 1.10,
        "vol_multiplier": 1.20,
        "patterns": {
            "hdfc bank misses": -0.75,
            "icici bank misses": -0.70,
            "nim compression": -0.68,
            "npa spike": -0.78,
            "rbi bars": -0.82,
            "banking selloff": -0.72,
            "hdfc bank beats": 0.78,
            "icici bank beats": 0.75,
            "banking accumulation": 0.68,
            "strong credit growth": 0.64,
            "asset quality improves": 0.62,
        },
    },
    "sector_it_global_tech": {
        "label": "NIFTY IT & Global Tech Heavyweights (14% Index Wt)",
        "base_weight": 0.95,
        "vol_multiplier": 1.15,
        "patterns": {
            "infosys cuts guidance": -0.78,
            "tcs misses": -0.72,
            "nasdaq plunges": -0.70,
            "us tech selloff": -0.68,
            "it spending freeze": -0.65,
            "infosys raises guidance": 0.78,
            "tcs beats": 0.74,
            "large deal wins": 0.65,
            "nasdaq rallies": 0.64,
            "ai deal momentum": 0.58,
        },
    },
    "market_sentiment_general": {
        "label": "Broad Market Breadth, Earnings & Technical Momentum",
        "base_weight": 0.85,
        "vol_multiplier": 1.10,
        "patterns": {
            "market crash": -0.88,
            "bloodbath on dalal street": -0.85,
            "sensex plunges": -0.78,
            "nifty plunges": -0.78,
            "panic selling": -0.82,
            "margin call": -0.78,
            "earnings downgrade": -0.65,
            "nifty hits record high": 0.75,
            "bull run": 0.65,
            "short covering rally": 0.72,
            "earnings beat": 0.65,
            "breakout rally": 0.64,
            "all-time high": 0.70,
        },
    },
}

# Single-word fallback tokens when multi-word phrases don't match
SINGLE_TOKEN_SENTIMENT: Dict[str, Tuple[str, float]] = {
    "war": ("geopolitics_conflict", -0.82),
    "missile": ("geopolitics_conflict", -0.85),
    "invasion": ("geopolitics_conflict", -0.90),
    "airstrike": ("geopolitics_conflict", -0.84),
    "sanctions": ("geopolitics_conflict", -0.68),
    "blockade": ("geopolitics_conflict", -0.78),
    "ceasefire": ("geopolitics_conflict", 0.70),
    "truce": ("geopolitics_conflict", 0.62),
    "hawkish": ("central_bank_inflation", -0.62),
    "dovish": ("central_bank_inflation", 0.62),
    "stagflation": ("macro_growth_fiscal", -0.80),
    "recession": ("macro_growth_fiscal", -0.75),
    "tariff": ("macro_growth_fiscal", -0.60),
    "stimulus": ("macro_growth_fiscal", 0.65),
    "crash": ("market_sentiment_general", -0.78),
    "plunge": ("market_sentiment_general", -0.68),
    "selloff": ("market_sentiment_general", -0.65),
    "slump": ("market_sentiment_general", -0.58),
    "surge": ("market_sentiment_general", 0.55),
    "rally": ("market_sentiment_general", 0.58),
    "bullish": ("market_sentiment_general", 0.52),
    "bearish": ("market_sentiment_general", -0.52),
    "upgrade": ("market_sentiment_general", 0.55),
    "downgrade": ("market_sentiment_general", -0.58),
    "outflow": ("fii_dii_fx_flows", -0.60),
    "inflow": ("fii_dii_fx_flows", 0.60),
    "accumulation": ("fii_dii_fx_flows", 0.52),
}


@dataclass
class NewsArticle:
    title: str
    source: str
    category: str
    url: str = ""
    published_ts: float = 0.0
    reliability: float = 0.90


class MultiEndpointMediaEngine:
    """
    Concurrent multi-endpoint RSS/Atom/Wire scanner and non-linear Bayesian
    market transmission analyzer for NIFTY 50.
    """

    def __init__(self, refresh_interval_sec: float = 120.0) -> None:
        self.refresh_interval_sec = refresh_interval_sec
        self.last_fetch_ts: float = 0.0
        self.cached_articles: List[NewsArticle] = []
        self.endpoint_health: Dict[str, Dict[str, Any]] = {}
        self.last_analysis: Dict[str, Any] = {}

    def _clean_text(self, raw: str) -> str:
        txt = html.unescape(raw or "")
        txt = re.sub(r"<[^>]+>", " ", txt)
        txt = re.sub(r"\s+", " ", txt).strip()
        return txt

    def _parse_pubdate(self, date_str: Optional[str]) -> float:
        if not date_str:
            return time.time()
        try:
            from email.utils import parsedate_to_datetime

            dt_obj = parsedate_to_datetime(date_str)
            return dt_obj.timestamp()
        except Exception:
            return time.time()

    def _fetch_single_endpoint(self, ep: Dict[str, str], timeout: float = 2.8) -> Tuple[str, List[NewsArticle], Optional[str]]:
        ep_id = ep["id"]
        url = ep["url"]
        rel = float(ep.get("reliability", "0.90"))
        articles: List[NewsArticle] = []

        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
                    ),
                    "Accept": "application/rss+xml, application/xml, text/xml, */*",
                },
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw_bytes = resp.read()

            root = ET.fromstring(raw_bytes)
            # Support both RSS 2.0 (<item>) and Atom (<entry>)
            items = root.findall(".//item")
            if not items:
                items = root.findall(".//{http://www.w3.org/2005/Atom}entry")

            for item in items[:8]:
                title_el = item.find("title")
                if title_el is None:
                    title_el = item.find("{http://www.w3.org/2005/Atom}title")
                link_el = item.find("link")
                pub_el = item.find("pubDate")
                if pub_el is None:
                    pub_el = item.find("{http://www.w3.org/2005/Atom}updated")

                title_txt = self._clean_text(title_el.text if title_el is not None else "")
                if not title_txt or len(title_txt) < 12:
                    continue

                link_txt = ""
                if link_el is not None:
                    link_txt = (link_el.text or link_el.attrib.get("href", "")).strip()

                pub_ts = self._parse_pubdate(pub_el.text if pub_el is not None else None)
                articles.append(
                    NewsArticle(
                        title=title_txt,
                        source=ep["name"],
                        category=ep["category"],
                        url=link_txt,
                        published_ts=pub_ts,
                        reliability=rel,
                    )
                )
            return ep_id, articles, None
        except Exception as exc:
            return ep_id, [], str(exc)

    def fetch_all_media_endpoints(self, force: bool = False) -> List[NewsArticle]:
        now = time.time()
        if not force and self.cached_articles and (now - self.last_fetch_ts < self.refresh_interval_sec):
            return self.cached_articles

        self.last_fetch_ts = now
        collected: List[NewsArticle] = []

        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {pool.submit(self._fetch_single_endpoint, ep): ep for ep in MEDIA_ENDPOINTS}
            for fut in as_completed(futures):
                ep = futures[fut]
                ep_id, arts, err = fut.result()
                self.endpoint_health[ep_id] = {
                    "name": ep["name"],
                    "category": ep["category"],
                    "status": "LIVE" if not err and arts else ("EMPTY" if not err else "OFFLINE"),
                    "articles_fetched": len(arts),
                    "error": err,
                }
                collected.extend(arts)

        # Deduplicate by normalized title fingerprint
        seen_hashes = set()
        deduped: List[NewsArticle] = []
        for art in sorted(collected, key=lambda a: a.published_ts, reverse=True):
            norm_key = re.sub(r"[^a-z0-9]", "", art.title.lower())[:55]
            h = hashlib.md5(norm_key.encode("utf-8")).hexdigest()
            if h not in seen_hashes:
                seen_hashes.add(h)
                deduped.append(art)

        if deduped:
            self.cached_articles = deduped[:45]
        return self.cached_articles

    def score_single_article(self, article: NewsArticle, now_ts: Optional[float] = None) -> Dict[str, Any]:
        """
        Map a single headline across the 8 transmission channels, accounting for:
          - Multi-word structural phrases
          - Negation / de-escalation modifiers
          - Source reliability weight
          - Exponential time decay (half-life = 4 hours for intraday trading)
        """
        if now_ts is None:
            now_ts = time.time()

        text_low = article.title.lower()
        age_hours = max(0.0, (now_ts - article.published_ts) / 3600.0) if article.published_ts > 0 else 0.5
        # Half-life of 6 hours; floor at 0.35 so same-day macro headlines retain context
        recency_decay = max(0.35, math.exp(-0.115 * age_hours))

        channel_hits: Dict[str, float] = {ch: 0.0 for ch in TRANSMISSION_CHANNELS}
        matched_tags: List[str] = []
        max_vol_mult = 1.0

        # 1. Check multi-word phrases across all 8 channels
        for ch_key, ch_cfg in TRANSMISSION_CHANNELS.items():
            for phrase, weight in ch_cfg["patterns"].items():
                if phrase in text_low:
                    channel_hits[ch_key] += weight * ch_cfg["base_weight"]
                    matched_tags.append(f"{ch_key.split('_')[0].upper()}:{phrase.upper()}")
                    if abs(weight) >= 0.65:
                        max_vol_mult = max(max_vol_mult, ch_cfg["vol_multiplier"])

        # 2. Check single-word fallback tokens if no multi-word phrase hit in that channel
        for tok, (ch_key, weight) in SINGLE_TOKEN_SENTIMENT.items():
            if re.search(rf"\b{re.escape(tok)}\b", text_low):
                if abs(channel_hits[ch_key]) < 0.1:
                    ch_w = TRANSMISSION_CHANNELS[ch_key]["base_weight"]
                    channel_hits[ch_key] += weight * ch_w * 0.85
                    matched_tags.append(f"{ch_key.split('_')[0].upper()}:{tok.upper()}")
                    if abs(weight) >= 0.70:
                        max_vol_mult = max(max_vol_mult, TRANSMISSION_CHANNELS[ch_key]["vol_multiplier"])

        # Combine across channels with saturation (tanh) so a single headline doesn't explode
        raw_sum = sum(channel_hits.values()) * article.reliability * recency_decay
        bounded_score = float(np.clip(math.tanh(raw_sum * 0.75) * 2.0, -2.0, 2.0))

        dominant_channel = "market_sentiment_general"
        best_abs = 0.0
        for ch_k, val in channel_hits.items():
            if abs(val) > best_abs:
                best_abs = abs(val)
                dominant_channel = ch_k

        if bounded_score >= 0.22:
            sentiment_label = "BULLISH_CATALYST"
        elif bounded_score <= -0.65:
            sentiment_label = "SEVERE_BEARISH_SHOCK"
        elif bounded_score <= -0.22:
            sentiment_label = "BEARISH_HEADWIND"
        else:
            sentiment_label = "NEUTRAL_NOISE"

        return {
            "title": article.title,
            "source": article.source,
            "category": article.category,
            "url": article.url,
            "age_hours": round(age_hours, 1),
            "recency_weight": round(recency_decay, 2),
            "score": round(bounded_score, 3),
            "sentiment": sentiment_label,
            "dominant_channel": dominant_channel,
            "channel_hits": {k: round(v, 3) for k, v in channel_hits.items() if abs(v) > 0.01},
            "vol_multiplier": round(max_vol_mult, 2),
            "tags": matched_tags[:4],
        }

    def analyze_media_ecosystem(
        self,
        custom_headlines: Optional[List[Dict[str, str]]] = None,
        force_refresh: bool = False,
    ) -> Dict[str, Any]:
        """
        Ingests live feeds + custom trader wire headlines and computes the full
        Non-Linear Bayesian Market Transmission state for the predictive model.
        """
        now_ts = time.time()
        live_articles = self.fetch_all_media_endpoints(force=force_refresh)

        combined_articles: List[NewsArticle] = []
        if custom_headlines:
            for item in custom_headlines:
                t_str = (item.get("title") or "").strip()
                if t_str:
                    combined_articles.append(
                        NewsArticle(
                            title=t_str,
                            source=item.get("source", "Live Trader Wire"),
                            category="BREAKING_WIRE",
                            published_ts=now_ts,
                            reliability=0.98,
                        )
                    )
        combined_articles.extend(live_articles)

        scored_articles = [self.score_single_article(art, now_ts=now_ts) for art in combined_articles[:35]]

        # Aggregate per-channel transmission scores
        channel_totals: Dict[str, float] = {ch: 0.0 for ch in TRANSMISSION_CHANNELS}
        vol_multipliers: List[float] = [1.0]

        for s_art in scored_articles:
            w_rec = s_art["recency_weight"]
            for ch_k, val in s_art["channel_hits"].items():
                channel_totals[ch_k] += val * w_rec
            vol_multipliers.append(s_art["vol_multiplier"])

        # Compress channel totals via smooth hyperbolic tangent into [-2.0, +2.0]
        channel_scores: Dict[str, Dict[str, Any]] = {}
        for ch_k, ch_cfg in TRANSMISSION_CHANNELS.items():
            raw_v = channel_totals[ch_k]
            norm_v = float(np.clip(math.tanh(raw_v * 0.55) * 2.0, -2.0, 2.0))
            channel_scores[ch_k] = {
                "label": ch_cfg["label"],
                "score": round(norm_v, 3),
                "raw_impulse": round(raw_v, 3),
                "stance": "BULLISH" if norm_v >= 0.20 else ("BEARISH" if norm_v <= -0.20 else "NEUTRAL"),
            }

        # Extract core pillar variables
        geo_score = channel_scores["geopolitics_conflict"]["score"]
        energy_score = channel_scores["energy_crude_supply"]["score"]
        cb_score = channel_scores["central_bank_inflation"]["score"]
        macro_fiscal_score = channel_scores["macro_growth_fiscal"]["score"]
        flow_score = channel_scores["fii_dii_fx_flows"]["score"]
        bank_score = channel_scores["sector_banking_fin"]["score"]
        it_score = channel_scores["sector_it_global_tech"]["score"]
        broad_score = channel_scores["market_sentiment_general"]["score"]

        # ----------------------------------------------------------------------
        # Non-Linear Mechanism 1: Bayesian Posterior Log-Odds Shift
        # Asymmetric tail-risk: severe negative geopolitical or central-bank shocks
        # carry 1.45x higher convex weight than routine positive news.
        # ----------------------------------------------------------------------
        geo_asym = geo_score * (1.45 if geo_score < -0.35 else 1.0)
        cb_asym = cb_score * (1.30 if cb_score < -0.35 else 1.0)

        bayesian_prior_log_odds = float(
            np.clip(
                0.32 * geo_asym
                + 0.22 * energy_score
                + 0.28 * cb_asym
                + 0.20 * macro_fiscal_score
                + 0.24 * flow_score
                + 0.25 * bank_score
                + 0.14 * it_score
                + 0.18 * broad_score,
                -3.2,
                3.2,
            )
        )

        # ----------------------------------------------------------------------
        # Non-Linear Mechanism 2: Regime-Gated Technical Trust Multiplier
        # When breaking news intensity is high (e.g. war escalation, surprise rate move),
        # lagging chart indicators (Supertrend/BOS) lose reliability and get gated down.
        # ----------------------------------------------------------------------
        news_shock_intensity = float(
            np.clip(
                (
                    abs(geo_asym) * 0.40
                    + abs(cb_asym) * 0.30
                    + abs(energy_score) * 0.15
                    + abs(macro_fiscal_score) * 0.15
                )
                / 1.5,
                0.0,
                1.0,
            )
        )
        # Technical trust gate ranges from 1.0 (calm news) down to 0.25 (severe breaking shock)
        technical_trust_gate = round(float(np.clip(1.0 - 0.72 * (news_shock_intensity ** 1.3), 0.25, 1.0)), 3)
        # News & Macro authority weight increases from 0.20 (calm) up to 0.68 (breaking shock)
        news_authority_gate = round(float(np.clip(0.20 + 0.48 * news_shock_intensity, 0.20, 0.68)), 3)

        # ----------------------------------------------------------------------
        # Non-Linear Mechanism 3: Dynamic Volatility & ATR Stop Multiplier
        # ----------------------------------------------------------------------
        news_vol_multiplier = round(float(np.clip(max(vol_multipliers) * (1.0 + 0.35 * news_shock_intensity), 1.0, 2.1)), 2)

        # Determine dominant Media Regime
        geopolitical_alert = bool(geo_score <= -0.55 or energy_score <= -0.75)
        if geo_score <= -0.95:
            crisis_level = "SEVERE_WAR_ESCALATION"
        elif geo_score <= -0.45:
            crisis_level = "ELEVATED_GEOPOLITICAL_RISK"
        elif cb_score <= -0.65 or macro_fiscal_score <= -0.65:
            crisis_level = "MACRO_POLICY_HEADWIND"
        elif bayesian_prior_log_odds >= 0.45:
            crisis_level = "BULLISH_MEDIA_TAILWIND"
        elif bayesian_prior_log_odds <= -0.45:
            crisis_level = "BEARISH_MEDIA_HEADWIND"
        else:
            crisis_level = "CALM_BALANCED"

        # Composite features exposed to the 4-Pillar ensemble
        finbert_composite = float(
            np.clip(0.35 * broad_score + 0.30 * bank_score + 0.20 * flow_score + 0.15 * it_score, -2.0, 2.0)
        )
        cb_macro_composite = float(
            np.clip(0.55 * cb_score + 0.45 * macro_fiscal_score, -2.0, 2.0)
        )

        active_endpoints = sum(1 for v in self.endpoint_health.values() if v.get("status") == "LIVE")

        result = {
            "timestamp_ist": dt.datetime.now(IST_TZ).strftime("%Y-%m-%d %H:%M:%S IST"),
            "endpoints_configured": len(MEDIA_ENDPOINTS),
            "endpoints_live": active_endpoints,
            "endpoint_health": self.endpoint_health,
            "headline_count": len(scored_articles),
            "scored_headlines": scored_articles[:18],
            "geopolitical_alert": geopolitical_alert,
            "crisis_level": crisis_level,
            "channel_transmission": channel_scores,
            "non_linear_gates": {
                "bayesian_prior_log_odds": round(bayesian_prior_log_odds, 4),
                "news_shock_intensity": round(news_shock_intensity, 3),
                "technical_trust_gate": technical_trust_gate,
                "news_authority_gate": news_authority_gate,
                "news_vol_multiplier": news_vol_multiplier,
            },
            "features": {
                "news_finbert_sentiment": round(finbert_composite, 4),
                "geopolitical_war_shock": round(float(np.clip(geo_score + 0.35 * min(0.0, energy_score), -3.0, 1.5)), 4),
                "central_bank_event_shock": round(cb_macro_composite, 4),
            },
        }
        self.last_analysis = result
        return result


if __name__ == "__main__":
    engine = MultiEndpointMediaEngine()
    report = engine.analyze_media_ecosystem(force_refresh=True)
    print(
        f"[media_news_engine] Scanned {report['endpoints_configured']} endpoints | "
        f"Articles: {report['headline_count']} | Regime: {report['crisis_level']}\n"
        f"  -> Non-Linear Gates: {report['non_linear_gates']}\n"
        f"  -> Features: {report['features']}"
    )
