"""News capture -> local-LLM scoring -> timestamped features.

Pipeline
  1. capture(): headlines from free RSS feeds, the Upstox News API (Analytics Token) and GDELT,
     de-duplicated, each stamped with `first_seen` = when this machine first saw it. Features only
     ever use `first_seen`, never the publisher's timestamp, so nothing is used before it existed.
  2. score_pending(): a local LLM (Ollama) reads each headline like an India-equity strategist and
     returns structured fields (India relevance, NIFTY direction, magnitude, surprise, horizon,
     transmission channel, heavyweights affected). The keyword lexicon is stored alongside.
  3. window_features(): aggregates for any decision window, e.g. 15:30 yesterday -> 09:10 today.

Look-ahead guard: an LLM may "remember" what happened after old headlines. Headlines first seen
on/before `llm_knowledge_cutoff` therefore use the lexicon score in every feature and test; the
LLM score is kept only for inspection.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import re
import time
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from ..calendar import IST, now_ist
from ..config import DATA_DIR, EngineConfig
from ..news import fetch_headlines, score_headline
from .llm import LocalLLM, clamp, pick

log = logging.getLogger(__name__)
STORE = DATA_DIR / "news"
RAW_DIR = STORE / "headlines"
SCORES = STORE / "scores.jsonl"

CHANNELS = ["rates", "inflation", "fx", "crude_energy", "geopolitics", "flows", "earnings",
            "policy_regulation", "global_equities", "commodities", "domestic_macro", "other"]
HORIZONS = ["intraday", "days", "weeks", "long_term"]
INDIA_WORDS = re.compile(r"\b(india|indian|nifty|sensex|rbi|rupee|inr|sebi|dalal|nse|bse|fii|fpi|modi|"
                         r"mumbai|delhi|budget)\b", re.I)

ITEM_SCHEMA = {
    "type": "object",
    "properties": {
        "id": {"type": "string"},
        "india_relevance": {"type": "number"},
        "nifty_direction": {"type": "number"},
        "magnitude": {"type": "number"},
        "surprise": {"type": "number"},
        "horizon": {"type": "string", "enum": HORIZONS},
        "channel": {"type": "string", "enum": CHANNELS},
        "heavyweights": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "number"},
        "reason": {"type": "string"},
    },
    "required": ["id", "india_relevance", "nifty_direction", "magnitude", "surprise", "horizon", "channel",
                 "confidence", "reason"],
}
BATCH_SCHEMA = {"type": "object", "properties": {"items": {"type": "array", "items": ITEM_SCHEMA}},
                "required": ["items"]}

SYSTEM = """You are a senior India equity strategist on an index-derivatives desk. For each news headline, judge how it is likely to move the NIFTY 50 index, using only the headline text. Think about the transmission channel (interest rates, inflation, the rupee, crude oil, geopolitics, foreign/domestic fund flows, company earnings, policy and regulation, global equities, commodities, domestic macro data).

Return one item per headline, with the same id:
- india_relevance: 0 (irrelevant to Indian markets) to 1 (directly about NIFTY, RBI, Indian macro or index heavyweights)
- nifty_direction: -1 (clearly negative for NIFTY) to +1 (clearly positive); 0 if neutral or unclear
- magnitude: 0.1 routine news, 0.5 notable, 1.0 crisis or regime-changing
- surprise: 0 (expected, scheduled or old news) to 1 (completely unexpected)
- horizon: intraday, days, weeks or long_term
- channel: the main transmission channel
- heavyweights: NSE symbols of NIFTY heavyweights directly affected, from this list only: {heavy}
- confidence: 0 to 1, how sure you are of the direction
- reason: at most 20 words
Be conservative: most headlines are noise with small magnitude. Do not guess facts that are not in the headline."""


def _norm(title: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", title.lower()).strip()


def headline_id(title: str) -> str:
    return hashlib.sha1(" ".join(_norm(title).split()).encode()).hexdigest()[:16]


def _mask_dates(text: str) -> str:
    """Hide explicit years so the model reasons from content, not from remembering the date."""
    return re.sub(r"\b(19|20)\d{2}\b", "[year]", text)


def _read_jsonl(path) -> List[Dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def _append_jsonl(path, rows: Iterable[Dict]) -> int:
    rows = list(rows)
    if not rows:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, default=str) + "\n")
    return len(rows)


def load_headlines(start: Optional[dt.date] = None, end: Optional[dt.date] = None) -> pd.DataFrame:
    rows = []
    if RAW_DIR.exists():
        for f in sorted(RAW_DIR.glob("*.jsonl")):
            d = dt.date.fromisoformat(f.stem)
            if (start and d < start) or (end and d > end):
                continue
            rows += _read_jsonl(f)
    if not rows:
        return pd.DataFrame(columns=["id", "title", "source", "first_seen"])
    df = pd.DataFrame(rows).drop_duplicates("id", keep="first")
    df["first_seen"] = pd.to_datetime(df["first_seen"], utc=True).dt.tz_convert(IST)
    return df.sort_values("first_seen").reset_index(drop=True)


def _known_ids(days: int = 4) -> set:
    since = now_ist().date() - dt.timedelta(days=days)
    return set(load_headlines(start=since)["id"]) if RAW_DIR.exists() else set()


def store_items(items: List[Dict], first_seen: Optional[dt.datetime] = None) -> int:
    """Persist unseen headlines. `first_seen` defaults to now (live capture); backfills pass the
    source's own discovery time (GDELT `seendate`)."""
    known = _known_ids()
    by_day: Dict[str, List[Dict]] = {}
    for it in items:
        hid = headline_id(it["title"])
        if hid in known:
            continue
        known.add(hid)
        fs = it.get("first_seen") or first_seen or now_ist()
        pub = it.get("ts")
        by_day.setdefault(fs.astimezone(IST).date().isoformat(), []).append({
            "id": hid, "title": it["title"], "summary": it.get("summary", ""), "source": it.get("source", ""),
            "url": it.get("url"), "published": pub.isoformat() if hasattr(pub, "isoformat") else pub,
            "first_seen": fs.astimezone(IST).isoformat()})
    return sum(_append_jsonl(RAW_DIR / f"{d}.jsonl", rows) for d, rows in by_day.items())


def capture(cfg: EngineConfig) -> Dict[str, int]:
    """One live capture pass over every free source."""
    stats = {}
    items = fetch_headlines()
    stats["rss"] = len(items)
    try:
        from ..data.instruments import HEAVYWEIGHTS, equity_keys
        from ..data.providers import UpstoxProvider
        up = UpstoxProvider()
        if up.available:
            keys = list(equity_keys(cfg.extra.get("breadth_symbols") or HEAVYWEIGHTS).values())
            un = up.news(keys)
            items += un
            stats["upstox"] = len(un)
    except Exception as e:  # noqa: BLE001
        log.warning("upstox news failed: %s", str(e)[:150])
    try:
        from ..data.providers import GdeltProvider
        end = now_ist()
        for q in _gdelt_queries():
            g = GdeltProvider().articles(q, end - dt.timedelta(hours=3), end)
            items += g
            stats["gdelt"] = stats.get("gdelt", 0) + len(g)
    except Exception as e:  # noqa: BLE001
        log.warning("gdelt failed: %s", str(e)[:150])
    stats["new"] = store_items([i for i in items if i.get("title")])
    return stats


def _gdelt_queries() -> List[str]:
    return ['(Nifty OR Sensex OR "Indian stocks" OR "Indian markets" OR RBI OR rupee OR "Dalal Street") sourcelang:english',
            '("Federal Reserve" OR "crude oil" OR "Treasury yields" OR tariffs OR geopolitical OR "Wall Street") sourcelang:english']


def backfill_gdelt(days: int = 85, sleep: float = 6.0) -> Dict[str, int]:
    """Fill the last ~3 months of pre-open news from GDELT (first_seen = GDELT's seendate)."""
    from ..data.providers import GdeltProvider
    g, n = GdeltProvider(), 0
    today = now_ist().date()
    for k in range(days, -1, -1):
        d = today - dt.timedelta(days=k)
        if d.weekday() >= 5:
            continue
        start = dt.datetime.combine(d - dt.timedelta(days=1), dt.time(15, 30), tzinfo=IST)
        end = dt.datetime.combine(d, dt.time(9, 10), tzinfo=IST)
        for q in _gdelt_queries():
            try:
                n += store_items(g.articles(q, start, end))
            except Exception as e:  # noqa: BLE001
                log.warning("gdelt %s: %s", d, str(e)[:120])
            time.sleep(sleep)
    return {"stored": n}


# ----------------------------------------------------------------------------- scoring
def load_scores() -> Dict[str, Dict]:
    return {r["id"]: r for r in _read_jsonl(SCORES)}


def lexicon_score(title: str) -> Dict:
    d = score_headline(title)
    rel = 0.6 if INDIA_WORDS.search(title) else 0.3
    return {"india_relevance": rel, "nifty_direction": d, "magnitude": 0.3 * abs(d), "surprise": 0.3,
            "horizon": "days", "channel": "other", "heavyweights": [], "confidence": 0.3 * abs(d), "reason": "lexicon"}


def _validate(it: Dict, heavy: List[str]) -> Dict:
    return {"india_relevance": clamp(it.get("india_relevance"), 0, 1), "nifty_direction": clamp(it.get("nifty_direction"), -1, 1),
            "magnitude": clamp(it.get("magnitude"), 0, 1), "surprise": clamp(it.get("surprise"), 0, 1),
            "horizon": pick(it.get("horizon"), HORIZONS, "days"), "channel": pick(it.get("channel"), CHANNELS, "other"),
            "heavyweights": [h for h in (it.get("heavyweights") or []) if h in heavy][:5],
            "confidence": clamp(it.get("confidence"), 0, 1), "reason": str(it.get("reason", ""))[:200]}


def score_pending(cfg: EngineConfig, max_items: int = 600, batch: int = 8, since_days: int = 120) -> Dict[str, int]:
    from ..data.instruments import HEAVYWEIGHTS
    heads = load_headlines(start=now_ist().date() - dt.timedelta(days=since_days))
    done = load_scores()
    todo = heads[~heads["id"].isin(done)].tail(max_items)
    if todo.empty:
        return {"scored": 0, "pending": 0}
    llm = LocalLLM(cfg.llm_model, cfg.llm_url)
    use_llm = llm.available()
    system = SYSTEM.replace("{heavy}", ", ".join(HEAVYWEIGHTS))
    out, n_llm = [], 0
    recs = todo.to_dict("records")
    for i in range(0, len(recs), batch):
        chunk = recs[i:i + batch]
        llm_items: Dict[str, Dict] = {}
        if use_llm:
            user = "\n".join(f"id={r['id']}: {_mask_dates(r['title'])}" for r in chunk)
            res = llm.json_chat(system, user, BATCH_SCHEMA)
            for it in (res or {}).get("items", []):
                if it.get("id") in {r["id"] for r in chunk}:
                    llm_items[it["id"]] = _validate(it, HEAVYWEIGHTS)
        for r in chunk:
            row = {"id": r["id"], "first_seen": r["first_seen"].isoformat(), "lex": lexicon_score(r["title"])}
            if r["id"] in llm_items:
                row["llm"], row["scorer"] = llm_items[r["id"]], llm.model
                n_llm += 1
            else:
                row["scorer"] = "lexicon"
            out.append(row)
    _append_jsonl(SCORES, out)
    return {"scored": len(out), "by_llm": n_llm, "llm_available": int(use_llm)}


# ----------------------------------------------------------------------------- features
def scored_frame(cfg: EngineConfig, start: Optional[dt.date] = None, end: Optional[dt.date] = None) -> pd.DataFrame:
    """Headlines joined with their effective score (LLM after the cutoff, lexicon otherwise)."""
    heads = load_headlines(start, end)
    if heads.empty:
        return heads
    sc = load_scores()
    cutoff = pd.Timestamp(cfg.llm_knowledge_cutoff, tz=IST) + pd.Timedelta(days=1)
    rows = []
    for r in heads.to_dict("records"):
        s = sc.get(r["id"])
        if s is None:
            eff, scorer = lexicon_score(r["title"]), "lexicon"
        elif "llm" in s and r["first_seen"] >= cutoff:
            eff, scorer = s["llm"], s.get("scorer", "llm")
        else:
            eff, scorer = s["lex"], "lexicon"
        rows.append({**r, **{k: eff.get(k) for k in ("india_relevance", "nifty_direction", "magnitude", "surprise",
                                                      "horizon", "channel", "confidence", "reason")},
                     "scorer": scorer})
    return pd.DataFrame(rows)


def window_features(S: pd.DataFrame, start: dt.datetime, end: dt.datetime, prefix: str = "nw_") -> Dict[str, float]:
    w = S[(S["first_seen"] > pd.Timestamp(start)) & (S["first_seen"] <= pd.Timestamp(end))] if len(S) else S
    f: Dict[str, float] = {f"{prefix}n": float(len(w))}
    if w.empty:
        return f
    rel = w["india_relevance"].astype(float)
    wt = rel * (0.25 + w["confidence"].astype(float))
    signed = w["nifty_direction"].astype(float) * w["magnitude"].astype(float)
    f[f"{prefix}rel_n"] = float((rel >= 0.5).sum())
    f[f"{prefix}dir"] = float((wt * signed).sum() / wt.sum()) if wt.sum() > 0 else 0.0
    shock = rel * w["magnitude"].astype(float) * (0.5 + 0.5 * w["surprise"].astype(float))
    i = shock.idxmax()
    f[f"{prefix}shock"] = float(shock.max())
    f[f"{prefix}shock_dir"] = float(np.sign(w.loc[i, "nifty_direction"]))
    relevant = w[rel >= 0.5]
    f[f"{prefix}neg_share"] = float((relevant["nifty_direction"] < -0.1).mean()) if len(relevant) else 0.0
    for ch in ("rates", "crude_energy", "geopolitics", "flows", "fx", "global_equities", "earnings"):
        m = w["channel"] == ch
        f[f"{prefix}ch_{ch}"] = float((wt[m] * signed[m]).sum())
    f[f"{prefix}llm_share"] = float((w["scorer"] != "lexicon").mean())
    return f


def preopen_window(day: dt.date, prev_session: dt.date, cutoff: dt.time = dt.time(9, 10)):
    return (dt.datetime.combine(prev_session, dt.time(15, 30), tzinfo=IST), dt.datetime.combine(day, cutoff, tzinfo=IST))


def daily_news_frame(cfg: EngineConfig, sessions: Optional[pd.DatetimeIndex] = None) -> pd.DataFrame:
    """One row per NIFTY session: news aggregates from the previous close to 09:10 IST."""
    S = scored_frame(cfg)
    if S.empty:
        return pd.DataFrame()
    if sessions is None:
        from ..data import store
        nf = store.load("nifty")
        sessions = nf.index if len(nf) else pd.DatetimeIndex([])
        today = pd.Timestamp(now_ist().date())
        if len(sessions) == 0 or sessions[-1] < today:
            sessions = sessions.append(pd.DatetimeIndex([today]))
    first = S["first_seen"].min().date()
    rows = []
    for i in range(1, len(sessions)):
        d, p = sessions[i].date(), sessions[i - 1].date()
        if d < first:
            continue
        rows.append({"date": sessions[i], **window_features(S, *preopen_window(d, p))})
    return pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame()


def top_headlines(cfg: EngineConfig, start: dt.datetime, end: dt.datetime, n: int = 12) -> List[Dict]:
    S = scored_frame(cfg, start.date() - dt.timedelta(days=1), end.date())
    if S.empty:
        return []
    w = S[(S["first_seen"] > pd.Timestamp(start)) & (S["first_seen"] <= pd.Timestamp(end))].copy()
    w["imp"] = w["india_relevance"].astype(float) * (0.2 + w["magnitude"].astype(float))
    w = w.sort_values("imp", ascending=False).head(n)
    return [{"title": r["title"], "dir": round(float(r["nifty_direction"]), 2), "mag": round(float(r["magnitude"]), 2),
             "channel": r["channel"], "seen": r["first_seen"].strftime("%d %b %H:%M"), "scorer": r["scorer"]}
            for _, r in w.iterrows()]
