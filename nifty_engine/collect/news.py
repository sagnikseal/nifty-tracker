"""One news poll for the collector: every free RSS/Atom source (data/news_sources.py) plus the
Upstox News API for the heavyweights. Headlines are stored in the engine's own format
(semantic/news_llm.store_items: id, title, summary, source, url, published, first_seen) with the
same dedupe id (news_llm.headline_id), so `sync` can merge them into store/news/headlines/.

No conditional-GET state is kept in CI (each run is a fresh machine); a full fetch of all feeds is
~1-3 MB and takes a few seconds with 8 parallel connections.
"""

from __future__ import annotations

import datetime as dt
import logging
from typing import Dict, List, Optional

from ..calendar import IST
from ..semantic.news_llm import headline_id
from .layout import NEWS_FIELDS, append_jsonl, day_dir, recent_news_ids

log = logging.getLogger(__name__)


def to_records(items: List[Dict], first_seen: dt.datetime, known: Optional[set] = None) -> List[Dict]:
    """Raw items ({title, url, ts, summary, source}) -> store records, minus already-known ids."""
    known = set(known or ())
    out = []
    fs = first_seen.astimezone(IST).isoformat()
    for it in items:
        title = " ".join(str(it.get("title") or "").split())
        if not title:
            continue
        hid = headline_id(title)
        if not hid or hid in known:
            continue
        known.add(hid)
        pub = it.get("ts") or it.get("published")
        out.append({"id": hid, "title": title, "summary": (it.get("summary") or "")[:400],
                    "source": it.get("source", ""), "url": it.get("url"),
                    "published": pub.isoformat() if hasattr(pub, "isoformat") else pub,
                    "first_seen": fs})
    return [{k: r.get(k) for k in NEWS_FIELDS} for r in out]


def poll(ctx) -> Dict:
    """Fetch all sources once and append unseen headlines to days/<today>/news.jsonl in ctx.out."""
    from ..data.news_sources import NewsPoller
    now = ctx.clock()
    stats: Dict = {}
    items: List[Dict] = []
    try:
        p = NewsPoller(ctx.sources, timeout=ctx.news_timeout, session=ctx.session)
        items = p.poll(force=True)
        h = p.health()
        stats["rss_items"] = len(items)
        stats["sources_ok"] = sum(1 for s in h if s.get("ok"))
        stats["sources_failed"] = sorted(s["name"] for s in h if s.get("ok") is False)
    except Exception as e:  # noqa: BLE001
        ctx.err("rss", e)
    if ctx.upstox_ok():
        keys = list((ctx.meta or {}).get("equities", {}).values())
        if keys:
            un = ctx.guard("upstox_news", ctx.up.news, keys)
            if un:
                items += un
                stats["upstox_items"] = len(un)
    known = recent_news_ids(ctx.base, now.date(), back=4, ahead=1)
    if ctx.base != ctx.out:
        known |= recent_news_ids(ctx.out, now.date(), back=4, ahead=1)
    rows = to_records(items, now, known)
    stats["new"] = append_jsonl(day_dir(ctx.out, now.astimezone(IST).date()) / "news.jsonl", rows, key="id")
    return stats
