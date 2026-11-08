"""Fast, free news sources for the live engine, with health tracking.

Design (why it is fast and polite):
  * every source is polled on its own schedule (default 60 s) with a conditional GET
    (If-None-Match / If-Modified-Since), so an unchanged feed costs a few hundred bytes;
  * sources are fetched concurrently, each with a short timeout, in a background thread that
    can never block the trading loop;
  * a source that answers 403/429/5xx backs off exponentially (up to 30 min) instead of being
    hammered; its health is shown in the dashboard and in `sources-check`;
  * every headline is stamped with `first_seen` = when THIS machine first saw it. Features and the
    news-impact study only ever use `first_seen` (headlines whose publisher time is much older
    than first_seen are flagged as stale and excluded from the impact study).

Add your own feeds (for example a Telegram channel through an RSSHub route such as
https://rsshub.app/telegram/channel/<channel>) in config/news_sources.json:
  [{"name": "my_channel", "url": "https://...", "category": "telegram", "poll_s": 30}]
Disable a default source: [{"name": "cnbc_world", "disabled": true}]
"""

from __future__ import annotations

import datetime as dt
import html as _html
import json
import logging
import re
import threading
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from typing import Dict, List, Optional

import requests

from ..calendar import IST, now_ist
from ..config import ROOT

log = logging.getLogger(__name__)
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/130.0 Safari/537.36 nifty-engine",
      "Accept": "application/rss+xml, application/atom+xml, application/xml, text/xml, */*"}
# Some publishers only answer clients that look like a browser.
BROWSER = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/537.36 (KHTML, like Gecko) "
                         "Chrome/130.0 Safari/537.36",
           "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
           "Accept-Language": "en-IN,en;q=0.9"}

# Status from the 2026-11-08 live check (sources-check): ok = parsed fine from a home connection.
DEFAULT_SOURCES: List[Dict] = [
    # --- regulators, exchanges, central banks (primary sources: fastest for what they publish)
    {"name": "rbi_press", "category": "regulator", "url": "https://rbi.org.in/pressreleases_rss.xml", "verified": True},
    {"name": "rbi_notifications", "category": "regulator", "url": "https://rbi.org.in/notifications_rss.xml", "verified": True},
    {"name": "sebi", "category": "regulator", "url": "https://www.sebi.gov.in/sebirss.xml", "verified": True},
    {"name": "fed_press", "category": "central_bank", "url": "https://www.federalreserve.gov/feeds/press_all.xml", "verified": True},
    {"name": "nse_announcements", "category": "exchange", "browser": True,
     "url": "https://nsearchives.nseindia.com/content/RSS/Online_announcements.xml"},
    {"name": "nse_results", "category": "exchange", "browser": True,
     "url": "https://nsearchives.nseindia.com/content/RSS/Financial_Results.xml"},
    {"name": "nse_board_meetings", "category": "exchange", "browser": True,
     "url": "https://nsearchives.nseindia.com/content/RSS/Board_Meetings.xml"},
    {"name": "nse_circulars", "category": "exchange", "browser": True, "poll_s": 300,
     "url": "https://nsearchives.nseindia.com/content/RSS/Circulars.xml"},
    # --- Indian financial media
    {"name": "et_markets", "category": "media_in", "verified": True, "url": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms"},
    {"name": "et_economy", "category": "media_in", "verified": True, "url": "https://economictimes.indiatimes.com/news/economy/rssfeeds/1373380680.cms"},
    {"name": "livemint_markets", "category": "media_in", "verified": True, "url": "https://www.livemint.com/rss/markets"},
    {"name": "cnbctv18_market", "category": "media_in", "verified": True, "url": "https://www.cnbctv18.com/commonfeeds/v1/cne/rss/market.xml"},
    {"name": "businessline_markets", "category": "media_in", "verified": True, "url": "https://www.thehindubusinessline.com/markets/feeder/default.rss"},
    {"name": "moneycontrol_latest", "category": "media_in", "browser": True, "url": "https://www.moneycontrol.com/rss/latestnews.xml"},
    {"name": "moneycontrol_markets", "category": "media_in", "browser": True, "url": "https://www.moneycontrol.com/rss/marketreports.xml"},
    {"name": "bs_markets", "category": "media_in", "browser": True, "url": "https://www.business-standard.com/rss/markets-106.rss"},
    {"name": "ndtvprofit", "category": "media_in", "timeout": 15, "url": "https://feeds.feedburner.com/ndtvprofit-latest"},
    {"name": "fe_market", "category": "media_in", "url": "https://www.financialexpress.com/market/feed/"},
    {"name": "investing_in", "category": "media_in", "poll_s": 120, "timeout": 15, "url": "https://in.investing.com/rss/news.rss"},
    # --- global / aggregators
    {"name": "gnews_india_mkts", "category": "aggregator", "verified": True, "poll_s": 120,
     "url": "https://news.google.com/rss/search?q=%22Nifty+50%22+OR+Sensex+OR+%22Reserve+Bank+of+India%22+OR+%22Dalal+Street%22+OR+%22Indian+stocks%22+when:1d&hl=en-IN&gl=IN&ceid=IN:en"},
    {"name": "gnews_global_macro", "category": "aggregator", "verified": True, "poll_s": 180,
     "url": "https://news.google.com/rss/search?q=%22Federal+Reserve%22+OR+%22Treasury+yields%22+OR+%22crude+oil%22+OR+OPEC+when:1d&hl=en-US&gl=US&ceid=US:en"},
    {"name": "marketwatch_top", "category": "media_global", "verified": True, "url": "https://feeds.content.dowjones.io/public/rss/mw_topstories"},
    {"name": "cnbc_world", "category": "media_global", "url": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000664"},
]


def load_sources() -> List[Dict]:
    out = [dict(s) for s in DEFAULT_SOURCES]
    f = ROOT / "config" / "news_sources.json"
    if f.exists():
        try:
            custom = json.loads(f.read_text(encoding="utf-8"))
            names = {s["name"] for s in out}
            for s in custom:
                if s.get("disabled"):
                    out = [o for o in out if o["name"] != s.get("name")]
                elif s.get("name") in names:
                    out = [dict(o, **s) if o["name"] == s["name"] else o for o in out]
                elif s.get("url"):
                    out.append(dict(s))
        except Exception as e:  # noqa: BLE001
            log.warning("config/news_sources.json ignored: %s", e)
    return out


# ----------------------------------------------------------------------------- parsing
_SEBI_DATE = re.compile(r"(\d{1,2}) (\w{3}),? (\d{4})")


def parse_date(s: Optional[str]) -> Optional[dt.datetime]:
    if not s:
        return None
    s = s.strip()
    try:
        d = parsedate_to_datetime(s)
        if d.tzinfo is None:
            d = d.replace(tzinfo=IST)         # e.g. RBI publishes local time without a zone
        return d.astimezone(IST)
    except Exception:  # noqa: BLE001
        pass
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%d %H:%M:%S"):
        try:
            d = dt.datetime.strptime(s.replace("Z", "+0000") if fmt.endswith("%z") else s, fmt)
            return (d if d.tzinfo else d.replace(tzinfo=IST)).astimezone(IST)
        except ValueError:
            continue
    m = _SEBI_DATE.search(s)                  # SEBI: "06 Nov, 2026 +0530" (date only)
    if m:
        try:
            return dt.datetime.strptime(" ".join(m.groups()), "%d %b %Y").replace(tzinfo=IST)
        except ValueError:
            return None
    return None


def _strip_ns(tag: str) -> str:
    return tag.split("}", 1)[-1]


_ITEM_RE = re.compile(rb"<(item|entry)\b.*?</\1>", re.S | re.I)
_FIELD_RE = {k: re.compile(rb"<%s\b[^>]*>(.*?)</%s>" % (k.encode(), k.encode()), re.S | re.I)
             for k in ("title", "link", "pubDate", "published", "updated", "description")}
_HREF_RE = re.compile(rb'href="([^"]+)"')


def _lenient(content: bytes, source: str) -> List[Dict]:
    """Fallback for feeds that are not well-formed XML (unescaped '&', stray bytes)."""
    out = []
    for m in _ITEM_RE.finditer(content):
        block = m.group(0)

        def field(k):
            f = _FIELD_RE[k].search(block)
            if not f:
                return None
            txt = f.group(1).decode("utf-8", "replace")
            txt = re.sub(r"<!\[CDATA\[(.*?)\]\]>", r"\1", txt, flags=re.S)
            return _html.unescape(re.sub(r"<[^>]+>", " ", txt)).strip()

        title = field("title")
        if not title:
            continue
        href = _HREF_RE.search(block)
        out.append({"source": source, "title": re.sub(r"\s+", " ", title),
                    "url": field("link") or (href.group(1).decode() if href else None),
                    "ts": parse_date(field("pubDate") or field("published") or field("updated")),
                    "summary": (field("description") or "")[:400]})
    return out


def parse_feed(content: bytes, source: str) -> List[Dict]:
    """RSS 2.0 or Atom -> [{title, url, ts, summary, source}]. Robust to namespaces and, via a
    regex fallback, to feeds that are not well-formed XML."""
    try:
        root = ET.fromstring(content)
    except ET.ParseError:
        return _lenient(content, source)
    items = []
    for el in root.iter():
        tag = _strip_ns(el.tag)
        if tag not in ("item", "entry"):
            continue
        rec = {"source": source}
        for ch in el:
            t = _strip_ns(ch.tag)
            if t == "title":
                rec["title"] = re.sub(r"\s+", " ", (ch.text or "")).strip()
            elif t == "link":
                rec["url"] = (ch.text or ch.attrib.get("href") or "").strip()
            elif t in ("pubDate", "published", "updated", "date") and "ts_raw" not in rec:
                rec["ts_raw"] = ch.text
            elif t in ("description", "summary") and ch.text:
                rec["summary"] = re.sub(r"<[^>]+>", " ", ch.text)[:400].strip()
        if rec.get("title"):
            rec["ts"] = parse_date(rec.pop("ts_raw", None))
            items.append(rec)
    return items


# ----------------------------------------------------------------------------- poller
@dataclass
class SourceState:
    cfg: Dict
    etag: Optional[str] = None
    last_modified: Optional[str] = None
    next_due: float = 0.0
    failures: int = 0
    ok: Optional[bool] = None
    last_ok: Optional[str] = None
    last_error: Optional[str] = None
    latency_ms: Optional[float] = None
    items_new_today: int = 0
    items_total: int = 0
    newest: Optional[str] = None
    day: str = ""

    def public(self) -> Dict:
        return {"name": self.cfg["name"], "category": self.cfg.get("category"), "ok": self.ok,
                "last_ok": self.last_ok, "latency_ms": self.latency_ms, "items_new_today": self.items_new_today,
                "error": self.last_error, "poll_s": self.cfg.get("poll_s", 60), "verified": bool(self.cfg.get("verified")),
                "newest": self.newest}


class NewsPoller:
    """Polls all sources on their own schedules; returns the items of sources that changed."""

    def __init__(self, sources: Optional[List[Dict]] = None, default_poll_s: int = 60, timeout: float = 8.0,
                 session: Optional[requests.Session] = None):
        self.states = {s["name"]: SourceState(s) for s in (sources or load_sources())}
        self.default_poll_s = default_poll_s
        self.timeout = timeout
        self.s = session or requests.Session()
        self.s.headers.update(UA)
        self._lock = threading.Lock()
        self._nse_warm = 0.0

    def _warm_nse(self) -> None:
        """nsearchives.nseindia.com answers 503 without the cookies nseindia.com sets on a normal
        visit. Visit the home page first (at most every 30 minutes)."""
        with self._lock:
            if time.time() - self._nse_warm < 1800:
                return
            self._nse_warm = time.time()
        try:
            self.s.get("https://www.nseindia.com/", timeout=self.timeout, headers=BROWSER)
        except Exception:  # noqa: BLE001
            pass

    def headers_for(self, st: SourceState) -> Dict:
        hdr = dict(BROWSER) if st.cfg.get("browser") else {}
        if "nseindia.com" in st.cfg["url"]:
            self._warm_nse()
            hdr["Referer"] = "https://www.nseindia.com/rss-feed"
        return hdr

    def _fetch(self, st: SourceState) -> List[Dict]:
        hdr = self.headers_for(st)
        if st.etag:
            hdr["If-None-Match"] = st.etag
        if st.last_modified:
            hdr["If-Modified-Since"] = st.last_modified
        t0 = time.time()
        today = now_ist().date().isoformat()
        if st.day != today:
            st.day, st.items_new_today = today, 0
        try:
            r = self.s.get(st.cfg["url"], headers=hdr, timeout=st.cfg.get("timeout", self.timeout))
            st.latency_ms = round((time.time() - t0) * 1000)
            if r.status_code == 304:
                st.ok, st.last_ok, st.last_error, st.failures = True, now_ist().isoformat(), None, 0
                return []
            if r.status_code in (403, 404, 410, 429) or r.status_code >= 500:
                raise requests.HTTPError(f"HTTP {r.status_code}")
            r.raise_for_status()
            items = parse_feed(r.content, st.cfg["name"])
            if not items:
                raise ValueError("no items parsed (not an RSS/Atom feed?)")
            st.etag, st.last_modified = r.headers.get("ETag"), r.headers.get("Last-Modified")
            st.ok, st.last_ok, st.last_error, st.failures = True, now_ist().isoformat(), None, 0
            ts = [i["ts"] for i in items if i.get("ts")]
            st.newest = max(ts).isoformat() if ts else st.newest
            for i in items:
                i["category"] = st.cfg.get("category")
            return items
        except Exception as e:  # noqa: BLE001
            st.ok, st.last_error = False, str(e)[:160]
            st.failures += 1
            st.latency_ms = round((time.time() - t0) * 1000)
            return []

    def poll(self, force: bool = False, max_workers: int = 8) -> List[Dict]:
        now = time.time()
        due = [st for st in self.states.values() if force or now >= st.next_due]
        if not due:
            return []
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            results = list(ex.map(self._fetch, due))
        out = []
        for st, items in zip(due, results):
            base = st.cfg.get("poll_s", self.default_poll_s)
            st.next_due = time.time() + (min(1800, base * (2 ** min(st.failures, 6))) if st.failures else base)
            out += items
        return out

    def record_new(self, n_by_source: Dict[str, int]) -> None:
        for name, n in n_by_source.items():
            if name in self.states:
                self.states[name].items_new_today += n
                self.states[name].items_total += n

    def health(self) -> List[Dict]:
        return [st.public() for st in self.states.values()]


# ----------------------------------------------------------------------------- one-off check
def check_all(sources: Optional[List[Dict]] = None, timeout: float = 12.0) -> List[Dict]:
    """Fetch every source once (no caching) and measure what you would get from this machine."""
    p = NewsPoller(sources, timeout=timeout)

    def one(st: SourceState) -> Dict:
        t0 = time.time()
        rec = {"name": st.cfg["name"], "category": st.cfg.get("category"), "url": st.cfg["url"]}
        try:
            r = p.s.get(st.cfg["url"], headers=p.headers_for(st), timeout=max(timeout, st.cfg.get("timeout", 0)))
            rec["http"] = r.status_code
            rec["latency_ms"] = round((time.time() - t0) * 1000)
            rec["etag_or_lastmod"] = bool(r.headers.get("ETag") or r.headers.get("Last-Modified"))
            items = parse_feed(r.content, st.cfg["name"]) if r.ok else []
            rec["items"] = len(items)
            ts = [i["ts"] for i in items if i.get("ts")]
            if ts:
                newest = max(ts)
                rec["newest"] = newest.isoformat()
                rec["newest_age_min"] = round((now_ist() - newest).total_seconds() / 60, 1)
                rec["newest_title"] = next(i["title"] for i in items if i.get("ts") == newest)[:120]
            rec["ok"] = bool(r.ok and items)
            if not rec["ok"] and not rec.get("error"):
                rec["error"] = f"HTTP {r.status_code}" if not r.ok else "no items parsed"
        except Exception as e:  # noqa: BLE001
            rec.update({"ok": False, "error": str(e)[:160], "latency_ms": round((time.time() - t0) * 1000)})
        return rec

    with ThreadPoolExecutor(max_workers=8) as ex:
        return list(ex.map(one, p.states.values()))


def write_check_report(rows: List[Dict], path) -> None:
    L = ["# News sources check", "", f"_Run from this machine at {now_ist().strftime('%Y-%m-%d %H:%M IST')}._", "",
         f"Working: **{sum(r.get('ok', False) for r in rows)} of {len(rows)}**. A feed is usable if it answers with "
         "parseable RSS/Atom. 'Newest age' is how old the newest item was at the time of the check; during market "
         "hours fast feeds are a few minutes old (on weekends and holidays, hours or days).", "",
         "| Source | Category | OK | HTTP | Latency | Items | Newest age (min) | Conditional GET | Newest headline / error |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (not r.get("ok"), r.get("category") or "", r["name"])):
        L.append(f"| {r['name']} | {r.get('category')} | {'yes' if r.get('ok') else 'NO'} | {r.get('http', '—')} | "
                 f"{r.get('latency_ms', '—')} ms | {r.get('items', 0)} | {r.get('newest_age_min', '—')} | "
                 f"{'yes' if r.get('etag_or_lastmod') else 'no'} | {(r.get('newest_title') or r.get('error') or '').replace('|', '/')} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L), encoding="utf-8")
