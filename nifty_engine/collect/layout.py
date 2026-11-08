"""On-disk layout of the `data` branch and append-safe gzip/JSONL helpers.

Everything for one session lives in one folder, so a CI run only needs a sparse checkout of the
last few days (the branch grows by ~2 MB per session and a full checkout would get slow):

    MANIFEST.json                         summary: days, sizes, last run per mode, expired options
    README.md                             what this branch is
    meta/instruments.json                 instrument keys resolved today (future, stocks, expiries)
    runs/YYYY-MM.jsonl                    one line per collector run (status, counts, errors)
    days/YYYY-MM-DD/
        chain_<expiry>.csv.gz             option-chain snapshots (recorder CHAIN_COLS + extras)
        chain_metrics.jsonl               derived metrics per snapshot (recorder format)
        quotes.csv.gz                     full quotes with 5-level depth, one row per instrument/snapshot
        news.jsonl                        headlines first seen this day (news_llm store format)
        candles/<series>_<1m|5m>.csv.gz   today's candles (nifty, indiavix, banknifty, niftyfut, stocks)
        expired_options_1m.csv.gz         (expiry days only) 1-minute candles of that expiry's options

Write rules (all idempotent, so a re-run or a retry after a push conflict never duplicates):
  * snapshot tables are appended as new gzip *members* containing only rows whose key is new; the
    old bytes are never rewritten, which keeps git deltas tiny;
  * candle / expired-option tables are upserted (key -> last value) and rewritten only if changed;
  * JSONL files are appended with rows whose key is new.
All gzip output uses mtime=0, so identical content gives identical bytes.
"""

from __future__ import annotations

import datetime as dt
import gzip
import io
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd

from ..calendar import IST, now_ist

LAYOUT_VERSION = 1

CHAIN_EXTRA = ["instrument_key", "close_price", "pop", "pcr"]
QUOTE_COLS = (["ts", "key", "symbol", "last_price", "open", "high", "low", "close", "volume", "average_price", "oi",
               "oi_day_high", "oi_day_low", "net_change", "total_buy_quantity", "total_sell_quantity",
               "lower_circuit_limit", "upper_circuit_limit", "last_trade_time"]
              + [f"{side}{i}_{f}" for side in ("bid", "ask") for i in range(1, 6) for f in ("price", "qty", "orders")])
CANDLE_COLS = ["ts", "open", "high", "low", "close", "volume", "oi"]
EXPIRED_COLS = ["ts", "open", "high", "low", "close", "volume", "oi", "strike", "type", "key"]
NEWS_FIELDS = ["id", "title", "summary", "source", "url", "published", "first_seen"]


# ----------------------------------------------------------------------------- paths / time
def day_dir(root: Path, d: dt.date) -> Path:
    return Path(root) / "days" / d.isoformat()


def ts_iso(x) -> Optional[str]:
    """IST-aware ISO string with microseconds (the recorder's format; one format per column keeps
    pandas' ISO parser happy)."""
    if x is None or (isinstance(x, float) and x != x):
        return None
    t = pd.Timestamp(x)
    if t is pd.NaT:
        return None
    t = t.tz_localize(IST) if t.tzinfo is None else t.tz_convert(IST)
    return t.isoformat(timespec="microseconds")


def ts_minute(x) -> Optional[str]:
    """IST-aware ISO string to the second (candle timestamps)."""
    if x is None:
        return None
    t = pd.Timestamp(x)
    if t is pd.NaT:
        return None
    t = t.tz_localize(IST) if t.tzinfo is None else t.tz_convert(IST)
    return t.isoformat(timespec="seconds")


def parse_ts(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True, format="ISO8601", errors="coerce").dt.tz_convert(IST)


# ----------------------------------------------------------------------------- CSV (gzip)
def read_csv_any(path: Path, as_str: bool = True) -> pd.DataFrame:
    """Read a .csv or .csv.gz (multi-member gzip is fine). Header lines repeated inside the file
    (e.g. a plain CSV appended to a .gz) are dropped. Missing or unreadable file -> empty frame."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        kw = {"dtype": str, "keep_default_na": False} if as_str else {}
        df = pd.read_csv(path, **kw)
    except (pd.errors.EmptyDataError, OSError, EOFError, ValueError):
        return pd.DataFrame()
    if len(df.columns) and len(df):
        first = df.columns[0]
        df = df[df[first].astype(str) != first]
    return df.reset_index(drop=True)


def _header(path: Path) -> List[str]:
    try:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            line = fh.readline()
    except (OSError, EOFError):
        return []
    return [c.strip() for c in line.rstrip("\n").split(",")] if line.strip() else []


def _norm_str(df: pd.DataFrame) -> pd.DataFrame:
    """Round-trip through CSV text so new rows are compared exactly as they will be stored."""
    if df.empty:
        return df
    buf = io.StringIO()
    df.to_csv(buf, index=False, lineterminator="\n")
    buf.seek(0)
    return pd.read_csv(buf, dtype=str, keep_default_na=False)


def key_series(df: pd.DataFrame, keys: Sequence[str]) -> pd.Series:
    """One string per row identifying it by `keys`; numbers and timestamps are normalised so
    '24500', '24500.0' and 24500 (or two spellings of the same instant) compare equal."""
    if df.empty:
        return pd.Series([], dtype=object)
    parts = []
    for k in keys:
        s = df[k] if k in df else pd.Series([""] * len(df), index=df.index)
        if k == "ts":
            t = pd.to_datetime(s, utc=True, format="ISO8601", errors="coerce")
            p = pd.Series([str(int(v.value)) if v is not pd.NaT and v == v else str(o)
                           for v, o in zip(t, s)], index=df.index, dtype=object)
        else:
            num = pd.to_numeric(s, errors="coerce")
            p = pd.Series([repr(float(n)) if n == n else str(o) for n, o in zip(num, s)], index=df.index, dtype=object)
        parts.append(p)
    out = parts[0]
    for p in parts[1:]:
        out = out + "|" + p
    return out


def _gz_bytes(text: str) -> bytes:
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as gz:
        gz.write(text.encode("utf-8"))
    return buf.getvalue()


def append_gz_csv(path: Path, df: pd.DataFrame, cols: Sequence[str], keys: Sequence[str]) -> int:
    """Append rows whose key is not yet in the file, as a new gzip member. Returns rows added."""
    if df is None or df.empty:
        return 0
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    header = _header(path) if exists else list(cols)
    if not header:                          # unreadable file: start a fresh one next to it
        header, exists = list(cols), False
        path.replace(path.with_suffix(path.suffix + ".corrupt"))
    new = _norm_str(df.reindex(columns=header))
    k_new = key_series(new, keys)
    new = new[~k_new.duplicated(keep="first").values]
    if exists:
        old = read_csv_any(path)
        have = set(key_series(old, keys)) if len(old) else set()
        new = new[[k not in have for k in key_series(new, keys)]]
    if new.empty:
        return 0
    text = new.to_csv(index=False, header=not exists, lineterminator="\n")
    with path.open("ab") as raw:
        raw.write(_gz_bytes(text))
    return len(new)


def upsert_gz_csv(path: Path, df: pd.DataFrame, cols: Sequence[str], keys: Sequence[str],
                  sort: Optional[Sequence[str]] = None) -> int:
    """Merge rows by key (new values win), sort, rewrite only if the content changed.
    Returns the number of rows that are new or changed."""
    if df is None or df.empty:
        return 0
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    old = read_csv_any(path)
    header = list(old.columns) if len(old.columns) else list(cols)
    for c in cols:
        if c not in header:
            header.append(c)
    old = old.reindex(columns=header, fill_value="")
    new = _norm_str(df.reindex(columns=header))
    new = new[~key_series(new, keys).duplicated(keep="last").values]
    if len(old):
        before = dict(zip(key_series(old, keys), old.astype(str).agg("\x1f".join, axis=1)))
        after = new.astype(str).agg("\x1f".join, axis=1)
        changed = sum(before.get(k) != v for k, v in zip(key_series(new, keys), after))
    else:
        changed = len(new)
    if changed == 0:
        return 0
    allr = pd.concat([old, new], ignore_index=True)
    allr = allr[~key_series(allr, keys).duplicated(keep="last").values]
    allr = sort_frame(allr, list(sort or keys))
    path.write_bytes(_gz_bytes(allr.to_csv(index=False, lineterminator="\n")))
    return int(changed)


def sort_frame(df: pd.DataFrame, order: Sequence[str]) -> pd.DataFrame:
    """Stable sort of a string frame: 'ts' chronologically, 'strike' numerically, others as text."""
    def sort_key(c):
        if c == "ts":
            return parse_ts(df[c])
        if c == "strike":
            return pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        return df[c].astype(str)
    sk = pd.DataFrame({c: sort_key(c) for c in order if c in df})
    if not len(sk.columns):
        return df
    return df.loc[sk.sort_values(list(sk.columns), kind="mergesort").index]


# ----------------------------------------------------------------------------- JSONL
def read_jsonl(path: Path) -> List[Dict]:
    path = Path(path)
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def _clean(v):
    if isinstance(v, float) and v != v:
        return None
    if hasattr(v, "item") and not isinstance(v, (str, bytes)):
        try:
            return v.item()
        except (ValueError, AttributeError):
            return v
    return v


def append_jsonl(path: Path, rows: Iterable[Dict], key: Optional[str] = None,
                 known: Optional[set] = None) -> int:
    """Append rows whose `key` value is not already in the file (or in `known`)."""
    rows = list(rows)
    if not rows:
        return 0
    path = Path(path)
    have = set(known or ())
    if key:
        have |= {str(r.get(key)) for r in read_jsonl(path)}
    out = []
    for r in rows:
        if key:
            k = str(r.get(key))
            if k in have:
                continue
            have.add(k)
        out.append(json.dumps({a: _clean(b) for a, b in r.items()}, default=str, ensure_ascii=False))
    if not out:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(out) + "\n")
    return len(out)


def recent_news_ids(root: Path, d: dt.date, back: int = 4, ahead: int = 0) -> set:
    ids = set()
    for k in range(-ahead, back + 1):
        for r in read_jsonl(day_dir(root, d - dt.timedelta(days=k)) / "news.jsonl"):
            if r.get("id"):
                ids.add(str(r["id"]))
    return ids


# ----------------------------------------------------------------------------- run records / manifest
def run_record_path(root: Path, ts: dt.datetime) -> Path:
    return Path(root) / "runs" / f"{ts.astimezone(IST).strftime('%Y-%m')}.jsonl"


def new_run_id() -> str:
    gh = os.environ.get("GITHUB_RUN_ID")
    if gh:
        return f"gh-{gh}-{os.environ.get('GITHUB_RUN_ATTEMPT', '1')}"
    return "local-" + now_ist().strftime("%Y%m%dT%H%M%S%f")


def load_manifest(root: Path) -> Dict:
    p = Path(root) / "MANIFEST.json"
    try:
        m = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(m, dict):
            return m
    except (OSError, ValueError):
        pass
    return {"layout_version": LAYOUT_VERSION, "created": ts_iso(now_ist()), "days": {}, "last_runs": {},
            "expired_options": {}}


def save_manifest(root: Path, m: Dict) -> None:
    m["layout_version"] = LAYOUT_VERSION
    m["updated"] = ts_iso(now_ist())
    days = m.get("days", {})
    m["totals"] = {"days": len(days), "bytes": int(sum(v.get("bytes", 0) for v in days.values())),
                   "snapshots": int(sum(v.get("snapshots", 0) for v in days.values())),
                   "news": int(sum(v.get("news", 0) for v in days.values()))}
    p = Path(root) / "MANIFEST.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(m, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")


def describe_day(root: Path, d: str) -> Dict:
    base = Path(root) / "days" / d
    files = {str(p.relative_to(base)).replace(os.sep, "/"): p.stat().st_size for p in sorted(base.rglob("*")) if p.is_file()}
    return {"files": files, "bytes": int(sum(files.values())),
            "snapshots": len(read_jsonl(base / "chain_metrics.jsonl")),
            "news": len(read_jsonl(base / "news.jsonl"))}


README = """# NIFTY engine data branch

This orphan branch is written by GitHub Actions (`.github/workflows/collect_market.yml` and
`collect_eod.yml` on `main`). It holds market data that cannot be downloaded later: option-chain
snapshots, quotes with order-book depth, first-seen news headlines, the day's 1-/5-minute candles
and 1-minute candles of each weekly expiry's options.

Do not edit by hand. Pull it into your local engine store with:

    python -m nifty_engine.collect sync

Layout (see `docs/DATA_COLLECTION.md` on `main` for details):

    MANIFEST.json                       summary (days, sizes, last run of each job)
    meta/instruments.json               instrument keys used today
    runs/YYYY-MM.jsonl                  one line per collector run
    days/YYYY-MM-DD/chain_<expiry>.csv.gz
    days/YYYY-MM-DD/chain_metrics.jsonl
    days/YYYY-MM-DD/quotes.csv.gz
    days/YYYY-MM-DD/news.jsonl
    days/YYYY-MM-DD/candles/<series>_<1m|5m>.csv.gz
    days/YYYY-MM-DD/expired_options_1m.csv.gz   (on/after weekly expiries; Upstox Plus)

All timestamps are ISO 8601 in IST (+05:30).
"""
