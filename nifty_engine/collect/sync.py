"""`python -m nifty_engine.collect sync`: pull the `data` branch into the local engine store.

Source: a shallow clone of the data branch, kept at store/cache/data_branch (created/updated with
`git clone/fetch --depth 1`, so your main checkout is never touched), or any folder given with
--data-branch-dir (then it is read as-is).

Targets, in the formats the engine already reads:
  days/D/chain_<e>.csv.gz      -> store/live/D/chain_<e>.csv.gz        (recorder CHAIN_COLS)
  days/D/chain_metrics.jsonl   -> store/live/D/chain_metrics.jsonl     (recorder format)
  days/D/quotes.csv.gz         -> store/live/D/quotes.csv.gz + depth.csv.gz (recorder DEPTH_COLS)
  days/D/news.jsonl            -> store/news/headlines/<first_seen date>.jsonl (no duplicate ids)
  days/D/candles/<s>.csv.gz    -> store/intraday/<s>.csv               (append, de-duplicated on ts)
  days/E/expired_options_1m    -> store/options_hist_1m/<E>.csv

Merging is idempotent. Recorder files for today are not touched before 15:45 IST (the live
recorder may still be appending); pass --include-today to override. A small state file
(store/cache/collect_sync.json) remembers which source files were already merged, so a daily sync
only reads what changed; --full ignores it.
"""

from __future__ import annotations

import datetime as dt
import io
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from ..calendar import IST, now_ist
from ..config import ROOT
from ..data.recorder import CHAIN_COLS, DEPTH_COLS
from .layout import NEWS_FIELDS, _gz_bytes, append_jsonl, parse_ts, read_csv_any, read_jsonl, sort_frame, ts_iso

log = logging.getLogger(__name__)
LIVE_SAFE_AFTER = dt.time(15, 45)
BAR_COLS = ["open", "high", "low", "close", "volume", "oi"]


# ----------------------------------------------------------------------------- fetching
def default_repo_url() -> Optional[str]:
    env = os.environ.get("NE_DATA_REPO", "").strip()
    if env:
        return env
    try:
        r = subprocess.run(["git", "-C", str(ROOT), "remote", "get-url", "origin"], capture_output=True, text=True,
                           timeout=20)
        return r.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def fetch_data_branch(dest: Path, repo: Optional[str] = None, branch: str = "data",
                      run: Callable = subprocess.run) -> Dict:
    """Create or update a shallow single-branch clone of the data branch at `dest`."""
    dest = Path(dest)
    if (dest / ".git").exists():
        cmds = [["git", "-C", str(dest), "fetch", "--depth", "1", "origin", branch],
                ["git", "-C", str(dest), "reset", "--hard", "-q", "FETCH_HEAD"]]
    else:
        repo = repo or default_repo_url()
        if not repo:
            return {"ok": False, "error": "no repository URL (set --repo or NE_DATA_REPO)"}
        dest.parent.mkdir(parents=True, exist_ok=True)
        cmds = [["git", "clone", "-q", "--depth", "1", "--single-branch", "--branch", branch, repo, str(dest)]]
    for c in cmds:
        r = run(c, capture_output=True, text=True, timeout=1800)
        if r.returncode != 0:
            return {"ok": False, "error": (r.stderr or r.stdout or "").strip()[-400:], "cmd": " ".join(c[:4])}
    return {"ok": True, "dir": str(dest)}


# ----------------------------------------------------------------------------- helpers
def _write_gz_if_changed(path: Path, df: pd.DataFrame) -> bool:
    data = _gz_bytes(df.to_csv(index=False, lineterminator="\n"))
    if path.exists():
        try:
            old = read_csv_any(path)
            if list(old.columns) == list(df.columns) and len(old) == len(df) and \
                    (old.astype(str).values == df.astype(str).values).all():
                return False
        except Exception:  # noqa: BLE001
            pass
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)
    return True


def _norm_ts_col(df: pd.DataFrame) -> pd.DataFrame:
    t = parse_ts(df["ts"])
    df = df[t.notna().values].copy()
    df["ts"] = [x.isoformat(timespec="microseconds") for x in t[t.notna()]]
    return df


def _merge_live_table(day: Path, name: str, new: pd.DataFrame, cols: List[str], keys: List[str],
                      sort: List[str]) -> int:
    """Union recorder file(s) <name>.csv.gz / <name>.csv with `new` (local rows win on equal keys),
    sort by time, write one .csv.gz and fold the plain .csv into it. Returns rows added."""
    from .layout import key_series
    gz, plain = day / f"{name}.csv.gz", day / f"{name}.csv"
    local = [read_csv_any(p) for p in (gz, plain)]
    local = [f.reindex(columns=cols, fill_value="") for f in local if len(f)]
    n_local = sum(len(f) for f in local)
    new = new.reindex(columns=cols)
    if len(new):
        new = pd.read_csv(io.StringIO(new.to_csv(index=False)), dtype=str, keep_default_na=False)
    allr = pd.concat(local + [new], ignore_index=True) if (local or len(new)) else pd.DataFrame(columns=cols)
    if allr.empty:
        return 0
    allr = _norm_ts_col(allr)
    allr = allr[~key_series(allr, keys).duplicated(keep="first").values]
    allr = sort_frame(allr, sort).reset_index(drop=True)
    _write_gz_if_changed(gz, allr)
    if plain.exists():          # its rows are now in the .csv.gz (what recorder.compress_day would do)
        plain.unlink()
    return max(0, len(allr) - n_local)


def quotes_to_depth(q: pd.DataFrame) -> pd.DataFrame:
    """Collector quotes (5-level REST depth) -> recorder depth rows (DEPTH_COLS)."""
    if q.empty:
        return pd.DataFrame(columns=DEPTH_COLS)
    num = lambda c: pd.to_numeric(q.get(c), errors="coerce") if c in q else pd.Series(np.nan, index=q.index)  # noqa: E731
    bq = sum(num(f"bid{i}_qty").fillna(0) for i in range(1, 6))
    aq = sum(num(f"ask{i}_qty").fillna(0) for i in range(1, 6))
    levels = pd.concat([(num(f"bid{i}_price") > 0) | (num(f"ask{i}_price") > 0) for i in range(1, 6)], axis=1).sum(axis=1)
    tot = bq + aq
    d = pd.DataFrame({
        "ts": q["ts"], "key": q["key"], "ltp": num("last_price"), "cp": num("last_price") - num("net_change"),
        "vtt": num("volume"), "oi": num("oi"), "tbq": num("total_buy_quantity"), "tsq": num("total_sell_quantity"),
        "bid1": num("bid1_price"), "ask1": num("ask1_price"), "bidq1": num("bid1_qty"), "askq1": num("ask1_qty"),
        "bidq_sum": bq, "askq_sum": aq, "imbalance": ((bq - aq) / tot.replace(0, np.nan)), "levels": levels,
        "iv": np.nan, "delta": np.nan, "gamma": np.nan, "atp": num("average_price")})
    has_book = levels > 0
    for c in ("bidq_sum", "askq_sum", "levels"):
        d.loc[~has_book, c] = np.nan
    return d[DEPTH_COLS]


def _merge_metrics(day: Path, rows: List[Dict]) -> int:
    p = day / "chain_metrics.jsonl"
    local = read_jsonl(p)
    seen, out = set(), []
    for r in local + rows:
        try:
            t = pd.Timestamp(r.get("ts"))
            t = t.tz_localize(IST) if t.tzinfo is None else t.tz_convert(IST)
        except (ValueError, TypeError):
            continue
        if t.value in seen:
            continue
        seen.add(t.value)
        out.append((t, {**r, "ts": t.isoformat(timespec="microseconds")}))
    out.sort(key=lambda x: x[0])
    text = "".join(json.dumps(r, default=str) + "\n" for _, r in out)
    if p.exists() and p.read_text(encoding="utf-8") == text:
        return 0
    day.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return max(0, len(out) - len({pd.Timestamp(r["ts"]).value for r in local if r.get("ts")}))


def _merge_bars(path: Path, df: pd.DataFrame) -> int:
    """Same rules as live/intraday_store._merge_save: session bars only, de-duplicated on ts."""
    new = df.copy()
    new.index = parse_ts(new["ts"])
    new = new[new.index.notna()]
    for c in BAR_COLS:
        new[c] = pd.to_numeric(new[c], errors="coerce") if c in new else 0
    new = new[BAR_COLS]
    new = new[(new.index.time >= dt.time(9, 15)) & (new.index.time <= dt.time(15, 29))]
    if new.empty:
        return 0
    if path.exists():
        old = pd.read_csv(path, index_col=0)
        old.index = pd.to_datetime(old.index, utc=True, format="ISO8601").tz_convert(IST)
        merged = pd.concat([old, new])
    else:
        old, merged = None, new
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    merged.index.name = "ts"
    if old is not None:
        o = old[~old.index.duplicated(keep="last")].sort_index()
        if len(o) == len(merged) and o.index.equals(merged.index) and \
                np.allclose(o[BAR_COLS].to_numpy(float), merged[BAR_COLS].to_numpy(float), equal_nan=True):
            return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(path)
    return int(len(merged) - (len(old) if old is not None else 0))


def _merge_options_1m(path: Path, df: pd.DataFrame) -> int:
    cols = ["ts", "open", "high", "low", "close", "volume", "oi", "strike", "type", "key"]
    new = df.reindex(columns=cols).copy()
    old = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=cols)
    allr = pd.concat([old.reindex(columns=cols), new], ignore_index=True)
    allr["ts"] = parse_ts(allr["ts"].astype(str))
    allr = allr[allr["ts"].notna()]
    for c in ("open", "high", "low", "close", "volume", "oi", "strike"):
        allr[c] = pd.to_numeric(allr[c], errors="coerce")
    allr = allr.drop_duplicates(["key", "ts"], keep="first").sort_values(["strike", "type", "ts"], kind="mergesort")
    if len(allr) == len(old):
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    allr.to_csv(path, index=False)
    return int(len(allr) - len(old))


def _merge_news(head_dir: Path, rows: List[Dict], cache: Dict[str, set]) -> int:
    """Append headlines whose id is not already stored within +/- 4 days (news_llm's window)."""
    def ids_for(d: dt.date) -> set:
        k = d.isoformat()
        if k not in cache:
            cache[k] = {str(r.get("id")) for r in read_jsonl(head_dir / f"{k}.jsonl") if r.get("id")}
        return cache[k]

    n = 0
    by_day: Dict[dt.date, List[Dict]] = {}
    for r in rows:
        if not r.get("id") or not r.get("title") or not r.get("first_seen"):
            continue
        try:
            d = pd.Timestamp(r["first_seen"]).tz_convert(IST).date()
        except (ValueError, TypeError):
            continue
        by_day.setdefault(d, []).append({k: r.get(k) for k in NEWS_FIELDS})
    for d, rs in sorted(by_day.items()):
        known = set().union(*[ids_for(d + dt.timedelta(days=k)) for k in range(-4, 5)])
        added = append_jsonl(head_dir / f"{d.isoformat()}.jsonl", rs, key="id", known=known)
        ids_for(d).update(r["id"] for r in rs)
        n += added
    return n


# ----------------------------------------------------------------------------- sync
def sync(src: Path, store: Path, include_today: bool = False, full: bool = False,
         now: Optional[dt.datetime] = None) -> Dict:
    src, store = Path(src), Path(store)
    now = (now or now_ist()).astimezone(IST)
    state_p = store / "cache" / "collect_sync.json"
    try:
        state = json.loads(state_p.read_text(encoding="utf-8")) if not full else {}
    except (OSError, ValueError):
        state = {}
    seen: Dict[str, List[int]] = state.get("files", {})
    stats = {"chain_rows": 0, "metrics": 0, "quotes": 0, "depth": 0, "news": 0, "bars": 0, "option_bars": 0,
             "files_read": 0, "files_unchanged": 0, "deferred": [], "errors": []}
    news_cache: Dict[str, set] = {}
    days_root = src / "days"
    for day in sorted(p for p in days_root.iterdir() if p.is_dir()) if days_root.exists() else []:
        try:
            D = dt.date.fromisoformat(day.name)
        except ValueError:
            continue
        live_ok = include_today or D < now.date() or now.time() >= LIVE_SAFE_AFTER
        live = store / "live" / D.isoformat()
        for f in sorted(p for p in day.rglob("*") if p.is_file()):
            rel = f.relative_to(src).as_posix()
            st = f.stat()
            fp = [st.st_size, st.st_mtime_ns]
            if seen.get(rel) == fp:
                stats["files_unchanged"] += 1
                continue
            name = f.relative_to(day).as_posix()
            is_live = name.startswith("chain_") or name == "quotes.csv.gz"
            if is_live and not live_ok:
                stats["deferred"].append(rel)
                continue
            try:
                if name.startswith("chain_") and name.endswith(".csv.gz"):
                    stats["chain_rows"] += _merge_live_table(live, name[:-7], read_csv_any(f), CHAIN_COLS,
                                                             ["ts", "strike", "type"], ["ts", "strike", "type"])
                elif name == "chain_metrics.jsonl":
                    stats["metrics"] += _merge_metrics(live, read_jsonl(f))
                elif name == "quotes.csv.gz":
                    q = read_csv_any(f)
                    stats["quotes"] += _merge_live_table(live, "quotes", q, list(q.columns), ["ts", "key"], ["ts", "key"])
                    stats["depth"] += _merge_live_table(live, "depth", quotes_to_depth(q), DEPTH_COLS, ["ts", "key"],
                                                        ["ts", "key"])
                elif name == "news.jsonl":
                    stats["news"] += _merge_news(store / "news" / "headlines", read_jsonl(f), news_cache)
                elif name.startswith("candles/") and name.endswith(".csv.gz"):
                    series = name[len("candles/"):-len(".csv.gz")]
                    stats["bars"] += _merge_bars(store / "intraday" / f"{series}.csv", read_csv_any(f))
                elif name == "expired_options_1m.csv.gz":
                    stats["option_bars"] += _merge_options_1m(store / "options_hist_1m" / f"{D.isoformat()}.csv",
                                                              read_csv_any(f))
                else:
                    seen[rel] = fp
                    continue
            except Exception as e:  # noqa: BLE001
                log.warning("sync %s failed: %s", rel, e)
                stats["errors"].append(f"{rel}: {str(e)[:160]}")
                continue
            seen[rel] = fp
            stats["files_read"] += 1
    state = {"files": seen, "last_sync": ts_iso(now), "source": str(src)}
    state_p.parent.mkdir(parents=True, exist_ok=True)
    state_p.write_text(json.dumps(state, indent=0), encoding="utf-8")
    stats["deferred"] = stats["deferred"][:20]
    return stats


def status(data_dir: Optional[Path], store: Path) -> Dict:
    out: Dict = {}
    if data_dir and (Path(data_dir) / "MANIFEST.json").exists():
        m = json.loads((Path(data_dir) / "MANIFEST.json").read_text(encoding="utf-8"))
        days = m.get("days", {})
        out["data_branch"] = {
            "dir": str(data_dir), "updated": m.get("updated"), "totals": m.get("totals"),
            "last_runs": m.get("last_runs"),
            "recent_days": {d: {k: v for k, v in days[d].items() if k != "files"} for d in sorted(days)[-10:]},
            "expired_options": {k: {a: v.get(a) for a in ("atm", "wanted", "got", "complete")}
                                for k, v in sorted(m.get("expired_options", {}).items())[-6:]}}
    else:
        out["data_branch"] = {"dir": str(data_dir) if data_dir else None, "note": "no MANIFEST.json (not synced yet?)"}
    sp = Path(store) / "cache" / "collect_sync.json"
    if sp.exists():
        s = json.loads(sp.read_text(encoding="utf-8"))
        out["local_sync"] = {"last_sync": s.get("last_sync"), "files_merged": len(s.get("files", {}))}
    live = Path(store) / "live"
    out["local_store"] = {
        "live_days": len([p for p in live.iterdir() if p.is_dir()]) if live.exists() else 0,
        "headline_days": len(list((Path(store) / "news" / "headlines").glob("*.jsonl"))),
        "intraday_series": sorted(p.stem for p in (Path(store) / "intraday").glob("*.csv")),
        "options_hist_1m_expiries": len(list((Path(store) / "options_hist_1m").glob("*.csv")))}
    return out
