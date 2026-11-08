"""Merge a collector staging folder into the data-branch working tree.

Used right after collection, and again by the workflow's push-retry loop: when the push is
rejected because the remote moved, the workflow resets the working tree to the new remote tip and
re-runs this merge from the same staging folder (a conflict-free "rebase" of this run's data).
Every rule is idempotent, so merging the same staging folder twice changes nothing.
"""

from __future__ import annotations

import datetime as dt
import fnmatch
import logging
import shutil
from pathlib import Path
from typing import Dict

from .layout import (README, append_gz_csv, append_jsonl, describe_day, load_manifest, read_csv_any, read_jsonl,
                     recent_news_ids, save_manifest, upsert_gz_csv)

log = logging.getLogger(__name__)


def _csv_append(keys):
    def f(src: Path, dst: Path, rel: str, root: Path) -> int:
        df = read_csv_any(src)
        return append_gz_csv(dst, df, list(df.columns), keys) if len(df) else 0
    return f


def _csv_upsert(keys, sort=None):
    def f(src: Path, dst: Path, rel: str, root: Path) -> int:
        df = read_csv_any(src)
        return upsert_gz_csv(dst, df, list(df.columns), keys, sort=sort) if len(df) else 0
    return f


def _jsonl(key):
    def f(src: Path, dst: Path, rel: str, root: Path) -> int:
        return append_jsonl(dst, read_jsonl(src), key=key)
    return f


def _news(src: Path, dst: Path, rel: str, root: Path) -> int:
    d = dt.date.fromisoformat(rel.split("/")[1])
    known = recent_news_ids(root, d, back=4, ahead=1)
    return append_jsonl(dst, read_jsonl(src), key="id", known=known)


def _copy(src: Path, dst: Path, rel: str, root: Path) -> int:
    if dst.exists() and dst.read_bytes() == src.read_bytes():
        return 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    return 1


RULES = [
    ("days/*/chain_*.csv.gz", _csv_append(["ts", "strike", "type"])),
    ("days/*/quotes.csv.gz", _csv_append(["ts", "key"])),
    ("days/*/candles/*.csv.gz", _csv_upsert(["ts"])),
    ("days/*/expired_options_1m.csv.gz", _csv_upsert(["key", "ts"], sort=["strike", "type", "ts"])),
    ("days/*/chain_metrics.jsonl", _jsonl("ts")),
    ("days/*/news.jsonl", _news),
    ("days/*/*.json", _copy),
    ("runs/*.jsonl", _jsonl("run_id")),
    ("meta/*", _copy),
]


def merge_tree(staging: Path, target: Path) -> Dict:
    staging, target = Path(staging), Path(target)
    target.mkdir(parents=True, exist_ok=True)
    stats = {"files": 0, "rows": 0, "skipped": []}
    days = set()
    runs = []
    for src in sorted(p for p in staging.rglob("*") if p.is_file()):
        rel = src.relative_to(staging).as_posix()
        rule = next((fn for pat, fn in RULES if fnmatch.fnmatch(rel, pat)), None)
        if rule is None:
            stats["skipped"].append(rel)
            continue
        try:
            n = rule(src, target / rel, rel, target)
        except Exception as e:  # noqa: BLE001 - one bad file must not stop the rest
            log.warning("merge %s failed: %s", rel, e)
            stats.setdefault("errors", []).append(f"{rel}: {str(e)[:160]}")
            continue
        stats["files"] += 1
        stats["rows"] += int(n or 0)
        if rel.startswith("days/"):
            days.add(rel.split("/")[1])
        if rel.startswith("runs/"):
            runs += read_jsonl(src)
    m = load_manifest(target)
    for d in sorted(days):
        m.setdefault("days", {})[d] = describe_day(target, d)
        side = target / "days" / d / "expired_options_1m.json"
        if side.exists():
            try:
                import json
                m.setdefault("expired_options", {})[d] = json.loads(side.read_text(encoding="utf-8"))
            except ValueError:
                pass
    for r in runs:
        keep = {k: r.get(k) for k in ("run_id", "mode", "started", "finished", "market", "token", "n_errors",
                                       "token_invalid", "chains", "quotes", "last_trade")}
        keep["news_new"] = (r.get("news") or {}).get("new")
        m.setdefault("last_runs", {})[r.get("mode", "?")] = {k: v for k, v in keep.items() if v is not None}
    save_manifest(target, m)
    if not (target / "README.md").exists():
        (target / "README.md").write_text(README, encoding="utf-8")
    stats["days"] = sorted(days)
    return stats
