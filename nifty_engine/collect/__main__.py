"""CLI: python -m nifty_engine.collect <snapshot|eod|news|merge|sync|status> [options]

  snapshot   quotes + option chains (market hours only) and one news poll
  eod        today's 1-/5-minute candles, just-expired options (Upstox Plus) and one news poll
  news       one news poll
  merge      merge a staging folder (--out) into the data branch (--data-branch-dir); used by the
             workflow's push-retry loop
  sync       pull the data branch into the local store (run on the Mac)
  status     what has been collected and synced

Collect commands write this run's data to --out (staging) and, with --data-branch-dir, merge it
into the data-branch working tree. They exit 0 when the Upstox token is missing (news is still
collected) and never fail because one instrument failed.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m nifty_engine.collect", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["snapshot", "eod", "news", "merge", "sync", "status"])
    ap.add_argument("--out", type=Path, help="staging folder for this run's data (collect/merge)")
    ap.add_argument("--data-branch-dir", type=Path,
                    help="working tree of the data branch (collect/merge: merge target; sync/status: source, "
                         "read as-is; default for sync: a shallow clone at store/cache/data_branch)")
    ap.add_argument("--force", action="store_true", help="collect even outside market hours / on stale quotes")
    ap.add_argument("--no-expired", action="store_true", help="eod: skip expired-option candles")
    ap.add_argument("--store", type=Path, help="sync/status: local store (default: NE_DATA_DIR or ./store)")
    ap.add_argument("--repo", help="sync: git URL of the repository holding the data branch "
                                   "(default: NE_DATA_REPO or this checkout's origin)")
    ap.add_argument("--no-fetch", action="store_true", help="sync: do not fetch, use the existing clone")
    ap.add_argument("--include-today", action="store_true",
                    help="sync: also merge today's recorder files before 15:45 IST")
    ap.add_argument("--full", action="store_true", help="sync: re-read every file (ignore the sync state)")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if a.verbose or a.command in ("snapshot", "eod", "news", "sync")
                        else logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        stream=sys.stderr)

    from ..config import DATA_DIR
    store = a.store or DATA_DIR

    if a.command in ("snapshot", "eod", "news"):
        from .market import run
        rec = run(a.command, out=a.out, data_branch_dir=a.data_branch_dir, force=a.force, expired=not a.no_expired)
        print(json.dumps(rec, indent=1, default=str))
        return 1 if rec.get("fatal") else 0

    if a.command == "merge":
        if not a.out or not a.data_branch_dir:
            ap.error("merge needs --out (staging) and --data-branch-dir")
        from .merge import merge_tree
        print(json.dumps(merge_tree(a.out, a.data_branch_dir), indent=1, default=str))
        return 0

    from .sync import fetch_data_branch, status, sync
    src = a.data_branch_dir
    if a.command == "sync":
        if src is None:
            src = Path(store) / "cache" / "data_branch"
            if not a.no_fetch:
                f = fetch_data_branch(src, a.repo)
                if not f["ok"]:
                    print(json.dumps({"error": "could not fetch the data branch", **f}, indent=1))
                    return 1
        if not (Path(src) / "days").exists() and not (Path(src) / "MANIFEST.json").exists():
            print(json.dumps({"error": f"{src} does not look like the data branch (no days/ or MANIFEST.json)"}))
            return 1
        res = sync(src, store, include_today=a.include_today, full=a.full)
        print(json.dumps(res, indent=1, default=str))
        return 0

    if src is None:
        src = Path(store) / "cache" / "data_branch"
    print(json.dumps(status(src, store), indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
