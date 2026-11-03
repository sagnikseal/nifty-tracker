"""Local CSV store with incremental refresh. Works offline once populated."""

from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from ..config import DATA_DIR
from .providers import YAHOO_SYMBOLS, YahooProvider

log = logging.getLogger(__name__)
RAW = DATA_DIR / "raw"


def _path(name: str) -> Path:
    return RAW / f"{name}.csv"


def load(name: str) -> pd.DataFrame:
    p = _path(name)
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p, index_col=0, parse_dates=True)
    df.index = pd.DatetimeIndex(df.index).normalize()
    df.index.name = "date"
    return df.sort_index()


def save(name: str, df: pd.DataFrame) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    df.sort_index().to_csv(_path(name))


def refresh_daily(start: str = "2007-01-01", provider: Optional[YahooProvider] = None,
                  names: Optional[list] = None) -> Dict[str, int]:
    """Download / append daily bars for every series in YAHOO_SYMBOLS."""
    provider = provider or YahooProvider()
    out = {}
    for name, sym in YAHOO_SYMBOLS.items():
        if names and name not in names:
            continue
        old = load(name)
        fetch_from = start
        if len(old):
            fetch_from = (old.index.max() - pd.Timedelta(days=10)).date().isoformat()
        try:
            new = provider.bars(sym, start=fetch_from)
        except Exception as e:  # noqa: BLE001
            log.warning("refresh %s failed: %s", name, e)
            out[name] = len(old)
            continue
        new.index = pd.DatetimeIndex(new.index).normalize()
        df = pd.concat([old, new]) if len(old) else new
        df = df[~df.index.duplicated(keep="last")].sort_index()
        # Drop today's bar while the Indian session is still running (incomplete close).
        today = pd.Timestamp(dt.date.today())
        if name in ("nifty", "indiavix", "banknifty") and len(df) and df.index[-1] == today:
            now = dt.datetime.now(dt.timezone(dt.timedelta(hours=5, minutes=30)))
            if now.time() < dt.time(15, 35):
                df = df.iloc[:-1]
        save(name, df)
        out[name] = len(df)
    return out


def load_all() -> Dict[str, pd.DataFrame]:
    return {n: load(n) for n in YAHOO_SYMBOLS}


def import_legacy_repo_json(path: Path) -> pd.DataFrame:
    """Seed NIFTY daily bars from the legacy data/nifty_data.json (offline bootstrap)."""
    import json
    js = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = [{"date": e["date"], "open": e["open"], "high": e["high"], "low": e["low"],
             "close": e["close"], "volume": e.get("volume", 0)} for e in js["data"] if e.get("close", 0) > 0]
    df = pd.DataFrame(rows).set_index("date")
    df.index = pd.DatetimeIndex(df.index)
    return df.sort_index()
