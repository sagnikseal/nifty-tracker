"""Keep store/intraday/{nifty_5m, indiavix_5m, nifty_1m}.csv current (run at 08:40 and after the close).

The movement model needs the previous sessions' bars for its context features (previous close,
high/low, volatility), so a stale store would silently distort live predictions. Source: Upstox
(v3 candles) when a token is set, otherwise Yahoo (5-minute bars ~60 days back, 1-minute ~7 days).
"""

from __future__ import annotations

import datetime as dt
import logging
from typing import Dict

import pandas as pd

from ..calendar import IST, now_ist
from ..config import DATA_DIR
from ..data.providers import UpstoxProvider, YahooProvider

log = logging.getLogger(__name__)
DIR = DATA_DIR / "intraday"
SERIES = {"nifty_5m": ("NIFTY", "^NSEI", 5), "indiavix_5m": ("VIX", "^INDIAVIX", 5), "nifty_1m": ("NIFTY", "^NSEI", 1)}


def _merge_save(name: str, new: pd.DataFrame) -> int:
    p = DIR / f"{name}.csv"
    DIR.mkdir(parents=True, exist_ok=True)
    cols = ["open", "high", "low", "close", "volume", "oi"]
    new = new.copy()
    for c in cols:
        if c not in new:
            new[c] = 0
    new = new[cols]
    new.index = pd.to_datetime(new.index, utc=True).tz_convert(IST)
    now = now_ist()
    # only completed sessions (today's bars are added after the close)
    new = new[(new.index.date < now.date()) | (now.time() >= dt.time(15, 35))]
    new = new[(new.index.time >= dt.time(9, 15)) & (new.index.time <= dt.time(15, 29))]
    if p.exists():
        old = pd.read_csv(p, index_col=0)
        old.index = pd.to_datetime(old.index, utc=True).tz_convert(IST)
        df = pd.concat([old, new])
    else:
        df = new
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df.index.name = "ts"
    df.to_csv(p)
    return len(new)


def refresh_intraday(days: int = 10) -> Dict[str, int]:
    up = UpstoxProvider()
    yh = YahooProvider()
    today = now_ist().date()
    out = {}
    for name, (ukey, ysym, mins) in SERIES.items():
        df = pd.DataFrame()
        if up.available:
            try:
                key = getattr(UpstoxProvider, ukey)
                df = up.candles(key, "minutes", mins, today - dt.timedelta(days=days), today)
                if now_ist().time() >= dt.time(15, 35):
                    t = up.intraday_today(key, "minutes", mins)
                    df = pd.concat([df, t]) if len(t) else df
            except Exception as e:  # noqa: BLE001
                log.warning("upstox %s failed: %s", name, e)
        if df.empty:
            try:
                back = min(days, 6) if mins == 1 else days
                df = yh.bars(ysym, start=(today - dt.timedelta(days=back)).isoformat(), interval=f"{mins}m")
            except Exception as e:  # noqa: BLE001
                log.warning("yahoo %s failed: %s", name, e)
        out[name] = _merge_save(name, df) if len(df) else 0
    if up.available:                      # Bank Nifty, heavyweights, NIFTY futures
        try:
            from ..data.upstox_history import update_recent
            out.update(update_recent(up))
        except Exception as e:  # noqa: BLE001
            log.warning("series top-up failed: %s", e)
    return out
