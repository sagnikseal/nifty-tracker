"""NSE calendar helpers: expiry days, sessions, IST time."""

from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")

# NIFTY weekly expiry moved from Thursday to Tuesday from the 2-Sep-2025 expiry
# (NSE circular, Aug-2025). Monthly = last Tuesday of month.
TUESDAY_EXPIRY_FROM = dt.date(2025, 9, 1)

MARKET_OPEN = dt.time(9, 15)
MARKET_CLOSE = dt.time(15, 30)


def now_ist() -> dt.datetime:
    return dt.datetime.now(IST)


def weekly_expiry_weekday(on: dt.date) -> int:
    return 1 if on >= TUESDAY_EXPIRY_FROM else 3  # Mon=0 .. Tue=1, Thu=3


def next_weekly_expiry(on: dt.date, min_days: int = 0) -> dt.date:
    """Nearest weekly expiry on/after `on` with at least `min_days` calendar days left.
    Holiday shifts are ignored here; live trading uses the broker's contract list."""
    d = on
    while True:
        wd = weekly_expiry_weekday(d)
        delta = (wd - d.weekday()) % 7
        exp = d + dt.timedelta(days=delta)
        if (exp - on).days >= min_days:
            return exp
        d = exp + dt.timedelta(days=1)


def is_expiry_day(on: dt.date) -> bool:
    return on.weekday() == weekly_expiry_weekday(on)


def days_to_expiry(on: dt.date) -> int:
    return (next_weekly_expiry(on) - on).days


def year_fraction(start: dt.datetime, end: dt.datetime) -> float:
    secs = max((end - start).total_seconds(), 60.0)
    return secs / (365.0 * 24 * 3600)


def is_market_open(ts: dt.datetime | None = None) -> bool:
    ts = ts or now_ist()
    if ts.weekday() >= 5:
        return False
    return MARKET_OPEN <= ts.time() <= MARKET_CLOSE


def parse_hhmm(s: str) -> dt.time:
    h, m = s.split(":")
    return dt.time(int(h), int(m))
