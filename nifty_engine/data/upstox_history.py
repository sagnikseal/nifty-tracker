"""Download all intraday history Upstox offers that can affect NIFTY.

  * Bank Nifty and the NIFTY heavyweight stocks: 5-minute candles since January 2022
    (V3 historical candles: minute data starts in January 2022; every plan).
  * A continuous near-month NIFTY FUTURES series with volume and open interest: each expired
    monthly contract's 5-minute candles (Upstox Plus "expired instruments"), stitched at expiry
    (the next contract takes over the day after expiry), plus the live current-month contract.
    Futures add what the cash index lacks: traded volume, open interest and the futures basis.

Writes store/intraday/<series>_5m.csv (index ts, columns open,high,low,close,volume,oi).
Resumable: series/contracts already on disk are skipped.
"""

from __future__ import annotations

import datetime as dt
import logging
import time
from typing import Dict, List

import pandas as pd

from ..calendar import IST, now_ist
from ..config import DATA_DIR
from .providers import UpstoxProvider

log = logging.getLogger(__name__)
DIR = DATA_DIR / "intraday"
START = dt.date(2022, 1, 3)
TOP_STOCKS = ["HDFCBANK", "RELIANCE", "ICICIBANK", "INFY", "BHARTIARTL", "LT", "TCS", "ITC", "SBIN", "AXISBANK"]


def _save(name: str, df: pd.DataFrame) -> int:
    DIR.mkdir(parents=True, exist_ok=True)
    p = DIR / f"{name}.csv"
    df = df.copy()
    for c in ("open", "high", "low", "close", "volume", "oi"):
        if c not in df:
            df[c] = 0.0
    df = df[["open", "high", "low", "close", "volume", "oi"]]
    df.index = pd.to_datetime(df.index, utc=True).tz_convert(IST)
    if p.exists():
        old = pd.read_csv(p, index_col=0)
        old.index = pd.to_datetime(old.index, utc=True).tz_convert(IST)
        df = pd.concat([old, df])
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df[(df.index.time >= dt.time(9, 15)) & (df.index.time <= dt.time(15, 29))]
    df.index.name = "ts"
    df.to_csv(p)
    return len(df)


def _last_ts(name: str):
    p = DIR / f"{name}.csv"
    if not p.exists():
        return None
    tail = pd.read_csv(p, usecols=[0]).iloc[-1, 0]
    return pd.Timestamp(tail).tz_convert(IST)


def fetch_index_and_stocks(up: UpstoxProvider, stocks: List[str] = TOP_STOCKS) -> Dict[str, int]:
    from .instruments import equity_keys
    out = {}
    targets = {"banknifty_5m": UpstoxProvider.BANKNIFTY}
    try:
        keys = equity_keys(stocks)
        targets.update({f"{s.lower().replace('&', '')}_5m": k for s, k in keys.items()})
    except Exception as e:  # noqa: BLE001
        log.warning("instrument master failed (stocks skipped): %s", e)
    for name, key in targets.items():
        last = _last_ts(name)
        start = (last.date() + dt.timedelta(days=1)) if last is not None else START
        if start > now_ist().date():
            out[name] = -1
            continue
        try:
            df = up.candles(key, "minutes", 5, start, now_ist().date())
            out[name] = _save(name, df) if len(df) else 0
            log.info("%s: %s rows", name, out[name])
        except Exception as e:  # noqa: BLE001
            log.warning("%s failed: %s", name, e)
            out[name] = 0
    return out


def fetch_nifty_futures(up: UpstoxProvider) -> Dict[str, int]:
    """Continuous near-month NIFTY futures (5-minute, with volume and OI)."""
    from .expired_options import expired_candles, expired_expiries
    name = "niftyfut_5m"
    have = _last_ts(name)
    exps = expired_expiries(up)
    # monthly expiries = the last expiry of each calendar month in the list
    monthly = sorted({max(e for e in exps if (e.year, e.month) == ym) for ym in {(e.year, e.month) for e in exps}})
    frames, prev, n_contracts = [], None, 0
    for e in monthly:
        if e < START:
            prev = e
            continue
        if have is not None and e <= have.date():
            prev = e
            continue
        try:
            js = up._get("/v2/expired-instruments/future/contract",
                         {"instrument_key": UpstoxProvider.NIFTY, "expiry_date": e.isoformat()}).get("data", [])
        except Exception as ex:  # noqa: BLE001
            log.warning("future contract %s failed: %s", e, ex)
            prev = e
            continue
        if not js:
            prev = e
            continue
        key = js[0].get("instrument_key")
        lo = (prev + dt.timedelta(days=1)) if prev else e - dt.timedelta(days=35)
        try:
            df = expired_candles(up, key, lo, e, interval="5minute")
        except Exception as ex:  # noqa: BLE001
            log.warning("future candles %s failed: %s", key, ex)
            prev = e
            continue
        if len(df):
            df = df.set_index("ts")
            df = df[(df.index.date >= lo) & (df.index.date <= e)]
            frames.append(df)
            n_contracts += 1
            log.info("NIFTY FUT %s: %d bars", e, len(df))
        prev = e
        time.sleep(0.5)
    # live contract(s): from the day after the last expired monthly expiry to today
    try:
        from .instruments import nifty_futures
        live = nifty_futures()
        if live and prev:
            df = up.candles(live[0]["key"], "minutes", 5, prev + dt.timedelta(days=1), now_ist().date())
            if len(df):
                frames.append(df)
    except Exception as ex:  # noqa: BLE001
        log.warning("current future failed: %s", ex)
    rows = _save(name, pd.concat(frames)) if frames else (0 if have is None else -1)
    return {name: rows, "contracts": n_contracts}


def update_recent(up: UpstoxProvider) -> Dict[str, int]:
    """Daily top-up (08:30 and after the close): Bank Nifty, stocks, and the live near-month future."""
    out = fetch_index_and_stocks(up)
    try:
        from .instruments import nifty_futures
        live = nifty_futures()
        last = _last_ts("niftyfut_5m")
        if live and last is not None:
            df = up.candles(live[0]["key"], "minutes", 5, last.date() + dt.timedelta(days=1), now_ist().date())
            if now_ist().time() >= dt.time(15, 35):
                t = up.intraday_today(live[0]["key"], "minutes", 5)
                df = pd.concat([df, t]) if len(t) else df
            out["niftyfut_5m"] = _save("niftyfut_5m", df) if len(df) else -1
    except Exception as e:  # noqa: BLE001
        log.warning("futures top-up failed: %s", e)
    return out


def fetch_all() -> Dict:
    up = UpstoxProvider()
    if not up.available:
        raise RuntimeError("Set UPSTOX_ANALYTICS_TOKEN in .env (Upstox Plus needed for expired futures)")
    out = fetch_index_and_stocks(up)
    try:
        out.update(fetch_nifty_futures(up))
    except Exception as e:  # noqa: BLE001
        out["niftyfut_error"] = str(e)[:200]
    return out
