"""Download real historical prices of expired NIFTY weekly options (Upstox Plus).

Only the contracts the paper strategy would actually have traded are fetched: for each
session, the ITM call/put (and their neighbours) of the nearest expiry with >= 1 day left,
based on that day's NIFTY open. Results: store/options_hist/<expiry>.csv (5-minute bars).
"""

from __future__ import annotations

import datetime as dt
import logging
import time
import urllib.parse
from typing import Dict, List

import pandas as pd

from ..calendar import IST
from ..config import DATA_DIR
from . import store
from .providers import UpstoxProvider

log = logging.getLogger(__name__)
OUT = DATA_DIR / "options_hist"
STEP = 50


def expired_expiries(up: UpstoxProvider) -> List[dt.date]:
    data = up._get("/v2/expired-instruments/expiries", {"instrument_key": up.NIFTY}).get("data", [])
    return sorted(dt.date.fromisoformat(d[:10]) for d in data)


def expired_contracts(up: UpstoxProvider, expiry: dt.date) -> pd.DataFrame:
    data = up._get("/v2/expired-instruments/option/contract",
                   {"instrument_key": up.NIFTY, "expiry_date": expiry.isoformat()}).get("data", [])
    df = pd.DataFrame(data)
    if df.empty:
        return df
    df = df.rename(columns={"strike_price": "strike", "instrument_type": "type"})
    return df[["instrument_key", "strike", "type"]]


def expired_candles(up: UpstoxProvider, key: str, start: dt.date, end: dt.date, interval: str = "5minute") -> pd.DataFrame:
    path = (f"/v2/expired-instruments/historical-candle/{urllib.parse.quote(key, safe='')}/"
            f"{interval}/{end.isoformat()}/{start.isoformat()}")
    import requests
    for attempt in range(6):
        try:
            c = up._get(path).get("data", {}).get("candles", [])
            break
        except requests.HTTPError as e:  # 429 = rate limit (Upstox: ~2000 requests / 30 min)
            if e.response is not None and e.response.status_code == 429 and attempt < 5:
                time.sleep(min(120, 5 * 2 ** attempt))
                continue
            raise
    if not c:
        return pd.DataFrame()
    df = pd.DataFrame(c, columns=["ts", "open", "high", "low", "close", "volume", "oi"][: len(c[0])])
    df["ts"] = pd.to_datetime(df["ts"]).dt.tz_convert(IST)
    return df.sort_values("ts")


def sessions_per_expiry(expiries: List[dt.date], days: List[dt.date]) -> Dict[dt.date, List[dt.date]]:
    """Map each expiry to the sessions that would trade it (nearest expiry >= 1 day ahead)."""
    out: Dict[dt.date, List[dt.date]] = {}
    for d in days:
        nxt = [e for e in expiries if (e - d).days >= 1]
        if nxt:
            out.setdefault(nxt[0], []).append(d)
    return out


def download(start: str = "2024-10-01", neighbours: int = 1, sleep: float = 0.9,
             max_expiries: int | None = None, mode: str = "itm") -> Dict:
    """mode 'itm': ATM +/- (neighbours+1) strikes, both types (directional buying).
    mode 'wings': OTM calls ATM+100..+600 and OTM puts ATM-100..-600 (option selling / condors)."""
    up = UpstoxProvider()
    nifty = store.load("nifty")
    days = [d.date() for d in nifty.index if d >= pd.Timestamp(start)]
    exps = expired_expiries(up)
    plan = sessions_per_expiry(exps, days)
    OUT.mkdir(parents=True, exist_ok=True)
    stats = {"expiries": 0, "contracts": 0, "bars": 0, "skipped_existing": 0, "missing": 0}
    for i, (exp, sess) in enumerate(sorted(plan.items())):
        if max_expiries and i >= max_expiries:
            break
        f = OUT / f"{exp.isoformat()}.csv"
        have = pd.read_csv(f) if f.exists() else pd.DataFrame()
        have_keys = set(have["key"]) if len(have) else set()
        try:
            contracts = expired_contracts(up, exp)
        except Exception as e:  # noqa: BLE001
            log.warning("contracts %s failed: %s", exp, e)
            continue
        if contracts.empty:
            continue
        need = set()
        for d in sess:
            o = float(nifty.loc[pd.Timestamp(d), "open"])
            atm = round(o / STEP) * STEP
            if mode == "wings":
                for k in range(2, 13):
                    need.add((atm + k * STEP, "CE"))
                    need.add((atm - k * STEP, "PE"))
            else:
                for k in range(-neighbours - 1, neighbours + 2):   # ITM strikes are atm -/+ 50
                    need.add((atm + k * STEP, "CE"))
                    need.add((atm + k * STEP, "PE"))
        pairs = list(zip(contracts["strike"].astype(float), contracts["type"]))
        mask = pd.Series([pt in need for pt in pairs], index=contracts.index)
        sel = contracts[mask & ~contracts["instrument_key"].isin(have_keys)]
        if sel.empty:
            stats["skipped_existing"] += 1
            continue
        frames = [have] if len(have) else []
        for _, c in sel.iterrows():
            try:
                df = expired_candles(up, c["instrument_key"], min(sess), max(sess))
            except Exception as e:  # noqa: BLE001
                log.warning("candles %s failed: %s", c["instrument_key"], e)
                stats["missing"] += 1
                continue
            if df.empty:
                stats["missing"] += 1
                continue
            df["strike"], df["type"], df["key"] = c["strike"], c["type"], c["instrument_key"]
            frames.append(df)
            stats["contracts"] += 1
            time.sleep(sleep)
        if len(frames) > (1 if len(have) else 0):
            all_ = pd.concat(frames)
            all_.to_csv(f, index=False)
            stats["bars"] += len(all_)
        stats["expiries"] += 1
        log.info("expiry %s: %d sessions, %d new contracts", exp, len(sess), len(frames) - (1 if len(have) else 0))
    return stats


def load_expiry(exp: dt.date) -> pd.DataFrame:
    f = OUT / f"{exp.isoformat()}.csv"
    if not f.exists():
        return pd.DataFrame()
    df = pd.read_csv(f)
    df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert(IST)
    return df
