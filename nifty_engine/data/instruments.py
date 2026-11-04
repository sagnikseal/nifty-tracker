"""Upstox instrument master (public file, no token needed): resolves instrument keys for NIFTY
futures, weekly options and NSE stocks. Cached once per day under store/instruments/."""

from __future__ import annotations

import datetime as dt
import gzip
import json
import logging
from typing import Dict, List, Optional

import pandas as pd
import requests

from ..calendar import IST, now_ist
from ..config import DATA_DIR
from .providers import UA

log = logging.getLogger(__name__)
URL = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz"
CACHE = DATA_DIR / "instruments"

# Largest NIFTY 50 weights (they move the index most). Edit `breadth_symbols` in
# config/engine.json `extra` to track a different or longer list.
HEAVYWEIGHTS = ["HDFCBANK", "RELIANCE", "ICICIBANK", "INFY", "BHARTIARTL", "LT", "TCS", "ITC", "SBIN",
                "AXISBANK", "KOTAKBANK", "M&M", "HINDUNILVR", "BAJFINANCE", "SUNPHARMA"]


def load_master(refresh: bool = False) -> pd.DataFrame:
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"NSE_{now_ist().date().isoformat()}.json.gz"
    if refresh or not f.exists():
        r = requests.get(URL, headers=UA, timeout=60)
        r.raise_for_status()
        f.write_bytes(r.content)
        for old in CACHE.glob("NSE_*.json.gz"):
            if old != f:
                old.unlink(missing_ok=True)
    rows = json.loads(gzip.decompress(f.read_bytes()))
    df = pd.DataFrame(rows)
    if "expiry" in df:
        exp = pd.to_numeric(df["expiry"], errors="coerce")
        df["expiry_date"] = pd.to_datetime(exp, unit="ms", utc=True).dt.tz_convert(IST).dt.date
    return df


def _underlying(df: pd.DataFrame, name: str) -> pd.Series:
    cols = [c for c in ("underlying_symbol", "asset_symbol", "name") if c in df]
    m = pd.Series(False, index=df.index)
    for c in cols:
        m |= df[c].astype(str).str.upper().eq(name)
    return m


def nifty_futures(df: Optional[pd.DataFrame] = None) -> List[Dict]:
    df = df if df is not None else load_master()
    f = df[(df.get("segment") == "NSE_FO") & (df.get("instrument_type") == "FUT") & _underlying(df, "NIFTY")]
    f = f.sort_values("expiry_date")
    return [{"expiry": r["expiry_date"], "key": r["instrument_key"], "symbol": r.get("trading_symbol")}
            for _, r in f.iterrows() if r["expiry_date"] and r["expiry_date"] >= now_ist().date()]


def nifty_options(expiry: dt.date, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    df = df if df is not None else load_master()
    o = df[(df.get("segment") == "NSE_FO") & df.get("instrument_type").isin(["CE", "PE"]) & _underlying(df, "NIFTY")
           & (df["expiry_date"] == expiry)]
    return o.rename(columns={"strike_price": "strike", "instrument_type": "type"})[["instrument_key", "strike", "type"]]


def option_expiries(df: Optional[pd.DataFrame] = None) -> List[dt.date]:
    df = df if df is not None else load_master()
    o = df[(df.get("segment") == "NSE_FO") & df.get("instrument_type").isin(["CE", "PE"]) & _underlying(df, "NIFTY")]
    return sorted(d for d in o["expiry_date"].dropna().unique() if d >= now_ist().date())


def equity_keys(symbols: List[str], df: Optional[pd.DataFrame] = None) -> Dict[str, str]:
    df = df if df is not None else load_master()
    eq = df[(df.get("segment") == "NSE_EQ") & df["trading_symbol"].isin(symbols)]
    return dict(zip(eq["trading_symbol"], eq["instrument_key"]))
