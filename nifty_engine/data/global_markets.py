"""Related markets, with the exact moment each number becomes known in IST.

  * Overnight futures (ES/NQ/crude/gold/BTC/USDINR, hourly bars): their move from NIFTY's close
    (15:30 IST, day t-1) to 09:00 IST (day t) is fully known before the 09:15 open. This is the
    free stand-in for GIFT Nifty, which no free feed carries.
  * ADRs of NIFTY heavyweights and the US-listed INDA ETF: priced in New York after NSE closes, so
    their premium over the NSE close (in INR, relative to its own 60-day median, which cancels the
    ADR ratio) measures how US investors repriced India overnight. Known at 01:30-02:30 IST.
"""

from __future__ import annotations

import datetime as dt
import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

from ..calendar import IST
from ..config import DATA_DIR
from .providers import YAHOO_HOURLY, YahooProvider

log = logging.getLogger(__name__)
RAW_1H = DATA_DIR / "raw_1h"

ADR_PAIRS = [("hdb", "hdfcbank"), ("ibn", "icicibank"), ("infy_adr", "infy_ns"), ("wit", "wipro_ns"),
             ("rdy", "drreddy_ns")]


def refresh_hourly(provider: Optional[YahooProvider] = None) -> Dict[str, int]:
    """Append hourly bars (Yahoo serves ~730 days); stored with tz-aware IST bar-start timestamps."""
    provider = provider or YahooProvider()
    RAW_1H.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, sym in YAHOO_HOURLY.items():
        f = RAW_1H / f"{name}.csv"
        old = load_hourly(name)
        start = (old.index.max() - pd.Timedelta(days=5)).date().isoformat() if len(old) else \
            (dt.date.today() - dt.timedelta(days=725)).isoformat()
        try:
            new = provider.bars(sym, start=start, interval="60m")
        except Exception as e:  # noqa: BLE001
            log.warning("hourly %s failed: %s", name, e)
            out[name] = len(old)
            continue
        new.index = pd.DatetimeIndex(new.index).tz_convert(IST)
        new = new[new.index.second == 0]          # Yahoo appends the still-forming bar (e.g. 18:00:26)
        df = pd.concat([old, new]) if len(old) else new
        df = df[~df.index.duplicated(keep="last")].sort_index()
        df.to_csv(f)
        out[name] = len(df)
    return out


def load_hourly(name: str) -> pd.DataFrame:
    f = RAW_1H / f"{name}.csv"
    if not f.exists():
        return pd.DataFrame()
    df = pd.read_csv(f, index_col=0)
    df.index = pd.to_datetime(df.index, utc=True).tz_convert(IST)
    return df[df.index.second == 0].sort_index()


def _price_asof(bars: pd.DataFrame, when: pd.Timestamp, bar_minutes: int = 60, _ends=None) -> float:
    """Close of the last bar that had *finished* by `when` (Yahoo stamps bars by their start)."""
    ends = _ends if _ends is not None else _bar_ends(bars, bar_minutes)
    w = pd.Timestamp(when).as_unit("ns").value          # pandas 3 may store us; compare in ns
    i = int(np.searchsorted(ends, w, side="right")) - 1
    if i < 0 or (w - ends[i]) > pd.Timedelta(days=4).value:
        return np.nan
    return float(bars["close"].iloc[i])


def _bar_ends(bars: pd.DataFrame, bar_minutes: int = 60) -> np.ndarray:
    return (bars.index + pd.Timedelta(minutes=bar_minutes)).as_unit("ns").asi8


def overnight_features(nifty_dates: pd.DatetimeIndex, names=None, cutoff: dt.time = dt.time(9, 0)) -> pd.DataFrame:
    """For each NIFTY session t: log move of each hourly series from 15:30 IST of the previous
    NIFTY session to `cutoff` IST on t."""
    names = names or list(YAHOO_HOURLY)
    X = pd.DataFrame(index=nifty_dates)
    for name in names:
        b = load_hourly(name)
        if b.empty:
            continue
        ends = _bar_ends(b)
        vals = []
        for i, d in enumerate(nifty_dates):
            if i == 0:
                vals.append(np.nan)
                continue
            t0 = pd.Timestamp(dt.datetime.combine(nifty_dates[i - 1].date(), dt.time(15, 30)), tz=IST)
            t1 = pd.Timestamp(dt.datetime.combine(d.date(), cutoff), tz=IST)
            if t1 < b.index[0] + pd.Timedelta(hours=2):
                vals.append(np.nan)
                continue
            p0, p1 = _price_asof(b, t0, _ends=ends), _price_asof(b, t1, _ends=ends)
            vals.append(np.log(p1 / p0) if p0 == p0 and p1 == p1 and p0 > 0 else np.nan)
        X[f"on_{name}"] = vals
    return X


def adr_features(raw: Dict[str, pd.DataFrame], nifty_dates: pd.DatetimeIndex, align) -> pd.DataFrame:
    """ADR / INDA premium vs the NSE close, de-meaned by its own 60-day median. `align` is
    features._align_backward (availability = US date + 1 day)."""
    X = pd.DataFrame(index=nifty_dates)
    fx = raw.get("usdinr")
    if fx is None or fx.empty:
        return X
    fxc = fx["close"]
    devs = []
    for adr, nse in ADR_PAIRS:
        a, s = raw.get(adr), raw.get(nse)
        if a is None or s is None or a.empty or s.empty:
            continue
        j = pd.concat([a["close"].rename("a"), s["close"].rename("s"), fxc.rename("fx")], axis=1).dropna()
        x = np.log(j["a"] * j["fx"]) - np.log(j["s"])
        dev = (x - x.rolling(60, min_periods=30).median()).clip(-0.1, 0.1)
        X[f"adr_dev_{adr}"] = align(nifty_dates, dev, 1)
        devs.append(f"adr_dev_{adr}")
    if devs:
        X["adr_dev_mean"] = X[devs].mean(axis=1)
    inda, nf = raw.get("inda"), raw.get("nifty")
    if inda is not None and len(inda) and nf is not None and len(nf):
        j = pd.concat([inda["close"].rename("i"), nf["close"].rename("n"), fxc.rename("fx")], axis=1).dropna()
        x = np.log(j["i"] * j["fx"]) - np.log(j["n"])
        X["inda_dev"] = align(nifty_dates, (x - x.rolling(60, min_periods=30).median()).clip(-0.1, 0.1), 1)
    return X
