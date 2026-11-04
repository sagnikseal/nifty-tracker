"""Scheduled-event calendar (config/events.json) mapped onto NIFTY sessions.

The 'impact session' is the first NIFTY session that can react: the same day for events
announced during NSE hours (RBI 10:00, Budget 11:00), the next session for events announced
after 15:30 IST (FOMC ~23:30, US payrolls ~18:00/19:00)."""

from __future__ import annotations

import datetime as dt
import json
from typing import Dict, List

import numpy as np
import pandas as pd

from ..config import ROOT

EVENTS_FILE = ROOT / "config" / "events.json"


def _first_friday(year: int, month: int) -> dt.date:
    d = dt.date(year, month, 1)
    return d + dt.timedelta(days=(4 - d.weekday()) % 7)


def load_events(with_nfp: bool = True, nfp_from: int = 2014) -> List[Dict]:
    ev = json.loads(EVENTS_FILE.read_text(encoding="utf-8")).get("events", []) if EVENTS_FILE.exists() else []
    out = []
    for e in ev:
        d = dt.date.fromisoformat(e["date"])
        t = e.get("ist_time", "10:00")
        after_close = t.endswith("+1") or dt.time.fromisoformat(t.split("+")[0]) >= dt.time(15, 30)
        out.append({"type": e["type"], "date": d, "after_close": after_close})
    if with_nfp:   # US payrolls: first Friday (approximation; BLS occasionally moves it)
        for y in range(nfp_from, dt.date.today().year + 1):
            for m in range(1, 13):
                out.append({"type": "us_nfp", "date": _first_friday(y, m), "after_close": True})
    return out


def impact_sessions(sessions: pd.DatetimeIndex, events: List[Dict] | None = None) -> Dict[pd.Timestamp, List[str]]:
    """Map each event to the first NIFTY session that can react to it."""
    events = events if events is not None else load_events()
    s = np.array(sessions.values, dtype="datetime64[ns]")
    out: Dict[pd.Timestamp, List[str]] = {}
    for e in events:
        ts = np.datetime64(pd.Timestamp(e["date"]) + (pd.Timedelta(days=1) if e["after_close"] else pd.Timedelta(0)))
        i = int(np.searchsorted(s, ts, side="left"))
        if i < len(s):
            out.setdefault(pd.Timestamp(s[i]), []).append(e["type"])
    return out


def event_features(sessions: pd.DatetimeIndex, events: List[Dict] | None = None) -> pd.DataFrame:
    """Per session: event flags for today's impact and for the next session (known in advance)."""
    # one extra business day so the last row (today, live) still knows tomorrow's schedule
    ext = sessions.append(pd.DatetimeIndex([sessions[-1] + pd.offsets.BDay(1)])) if len(sessions) else sessions
    imp = impact_sessions(ext, events)
    X = pd.DataFrame(index=ext)
    types = sorted({t for v in imp.values() for t in v})
    for t in types:
        X[f"ev_{t}"] = [float(t in imp.get(d, [])) for d in ext]
        X[f"ev_{t}_next"] = X[f"ev_{t}"].shift(-1).fillna(0.0)   # schedules are public in advance
    X = X.iloc[:-1] if len(ext) > len(sessions) else X
    X["ev_any"] = X[[f"ev_{t}" for t in types]].max(axis=1) if types else 0.0
    X["ev_any_next"] = X[[f"ev_{t}_next" for t in types]].max(axis=1) if types else 0.0
    return X
