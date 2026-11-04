"""Trade plan and decision context shared by every strategy book."""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from ..config import EngineConfig
from ..options import STRIKE_STEP, Leg


@dataclass
class TradePlan:
    book: str
    day: dt.date
    entry_time: dt.time
    direction: int                       # +1 bullish, -1 bearish, 0 market-neutral
    style: str
    legs: List[Leg]
    expiry: dt.date
    exit_time: dt.time = dt.time(15, 15)
    exit_day: Optional[dt.date] = None   # positional books close on this session (default: same day)
    stop_pct: Optional[float] = None     # debit: exit when value falls this % below entry
    target_pct: Optional[float] = None   # debit: exit when value rises this % above entry
    credit_stop_mult: Optional[float] = None    # credit: exit when the loss reaches mult x credit
    credit_target_pct: Optional[float] = None   # credit: exit after capturing this % of the credit
    underlying_stop: Optional[float] = None     # exit when NIFTY trades through this level
    underlying_target: Optional[float] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def rules(self) -> Dict[str, Any]:
        r = {k: v for k, v in asdict(self).items() if k in (
            "stop_pct", "target_pct", "credit_stop_mult", "credit_target_pct", "underlying_stop", "underlying_target")}
        r.update({"exit_time": self.exit_time.strftime("%H:%M"),
                  "exit_day": (self.exit_day or self.day).isoformat(), "direction": self.direction})
        return r


@dataclass
class Context:
    """Everything a book may look at when deciding. Live and research fill the same fields."""
    cfg: EngineConfig
    day: dt.date
    now: dt.time
    spot: float
    vix: float                              # India VIX (previous close in research, live in paper)
    expiries: List[dt.date]
    p_up: Optional[float] = None            # champion daily model (calibrated)
    p_range: Optional[float] = None         # range model: P(NIFTY stays inside the condor band)
    regime: Dict[str, float] = field(default_factory=dict)      # har_vol_fc, vrp_har, vix_pct_1y ...
    events_today: List[str] = field(default_factory=list)
    events_next: List[str] = field(default_factory=list)
    chain: Dict[str, float] = field(default_factory=dict)
    intraday: Optional[pd.DataFrame] = None  # 5-min bars closed so far (several sessions, for SMC)
    brief: Optional[Dict[str, Any]] = None   # local-LLM pre-open brief
    gap: Optional[float] = None              # log(open / previous close)
    sessions_left_in_month: Optional[int] = None   # including today
    session_of_month: Optional[int] = None         # 1 = first session of the month
    next_sessions: List[dt.date] = field(default_factory=list)

    @property
    def sigma_day(self) -> float:
        return max(self.vix, 8.0) / 100 / math.sqrt(252)

    def expiry(self, min_days: int = 1, after: Optional[dt.date] = None) -> Optional[dt.date]:
        ref = after or self.day
        ok = [e for e in sorted(self.expiries) if (e - ref).days >= min_days]
        return ok[0] if ok else None


def strike(x: float, mode: str = "round") -> float:
    f = {"round": round, "up": math.ceil, "down": math.floor}[mode]
    return float(f(x / STRIKE_STEP) * STRIKE_STEP)
