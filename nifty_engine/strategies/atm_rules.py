"""The user's ATM option rules, as one pure state machine shared by the historical replay and the
live paper trader (so the backtest tests exactly what trades live).

Rules (defaults from config, all in rupees for the whole position):
  * Entry: buy 1 lot of the at-the-money call (bullish) or put (bearish) when the movement model's
    P(up by 15:15) is >= entry_threshold (or <= 1 - entry_threshold), between entry_start and
    entry_end. At most `max_trades_per_day` entries.
  * Stop: if the open loss reaches stop_inr (Rs 2,000), exit immediately (checked on every price).
  * Target: when the open profit reaches target_inr (Rs 5,000):
      - if the model's probability that NIFTY keeps moving our way over the next 30 minutes is
        > extend_prob (80%), hold for another extend_minutes (30) and re-check then (repeat);
      - otherwise exit.
  * Against the prediction: if NIFTY closes against the trade's direction on `against_bars` (2)
    consecutive 5-minute bars, exit and keep whatever profit there is. `against_scope` decides when
    this applies: "in_profit" (default: only while the trade is in profit, matching "make peace
    with the profit"), "always", or "after_target" (only after Rs 5,000 was reached).
  * Time exit at exit_time (15:15). Optional: exit on a strong adverse headline (learned impact),
    only once the news-impact model has passed its pre-registered test.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import asdict, dataclass, field
from typing import Dict, Optional


EPS = 0.005    # half a paisa: floating-point tolerance for the rupee thresholds


def _t(s: str) -> dt.time:
    h, m = s.split(":")
    return dt.time(int(h), int(m))


@dataclass
class ATMRules:
    target_inr: float = 5000.0
    stop_inr: float = 2000.0
    extend_prob: float = 0.80
    extend_minutes: int = 30
    against_bars: int = 2
    against_scope: str = "in_profit"        # in_profit | always | after_target
    entry_threshold: float = 0.60
    entry_start: str = "09:20"
    entry_end: str = "13:30"
    exit_time: str = "15:15"
    lots: int = 1
    lot_size: int = 65
    max_trades_per_day: int = 1
    news_exit_bps: Optional[float] = None   # e.g. 8.0: exit if a new headline's learned 30-min impact
                                            # is at least this many bps against the trade

    @classmethod
    def from_cfg(cls, cfg) -> "ATMRules":
        r = cls(lot_size=int(cfg.lot_size))
        for k in asdict(r):
            v = getattr(cfg, f"atm_{k}", None)
            if v is not None:
                setattr(r, k, type(getattr(r, k))(v) if getattr(r, k) is not None else v)
        return r

    @property
    def qty(self) -> int:
        return self.lots * self.lot_size

    def to_public(self) -> Dict:
        return {"target_inr": self.target_inr, "stop_inr": self.stop_inr, "extend_prob": self.extend_prob,
                "extend_minutes": self.extend_minutes, "against_bars": self.against_bars,
                "against_scope": self.against_scope, "entry_threshold": self.entry_threshold,
                "entry_window": [self.entry_start, self.entry_end], "exit_time": self.exit_time,
                "lots": self.lots, "news_exit_bps": self.news_exit_bps}

    # ------------------------------------------------------------------ entry
    def entry_direction(self, p_close: Optional[float], now: dt.datetime, trades_today: int) -> int:
        if p_close is None or p_close != p_close or trades_today >= self.max_trades_per_day:
            return 0
        if not (_t(self.entry_start) <= now.time() <= _t(self.entry_end)):
            return 0
        if p_close >= self.entry_threshold:
            return 1
        if p_close <= 1 - self.entry_threshold:
            return -1
        return 0


@dataclass
class PositionState:
    direction: int                   # +1 long call, -1 long put
    entry_price: float               # per unit, after costs of crossing the spread
    qty: int
    entry_ts: dt.datetime
    last_spot_close: float           # NIFTY at entry (reference for the first 5-min bar)
    target_hit: bool = False
    extend_until: Optional[dt.datetime] = None
    extensions: int = 0
    against_count: int = 0
    peak_pnl: float = 0.0
    log: list = field(default_factory=list)

    def pnl(self, price: float) -> float:
        return (price - self.entry_price) * self.qty

    def p_dir(self, p_up: Optional[float]) -> Optional[float]:
        if p_up is None or p_up != p_up:
            return None
        return p_up if self.direction > 0 else 1 - p_up

    def to_public(self, rules: ATMRules, price: Optional[float] = None) -> Dict:
        pnl = self.pnl(price) if price is not None else None
        if self.extend_until is not None:
            state, note = "EXTENDED", f"target reached; holding until {self.extend_until.strftime('%H:%M')} (P > {rules.extend_prob:.0%})"
        else:
            state, note = "WAITING_TARGET", f"{self.against_count} of {rules.against_bars} consecutive 5-min closes against the trade"
        return {"target_inr": rules.target_inr, "stop_inr": rules.stop_inr, "target_hit": self.target_hit,
                "extend_until": self.extend_until.isoformat() if self.extend_until else None,
                "extensions": self.extensions, "against_count": self.against_count,
                "against_limit": rules.against_bars, "rule_state": state, "rule_note": note,
                "pnl_inr": None if pnl is None else round(pnl, 2), "peak_pnl": round(self.peak_pnl, 2)}


# ---------------------------------------------------------------------- transitions
def on_price(rules: ATMRules, st: PositionState, price: float, now: dt.datetime,
             p_up_30m: Optional[float]) -> Optional[str]:
    """Every price update (tick / 1-minute bar). Returns an exit reason or None.
    `price` = the price you could sell at now (bid, or mid minus spread and slippage)."""
    pnl = st.pnl(price)
    st.peak_pnl = max(st.peak_pnl, pnl)
    if pnl <= -rules.stop_inr + EPS:
        return "stop"
    if now.time() >= _t(rules.exit_time):
        return "time"
    if not st.target_hit and pnl >= rules.target_inr - EPS:
        st.target_hit = True
        return _decide_extension(rules, st, now, p_up_30m, first=True)
    if st.extend_until is not None and now >= st.extend_until:
        return _decide_extension(rules, st, now, p_up_30m, first=False)
    return None


def _decide_extension(rules, st, now, p_up_30m, first: bool) -> Optional[str]:
    pd_ = st.p_dir(p_up_30m)
    if pd_ is not None and pd_ > rules.extend_prob:
        st.extend_until = now + dt.timedelta(minutes=rules.extend_minutes)
        st.extensions += 1
        st.log.append((now.isoformat(), f"extend (P={pd_:.2f})"))
        return None
    st.extend_until = None
    return "target" if first else "extension_end"


def on_bar_close(rules: ATMRules, st: PositionState, spot_close: float, price: float,
                 now: dt.datetime) -> Optional[str]:
    """At every 5-minute bar close (NIFTY spot). Counts consecutive closes against the trade."""
    against = (spot_close - st.last_spot_close) * st.direction < 0
    st.against_count = st.against_count + 1 if against else 0
    st.last_spot_close = spot_close
    pnl = st.pnl(price)
    scope = rules.against_scope
    in_scope = scope == "always" or (scope == "in_profit" and pnl > 0) or (scope == "after_target" and st.target_hit)
    if st.against_count >= rules.against_bars and in_scope:
        return "against"
    return None


def on_headline(rules: ATMRules, st: PositionState, exp_bps_30m: Optional[float], validated: bool) -> Optional[str]:
    """Reaction rule: exit on a strong headline whose learned impact is against the trade."""
    if rules.news_exit_bps is None or not validated or exp_bps_30m is None:
        return None
    if exp_bps_30m * st.direction <= -abs(rules.news_exit_bps):
        return "news"
    return None
