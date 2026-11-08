"""The research strategy books. Rules and parameters are fixed here *before* testing (pre-registered);
the research harness reports every book, including the ones that lose.

| book            | when                         | idea                                                     |
|-----------------|------------------------------|----------------------------------------------------------|
| directional_itm | 09:20                        | daily model: buy 1 ITM option on confident days          |
| debit_spread    | 09:20, options rich          | same signal, ATM/OTM debit spread when IV is high        |
| credit_spread   | 09:20, VRP > 0               | sell the OTM spread on the side the model fades          |
| iron_condor     | 09:20, VRP > 0, range model  | intraday short condor at k x sigma                       |
| iron_fly_0dte   | 09:45 on expiry day          | short ATM straddle hedged with wings, 0DTE theta         |
| event_straddle  | 15:10 before RBI/FOMC/Budget | long ATM straddle into the event, exit 10:30 next day    |
| turn_of_month   | 15:10, 2nd-last session      | long ITM call over the turn of the month (TD-2 -> TD+3)  |
| smc_intraday    | 5-min bars 09:30-13:30       | sweep -> BOS/CHoCH -> retrace into FVG/OB, ITM option    |
| llm_analyst     | 09:20                        | the local LLM's pre-open brief, ITM option               |
"""

from __future__ import annotations

import datetime as dt
from typing import Dict, List, Optional

import numpy as np

from ..options import Leg
from ..paper.sim import decide
from .base import Context, TradePlan, strike

T0920, T0945, T1510 = dt.time(9, 20), dt.time(9, 45), dt.time(15, 10)


class Book:
    name = "book"
    times: List[dt.time] = [T0920]          # decision times (first eligible check at/after each)
    once_per_day = True

    def propose(self, ctx: Context) -> Optional[TradePlan]:
        raise NotImplementedError


def _itm_leg(direction: int, spot: float, depth: int = 1) -> Leg:
    a = strike(spot)
    k = a - 50 * depth if direction > 0 else a + 50 * depth
    return Leg("CE" if direction > 0 else "PE", k, +1)


class DirectionalITM(Book):
    name = "directional_itm"

    def propose(self, ctx):
        d = decide(ctx.p_up, ctx.cfg) if ctx.p_up is not None else 0
        e = ctx.expiry(1)
        if d == 0 or e is None:
            return None
        return TradePlan(self.name, ctx.day, ctx.now, d, "itm_buy", [_itm_leg(d, ctx.spot)], e,
                         stop_pct=ctx.cfg.option_stop_pct, target_pct=ctx.cfg.option_target_pct,
                         meta={"p_up": ctx.p_up})


class DebitSpread(Book):
    name = "debit_spread"
    iv_pct_min = 0.6        # only when India VIX is in the top 40% of its last year

    def propose(self, ctx):
        d = decide(ctx.p_up, ctx.cfg) if ctx.p_up is not None else 0
        e = ctx.expiry(1)
        ivp = ctx.regime.get("vix_pct_1y")
        if d == 0 or e is None or ivp is None or ivp != ivp or ivp < self.iv_pct_min:
            return None
        typ, a = ("CE" if d > 0 else "PE"), strike(ctx.spot)
        far = a + d * ctx.cfg.spread_width_pts
        return TradePlan(self.name, ctx.day, ctx.now, d, "debit_spread", [Leg(typ, a, +1), Leg(typ, far, -1)], e,
                         stop_pct=50.0, target_pct=80.0, meta={"p_up": ctx.p_up, "vix_pct_1y": ivp})


class CreditSpread(Book):
    name = "credit_spread"
    k_sigma, wing = 0.75, 150

    def propose(self, ctx):
        d = decide(ctx.p_up, ctx.cfg) if ctx.p_up is not None else 0
        e = ctx.expiry(1)
        vrp = ctx.regime.get("vrp_har")
        if d == 0 or e is None or vrp is None or vrp != vrp or vrp <= 0:
            return None
        dist = self.k_sigma * ctx.sigma_day * ctx.spot
        if d > 0:   # bullish -> sell a put spread below
            ks = strike(ctx.spot - dist, "down")
            legs = [Leg("PE", ks, -1), Leg("PE", ks - self.wing, +1)]
        else:
            ks = strike(ctx.spot + dist, "up")
            legs = [Leg("CE", ks, -1), Leg("CE", ks + self.wing, +1)]
        return TradePlan(self.name, ctx.day, ctx.now, d, "credit_spread", legs, e, credit_stop_mult=1.5,
                         credit_target_pct=60.0, meta={"p_up": ctx.p_up, "vrp_har": vrp})


class IronCondor(Book):
    name = "iron_condor"

    def propose(self, ctx):
        e = ctx.expiry(1)
        vrp = ctx.regime.get("vrp_har")
        if e is None or vrp is None or vrp != vrp or vrp <= 0:
            return None
        if ctx.p_range is not None and ctx.p_range < ctx.cfg.condor_p_threshold:
            return None
        if ctx.events_today:
            return None
        dist = ctx.cfg.condor_k * ctx.sigma_day * ctx.spot
        sc, sp, w = strike(ctx.spot + dist, "up"), strike(ctx.spot - dist, "down"), ctx.cfg.condor_wing
        legs = [Leg("CE", sc, -1), Leg("CE", sc + w, +1), Leg("PE", sp, -1), Leg("PE", sp - w, +1)]
        return TradePlan(self.name, ctx.day, ctx.now, 0, "iron_condor", legs, e,
                         credit_stop_mult=ctx.cfg.condor_stop_mult, meta={"p_range": ctx.p_range, "vrp_har": vrp})


class IronFly0DTE(Book):
    name = "iron_fly_0dte"
    times = [T0945]

    def propose(self, ctx):
        e = ctx.expiry(0)
        vrp = ctx.regime.get("vrp_har")
        if e is None or e != ctx.day or vrp is None or vrp != vrp or vrp <= 0 or ctx.events_today:
            return None
        if ctx.gap is not None and abs(ctx.gap) > ctx.sigma_day:
            return None
        a = strike(ctx.spot)
        w = max(100.0, strike(ctx.sigma_day * ctx.spot))
        legs = [Leg("CE", a, -1), Leg("PE", a, -1), Leg("CE", a + w, +1), Leg("PE", a - w, +1)]
        return TradePlan(self.name, ctx.day, ctx.now, 0, "iron_fly", legs, e, credit_stop_mult=1.0,
                         credit_target_pct=40.0, meta={"vrp_har": vrp, "wing": w})


class EventStraddle(Book):
    name = "event_straddle"
    times = [T1510]
    events = ("rbi_mpc", "fomc", "union_budget")

    def propose(self, ctx):
        if not any(ev in ctx.events_next for ev in self.events) or not ctx.next_sessions:
            return None
        nxt = ctx.next_sessions[0]
        e = ctx.expiry(2, after=nxt) or ctx.expiry(1, after=nxt)
        if e is None:
            return None
        a = strike(ctx.spot)
        return TradePlan(self.name, ctx.day, ctx.now, 0, "long_straddle", [Leg("CE", a, +1), Leg("PE", a, +1)], e,
                         exit_time=dt.time(10, 30), exit_day=nxt, meta={"events": ctx.events_next})


class TurnOfMonth(Book):
    """Long from the close of the 2nd-last session of the month to the close of the 3rd session
    of the next month (Quantpedia/RFB 'TD-2 -> TD+3'), via an ITM call that outlives the hold."""
    name = "turn_of_month"
    times = [T1510]

    def propose(self, ctx):
        if ctx.sessions_left_in_month != 2 or len(ctx.next_sessions) < 4:
            return None
        exit_day = None
        month = ctx.day.month
        n_new = 0
        for d in ctx.next_sessions:
            if d.month != month:
                n_new += 1
                if n_new == 3:
                    exit_day = d
                    break
        if exit_day is None:
            return None
        e = ctx.expiry(1, after=exit_day)
        if e is None:
            return None
        return TradePlan(self.name, ctx.day, ctx.now, 1, "itm_buy", [_itm_leg(1, ctx.spot, depth=2)], e,
                         exit_time=T1510, exit_day=exit_day, stop_pct=60.0, meta={"exit_day": exit_day.isoformat()})


class SMCIntraday(Book):
    name = "smc_intraday"
    times = []                      # evaluated on every closed 5-minute bar in the window
    window = (dt.time(9, 30), dt.time(13, 30))
    rr = 2.0

    def propose(self, ctx):
        from ..analytics.smc import smc_setups
        b = ctx.intraday
        e = ctx.expiry(1)
        if b is None or len(b) < 60 or e is None:
            return None
        sess = b.index.to_series().dt.date
        setups = smc_setups(b, session=sess, rr=self.rr, start_time=self.window[0], end_time=self.window[1], k=3)
        if setups.empty:
            return None
        last = setups.iloc[-1]
        # only act on a setup signalled on the most recent closed bar of today
        if last["ts_signal"] != b.index[-1] or last["ts_signal"].date() != ctx.day:
            return None
        d = int(last["direction"])
        return TradePlan(self.name, ctx.day, ctx.now, d, "itm_buy", [_itm_leg(d, ctx.spot)], e,
                         stop_pct=60.0, underlying_stop=float(last["stop"]), underlying_target=float(last["target"]),
                         meta={"setup": {k: str(v) for k, v in last.items()}})


class LLMAnalyst(Book):
    name = "llm_analyst"
    threshold = 0.58

    def propose(self, ctx):
        b = (ctx.brief or {}).get("brief")
        e = ctx.expiry(1)
        if not b or e is None:
            return None
        p = float(b.get("p_up", 0.5))
        d = 1 if p >= self.threshold else (-1 if p <= 1 - self.threshold else 0)
        if d == 0:
            return None
        return TradePlan(self.name, ctx.day, ctx.now, d, "itm_buy", [_itm_leg(d, ctx.spot)], e,
                         stop_pct=ctx.cfg.option_stop_pct, target_pct=ctx.cfg.option_target_pct,
                         meta={"p_up_llm": p, "stance": b.get("stance")})


BOOKS: Dict[str, Book] = {b.name: b for b in (DirectionalITM(), DebitSpread(), CreditSpread(), IronCondor(),
                                             IronFly0DTE(), EventStraddle(), TurnOfMonth(), SMCIntraday(), LLMAnalyst())}


def books_for(names) -> List[Book]:
    return [BOOKS[n] for n in names if n in BOOKS]
