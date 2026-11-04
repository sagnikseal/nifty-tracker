"""Live paper trading for the v5 strategy books (never sends orders).

Each book gets its own paper equity (`book_capital`). Fills use real Upstox bid/ask when the
option chain is available and the VIX model otherwise (recorded per trade as `price_source`).
Exit rules travel with the trade (`rules` column), so a restart doesn't change them."""

from __future__ import annotations

import datetime as dt
import json
import logging
import math
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..calendar import IST, next_weekly_expiry, now_ist, parse_hhmm
from ..config import EngineConfig
from ..data import store
from ..options import Leg, Structure
from ..strategies.base import Context, TradePlan
from ..strategies.books import books_for
from .charges import order_charges
from .live import LivePaperTrader
from .sim import rates_from

log = logging.getLogger(__name__)
WINDOW = dt.timedelta(minutes=20)        # a decision time stays valid this long (restarts, slow data)


class BookRunner:
    def __init__(self, cfg: EngineConfig, trader: LivePaperTrader):
        self.cfg, self.trader, self.ledger = cfg, trader, trader.ledger
        self.books = books_for(cfg.strategy_list)
        self._daily_cache: Dict = {}

    # ------------------------------------------------------------------ context
    def _daily_context(self, day: dt.date, open_px: Optional[float]) -> Dict:
        if day in self._daily_cache:
            return self._daily_cache[day]
        from ..analytics.regime import regime_features
        from ..data.events import event_features
        raw = store.load_all()
        nf = raw["nifty"]
        ts = pd.Timestamp(day)
        if open_px and (len(nf) == 0 or nf.index[-1] < ts):
            nf = pd.concat([nf, pd.DataFrame({"open": [open_px], "high": [open_px], "low": [open_px], "close": [open_px],
                                              "volume": [0]}, index=pd.DatetimeIndex([ts]))])
        vix = raw["indiavix"]["close"] if len(raw.get("indiavix", [])) else None
        out: Dict = {}
        try:
            R = regime_features(nf, vix, lag=1).iloc[-1]
            out["regime"] = {k: float(v) for k, v in R.items() if v == v}
        except Exception as e:  # noqa: BLE001
            log.warning("regime failed: %s", e)
            out["regime"] = {}
        try:
            E = event_features(nf.index).iloc[-1]
            out["events_today"] = [k[3:] for k, v in E.items() if v == 1 and not k.endswith("_next") and k != "ev_any"]
            out["events_next"] = [k[3:-5] for k, v in E.items() if v == 1 and k.endswith("_next") and k != "ev_any_next"]
        except Exception:  # noqa: BLE001
            out["events_today"], out["events_next"] = [], []
        out["gap"] = math.log(open_px / float(nf["close"].iloc[-2])) if open_px and len(nf) > 1 else None
        out["vix_prev"] = float(vix.iloc[-1]) if vix is not None and len(vix) else None
        bd = pd.bdate_range(day, periods=35)
        out["next_sessions"] = [d.date() for d in bd[1:]]
        out["sessions_left_in_month"] = int(sum(1 for d in bd if d.month == day.month))
        self._daily_cache = {day: out}
        return out

    def _intraday(self, now: dt.datetime) -> Optional[pd.DataFrame]:
        """5-minute NIFTY bars closed by `now` (Upstox v3 when available, Yahoo otherwise)."""
        df = pd.DataFrame()
        up = self.trader.upstox
        if up.available:
            try:
                hist = up.candles(up.NIFTY, "minutes", 5, now.date() - dt.timedelta(days=6), now.date() - dt.timedelta(days=1))
                today = up.intraday_today(up.NIFTY, "minutes", 5)
                df = pd.concat([x for x in (hist, today) if len(x)]) if len(hist) or len(today) else pd.DataFrame()
                df = df[~df.index.duplicated(keep="last")].sort_index() if len(df) else df
            except Exception as e:  # noqa: BLE001
                log.warning("upstox 5m failed: %s", e)
                df = pd.DataFrame()
        if df.empty:
            try:
                df = self.trader.yahoo.bars("^NSEI", start=(now.date() - dt.timedelta(days=6)).isoformat(), interval="5m")
                df.index = pd.DatetimeIndex(df.index).tz_convert(IST)
            except Exception:  # noqa: BLE001
                return None
        df = df[(df.index.time >= dt.time(9, 15)) & (df.index.time <= dt.time(15, 25))]
        return df[df.index + pd.Timedelta(minutes=5) <= pd.Timestamp(now)]

    def context(self, now: dt.datetime) -> Optional[Context]:
        q = self.trader.spot_and_vix()
        if not q.get("spot"):
            return None
        day = now.date()
        sig = self.ledger.last_signal() or {}
        p_up = sig.get("p_up") if sig.get("date") == day.isoformat() else None
        dc = self._daily_context(day, sig.get("open") if sig.get("date") == day.isoformat() else q["spot"])
        try:
            exps = [e for e in self.trader.upstox.option_expiries() if e >= day] if self.trader.upstox.available else []
        except Exception:  # noqa: BLE001
            exps = []
        if not exps:
            exps, d = [], day
            for _ in range(6):
                e = next_weekly_expiry(d)
                exps.append(e)
                d = e + dt.timedelta(days=1)
        rng = self.ledger.kv_get(f"range_{day.isoformat()}")
        brief = None
        try:
            from ..semantic.market_brief import BRIEFS
            f = BRIEFS / f"{day.isoformat()}.json"
            brief = json.loads(f.read_text(encoding="utf-8")) if f.exists() else None
        except Exception:  # noqa: BLE001
            pass
        chain = {}
        try:
            from ..data.recorder import latest_chain_metrics
            chain = latest_chain_metrics(day)
        except Exception:  # noqa: BLE001
            pass
        need_bars = any(b.name == "smc_intraday" for b in self.books)
        return Context(cfg=self.cfg, day=day, now=now.time().replace(second=0, microsecond=0), spot=float(q["spot"]),
                       vix=float(q.get("vix") or dc.get("vix_prev") or 15.0), expiries=exps, p_up=p_up,
                       p_range=rng.get("p_inside") if isinstance(rng, dict) else None, regime=dc["regime"],
                       events_today=dc["events_today"], events_next=dc["events_next"], chain=chain,
                       intraday=self._intraday(now) if need_bars else None, brief=brief, gap=dc["gap"],
                       sessions_left_in_month=dc["sessions_left_in_month"], next_sessions=dc["next_sessions"])

    # ------------------------------------------------------------------ scheduling
    def _due(self, book, now: dt.datetime) -> bool:
        d = now.date().isoformat()
        if self.ledger.kv_get(f"book_{book.name}_{d}_traded"):
            return False
        if not book.times:   # bar-driven books (SMC): every new closed 5-minute bar in their window
            lo, hi = getattr(book, "window", (dt.time(9, 30), dt.time(13, 30)))
            if not (lo <= now.time() <= hi):
                return False
            slot = now.replace(minute=now.minute - now.minute % 5, second=0, microsecond=0).strftime("%H%M")
            key = f"book_{book.name}_{d}_{slot}"
        else:
            hits = [t for t in book.times
                    if dt.datetime.combine(now.date(), t, tzinfo=IST) <= now < dt.datetime.combine(now.date(), t, tzinfo=IST) + WINDOW]
            if not hits:
                return False
            key = f"book_{book.name}_{d}_{hits[0].strftime('%H%M')}"
        if self.ledger.kv_get(key):
            return False
        self.ledger.kv_set(key, now.isoformat())
        return True

    def step(self, now: Optional[dt.datetime] = None) -> List[int]:
        now = now or now_ist()
        due = [b for b in self.books if self._due(b, now)]
        opened = []
        if due:
            ctx = self.context(now)
            if ctx is None:
                return opened
            for b in due:
                try:
                    plan = b.propose(ctx)
                except Exception as e:  # noqa: BLE001
                    log.warning("book %s failed: %s", b.name, e)
                    continue
                if plan:
                    tid = self.open(plan, ctx)
                    if tid:
                        opened.append(tid)
                        self.ledger.kv_set(f"book_{b.name}_{now.date().isoformat()}_traded", tid)
        self.manage(now)
        return opened

    # ------------------------------------------------------------------ execution
    def equity(self, book: str) -> float:
        return self.cfg.book_capital + self.ledger.realized_pnl(book, "live")

    def open(self, plan: TradePlan, ctx: Context) -> Optional[int]:
        srcs, net = set(), 0.0
        for l in plan.legs:
            l.entry, src = self.trader.leg_price(l, plan.expiry, ctx.spot, ctx.vix, buying=l.qty_sign > 0)
            srcs.add(src)
            net += l.qty_sign * l.entry
        lot = self.cfg.lot_size
        if net < 0:
            ks = sorted({l.strike for l in plan.legs})
            width = max(ks[i + 1] - ks[i] for i in range(len(ks) - 1)) if len(ks) > 1 else 0
            max_loss = (width + net) * lot
        else:
            max_loss = net * lot
        eq = self.equity(plan.book)
        if max_loss <= 0 or max_loss > eq * self.cfg.max_premium_pct / 100:
            log.info("book %s skipped: max loss Rs %.0f vs equity Rs %.0f", plan.book, max_loss, eq)
            return None
        rates = rates_from(self.cfg)
        ch = sum(order_charges("option", "BUY" if l.qty_sign > 0 else "SELL", l.entry * lot, rates)["total"] for l in plan.legs)
        tid = self.ledger.insert_trade({
            "book": plan.book, "mode": "live", "date": plan.day.isoformat(), "entry_ts": now_ist().isoformat(),
            "direction": plan.direction, "style": plan.style, "lots": 1, "entry_value": round(net, 2),
            "legs": [{"typ": l.typ, "strike": l.strike, "sign": l.qty_sign, "entry": l.entry, "key": l.instrument_key,
                      "expiry": plan.expiry.isoformat()} for l in plan.legs],
            "charges": round(ch, 2), "p_up": ctx.p_up, "model_version": "v5:" + plan.book, "status": "OPEN",
            "price_source": ",".join(sorted(srcs)), "rules": {**plan.rules(), "meta": plan.meta}})
        log.info("BOOK %s ENTER #%s %s net=%.2f", plan.book, tid, plan.style, net)
        return tid

    def manage(self, now: Optional[dt.datetime] = None, force_exit: bool = False) -> None:
        now = now or now_ist()
        open_t = self.ledger.open_trades()
        if open_t.empty or "rules" not in open_t:
            return
        open_t = open_t[open_t["rules"].notna()]
        if open_t.empty:
            return
        q = self.trader.spot_and_vix()
        spot, vix = q.get("spot"), q.get("vix") or 15.0
        if not spot:
            return
        for _, t in open_t.iterrows():
            rules = json.loads(t["rules"])
            legs = json.loads(t["legs"])
            exp = dt.date.fromisoformat(legs[0]["expiry"])
            st = Structure(int(t["direction"]), t["style"],
                           [Leg(l["typ"], l["strike"], l["sign"], l["entry"], 0.0, l.get("key", "")) for l in legs])
            val = 0.0
            for l in st.legs:
                l.exit, _ = self.trader.leg_price(l, exp, spot, vix, buying=l.qty_sign < 0)
                val += l.qty_sign * l.exit
            v0 = float(t["entry_value"])
            pnl = val - v0
            reason = None
            exit_day = dt.date.fromisoformat(rules.get("exit_day", t["date"]))
            exit_at = dt.datetime.combine(exit_day, parse_hhmm(rules.get("exit_time", "15:15")), tzinfo=IST)
            if force_exit or now >= exit_at:
                reason = "time"
            elif v0 > 0:
                if rules.get("stop_pct") is not None and val <= v0 * (1 - rules["stop_pct"] / 100):
                    reason = "stop"
                elif rules.get("target_pct") is not None and val >= v0 * (1 + rules["target_pct"] / 100):
                    reason = "target"
            else:
                credit = -v0
                if rules.get("credit_stop_mult") is not None and pnl <= -rules["credit_stop_mult"] * credit:
                    reason = "stop"
                elif rules.get("credit_target_pct") is not None and pnl >= rules["credit_target_pct"] / 100 * credit:
                    reason = "target"
            d = int(rules.get("direction", 0))
            if reason is None and d != 0:
                if rules.get("underlying_stop") is not None and (spot - rules["underlying_stop"]) * d <= 0:
                    reason = "stop_underlying"
                elif rules.get("underlying_target") is not None and (spot - rules["underlying_target"]) * d >= 0:
                    reason = "target_underlying"
            if reason:
                self.trader.close(t, st, val, reason)

    def end_of_day(self) -> None:
        d = now_ist().date().isoformat()
        for b in self.books:
            self.ledger.set_equity(d, b.name, self.equity(b.name))

    def scorecard(self) -> Dict[str, Dict]:
        tr = self.ledger.trades("live")
        out = {}
        for b in self.books:
            t = tr[(tr["book"] == b.name) & (tr["status"] == "CLOSED")] if len(tr) else tr
            out[b.name] = {"closed": int(len(t)), "net": round(float(t["net_pnl"].sum()), 0) if len(t) else 0.0,
                           "win_rate": round(float((t["net_pnl"] > 0).mean()), 3) if len(t) else None,
                           "equity": round(self.equity(b.name), 0)}
        return out
