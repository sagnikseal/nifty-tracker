"""Live runtime (paper only): one process, independent threads, one shared snapshot.

  price thread  every `price_poll_seconds` (5 s): NIFTY + India VIX quote; price of every open
                option position; stop / target / extension / time rules -> exits.
  bar thread    every 15 s: today's 1-minute bars; on each new 5-minute close -> movement-model
                predictions, the "2 closes against" rule, entries for the model book, and a wake-up
                for the agent.
  agent thread  on each 5-minute close: the local LLM decides for the agent book (guardrailed).
  news thread   polls fast sources on their own schedules, stores new headlines (first_seen = now),
                scores them with the local LLM in batches, attaches learned impacts, applies the
                adverse-headline reaction rule (only once validated).
  jobs thread   08:40 data refresh, 09:10 option-chain recorder, 09:20 daily model, 15:45 end of
                day, Friday 16:15 retrain.
Slow work (news, LLM, retraining) can never delay a stop: it runs in other threads.
Nothing here can place a real order.
"""

from __future__ import annotations

import collections
import datetime as dt
import json
import logging
import math
import threading
import time
import traceback
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..calendar import IST, now_ist, year_fraction
from ..config import DATA_DIR, EngineConfig
from ..data.providers import UpstoxProvider, YahooProvider
from ..options import atm, bs_price, half_spread, smile_iv, Leg
from ..paper.charges import order_charges
from ..paper.ledger import Ledger
from ..paper.sim import rates_from
from ..strategies.atm_rules import ATMRules, PositionState, on_bar_close, on_headline, on_price

log = logging.getLogger(__name__)
BOOKS = {"model_rules": "Model + your rules", "agent": "LLM agent + your rules", "manual": "Manual (you)"}


def market_status(now: dt.datetime) -> str:
    if now.weekday() >= 5:
        return "CLOSED"
    t = now.time()
    if dt.time(9, 0) <= t < dt.time(9, 15):
        return "PRE_OPEN"
    if dt.time(9, 15) <= t <= dt.time(15, 30):
        return "OPEN"
    return "CLOSED"


class Runtime:
    def __init__(self, cfg: EngineConfig, ledger: Optional[Ledger] = None, start_threads: bool = True):
        self.cfg = cfg
        self.ledger = ledger or Ledger()
        self.rules = ATMRules.from_cfg(cfg)
        self.upstox = UpstoxProvider()
        self.yahoo = YahooProvider()
        self.lock = threading.RLock()
        self.changed = threading.Event()          # set when the snapshot changes (SSE push)
        self.bar_event = threading.Event()        # set on each new 5-minute close (agent wake-up)
        self.started = time.time()
        self.alerts = collections.deque(maxlen=40)
        self.quote: Dict = {"source": "none"}
        self.bars1: pd.DataFrame = pd.DataFrame()
        self.vix1: pd.DataFrame = pd.DataFrame()
        self.last5: Optional[pd.Timestamp] = None
        self.pred: Dict = {"history": []}
        self.features: Dict = {}
        self.p_daily: Optional[float] = None
        self.daily_sig: Dict = {}
        self.news: List[Dict] = []
        self.agent_last: Optional[Dict] = None
        self.agent_count = 0
        self.positions: Dict[str, Dict] = {}      # book -> {"state": PositionState, "trade": row, "leg": {...}}
        self.health = {"engine": "starting", "llm": False, "upstox": self.upstox.available, "news_thread": False,
                       "last_bar": None, "loop_lag_s": None}
        self.auto = {b: bool(self.ledger.kv_get(f"auto_{b}", default=self._auto_default(b))) for b in BOOKS}
        self._load_models()
        self._restore_positions()
        from ..data.news_sources import NewsPoller
        self.poller = NewsPoller(default_poll_s=cfg.news_poll_seconds)
        self.agent = None
        if cfg.agent_enabled:
            from ..agent.llm_agent import TradingAgent
            self.agent = TradingAgent(cfg, self.rules)
        self.research_books = None                # optional research books (config "strategy_books")
        if cfg.strategy_list:
            from ..paper.books_live import BookRunner
            from ..paper.live import LivePaperTrader
            self.research_books = BookRunner(cfg, LivePaperTrader(cfg, self.ledger))
        self.threads: List[threading.Thread] = []
        if start_threads:
            self.start()

    # ------------------------------------------------------------------ setup
    def _auto_default(self, b: str) -> bool:
        return {"model_rules": self.cfg.model_rules_auto, "agent": self.cfg.agent_auto, "manual": False}[b]

    def _load_models(self) -> None:
        from ..intraday_model import load_champion as load_intraday
        from ..model import load_champion as load_daily
        from ..semantic.news_impact import load_model
        try:
            self.intraday, self.intraday_meta = load_intraday()
        except Exception as e:  # noqa: BLE001
            self.intraday, self.intraday_meta = None, None
            self.alert("warn", f"intraday model failed to load: {e}")
        if self.intraday is None:
            self.alert("warn", "No intraday model yet: run `python -m nifty_engine intraday-train` (about 2 minutes)")
        try:
            _, self.daily_meta = load_daily()
        except Exception:  # noqa: BLE001
            self.daily_meta = None
        self.impact = load_model()

    def _restore_positions(self) -> None:
        tr = self.ledger.open_trades()
        for _, t in tr.iterrows():
            if t["book"] not in BOOKS:
                continue
            legs = json.loads(t["legs"] or "[]")
            rules = json.loads(t["rules"]) if isinstance(t.get("rules"), str) and t["rules"] else {}
            if not legs:
                continue
            ts = pd.Timestamp(t["entry_ts"]).to_pydatetime()
            st = PositionState(int(t["direction"]), float(t["entry_value"]), self.rules.lots * self.cfg.lot_size, ts,
                               float(rules.get("spot_at_entry") or 0.0))
            st.target_hit = bool(rules.get("target_hit", False))
            self.positions[t["book"]] = {"state": st, "trade": t.to_dict(), "leg": legs[0], "ltp": None, "source": None}

    def alert(self, level: str, text: str) -> None:
        self.alerts.appendleft({"ts": now_ist().isoformat(), "level": level, "text": text})
        log.log(logging.WARNING if level in ("warn", "error") else logging.INFO, text)
        self.changed.set()

    def start(self) -> None:
        for name, fn in (("price", self._price_loop), ("bars", self._bar_loop), ("news", self._news_loop),
                         ("agent", self._agent_loop), ("jobs", self._jobs_loop)):
            t = threading.Thread(target=self._guard(name, fn), name=f"ne-{name}", daemon=True)
            t.start()
            self.threads.append(t)
        self.health["engine"] = "running"

    def _guard(self, name, fn):
        def run():
            while True:
                try:
                    fn()
                except Exception:  # noqa: BLE001
                    log.error("%s thread error:\n%s", name, traceback.format_exc())
                    self.alert("error", f"{name} thread error (restarting): {traceback.format_exc().splitlines()[-1][:160]}")
                    time.sleep(5)
        return run

    # ------------------------------------------------------------------ market data
    def fetch_quote(self) -> Dict:
        q: Dict = {}
        if self.upstox.available:
            try:
                d = self.upstox.ltp([UpstoxProvider.NIFTY, UpstoxProvider.VIX])
                vals = list(d.values())
                q = {"spot": d.get(UpstoxProvider.NIFTY) or d.get("NSE_INDEX:Nifty 50") or (vals[0] if vals else None),
                     "vix": d.get(UpstoxProvider.VIX) or d.get("NSE_INDEX:India VIX") or (vals[1] if len(vals) > 1 else None),
                     "source": "upstox"}
                self.health["upstox"] = True
            except Exception as e:  # noqa: BLE001
                self.health["upstox"] = False
                log.debug("upstox ltp failed: %s", e)
        if not q.get("spot"):
            if time.time() - self.quote.get("_yahoo_t", 0) < 30 and self.quote.get("source") == "yahoo":
                return self.quote                       # Yahoo is ~1 min delayed: don't hammer it
            q = {"spot": self.yahoo.last_price("^NSEI"), "vix": self.yahoo.last_price("^INDIAVIX"), "source": "yahoo",
                 "_yahoo_t": time.time()}
        q["fetched"] = time.time()
        return q

    def _price_loop(self) -> None:
        t0 = time.time()
        now = now_ist()
        if market_status(now) in ("OPEN", "PRE_OPEN") or self.positions or not self.quote.get("spot"):
            q = self.fetch_quote()
            if q.get("spot"):
                with self.lock:
                    self.quote.update(q)
                self.changed.set()
            if self.positions:
                self.manage_positions(now)
        self.health["loop_lag_s"] = round(time.time() - t0, 2)
        time.sleep(max(0.5, self.cfg.price_poll_seconds - (time.time() - t0)))

    def option_price(self, leg: Dict, spot: float, vix: float) -> tuple:
        """Sell-side price of a long option now: broker bid, else LTP minus spread, else VIX model."""
        key = leg.get("key")
        if self.upstox.available and key:
            try:
                qd = self.upstox.quotes([key])
                v = next(iter(qd.values())) if qd else {}
                depth = (v.get("depth") or {}).get("buy") or []
                bid = depth[0].get("price") if depth else None
                if bid and bid > 0:
                    return max(0.05, float(bid) - self.cfg.option_slippage_pts), "broker_bid_ask"
                ltp = v.get("last_price")
                if ltp:
                    return max(0.05, float(ltp) - half_spread(float(ltp)) - self.cfg.option_slippage_pts), "broker_ltp"
            except Exception as e:  # noqa: BLE001
                log.debug("option quote failed: %s", e)
        exp = dt.date.fromisoformat(leg["expiry"])
        T = year_fraction(now_ist(), dt.datetime.combine(exp, dt.time(15, 30), tzinfo=IST))
        mid = bs_price(spot, leg["strike"], T, smile_iv(vix or 15.0, spot, leg["strike"]), leg["typ"])
        return max(0.05, mid - half_spread(mid) - self.cfg.option_slippage_pts), "model_vix"

    def manage_positions(self, now: dt.datetime) -> None:
        spot, vix = self.quote.get("spot"), self.quote.get("vix")
        if not spot:
            return
        for book, p in list(self.positions.items()):
            px, src = self.option_price(p["leg"], spot, vix)
            p["ltp"], p["source"] = px, src
            p30 = self._p_dir_source(book)
            reason = on_price(self.rules, p["state"], px, now, p30)
            if reason:
                self.close(book, px, reason, src)
        self.changed.set()

    def _p_dir_source(self, book: str) -> Optional[float]:
        """P(up next 30 min) used by the 80% hold rule: the agent's own for the agent book."""
        if book == "agent" and self.agent_last and self.agent_last.get("p_up_30m") is not None:
            return self.agent_last["p_up_30m"]
        return self.pred.get("p_up_30m")

    # ------------------------------------------------------------------ bars + predictions
    def fetch_today_bars(self) -> None:
        today = now_ist().date()
        n1 = v1 = pd.DataFrame()
        if self.upstox.available:
            try:
                n1 = self.upstox.intraday_today(UpstoxProvider.NIFTY, "minutes", 1)
                v1 = self.upstox.intraday_today(UpstoxProvider.VIX, "minutes", 1)
            except Exception as e:  # noqa: BLE001
                log.debug("upstox intraday failed: %s", e)
        if n1.empty:
            try:
                n1 = self.yahoo.bars("^NSEI", start=(today - dt.timedelta(days=4)).isoformat(), interval="1m")
                v1 = self.yahoo.bars("^INDIAVIX", start=(today - dt.timedelta(days=4)).isoformat(), interval="1m")
                for df in (n1, v1):
                    if len(df):
                        df.index = df.index.tz_convert(IST)
            except Exception as e:  # noqa: BLE001
                log.debug("yahoo 1m failed: %s", e)
        now = now_ist()
        for name, df in (("n", n1), ("v", v1)):
            if df is None or df.empty:
                continue
            df = df[(df.index.date == today) & (df.index + pd.Timedelta(minutes=1) <= pd.Timestamp(now))]
            df = df[(df.index.time >= dt.time(9, 15)) & (df.index.time <= dt.time(15, 29))]
            if name == "n":
                self.bars1 = df[["open", "high", "low", "close"]]
            else:
                self.vix1 = df[["open", "high", "low", "close"]]

    def last_session_bars(self) -> pd.DataFrame:
        from ..intraday_model import load_bars
        try:
            b = load_bars("nifty_1m")
            if b.empty:
                b = load_bars("nifty_5m")
            if b.empty:
                return b
            d = b.index[-1].date()
            return b[b.index.date == d][["open", "high", "low", "close"]]
        except Exception:  # noqa: BLE001
            return pd.DataFrame()

    def _bar_loop(self) -> None:
        now = now_ist()
        st = market_status(now)
        if st == "OPEN":
            self.fetch_today_bars()
            if len(self.bars1):
                self.health["last_bar"] = self.bars1.index[-1].isoformat()
                from ..intraday_model import to_5m
                b5 = to_5m(self.bars1)
                b5 = b5[b5.index + pd.Timedelta(minutes=5) <= pd.Timestamp(now)]
                if len(b5) and (self.last5 is None or b5.index[-1] > self.last5):
                    self.last5 = b5.index[-1]
                    self.on_five_minute_close(b5, now)
            if self.research_books is not None:
                try:
                    self.research_books.step(now)
                except Exception as e:  # noqa: BLE001
                    log.warning("research books step failed: %s", e)
            self.changed.set()
            time.sleep(15)
        else:
            if self.bars1.empty or self.bars1.index[-1].date() != now.date():
                self.bars1 = self.last_session_bars() if self.bars1.empty else self.bars1
            time.sleep(30)

    def predict(self, today5: pd.DataFrame) -> Optional[Dict]:
        if self.intraday is None or today5.empty:
            return None
        from ..data import store
        from ..intraday_model import build_frame, load_bars, predict_rows, to_5m
        hist = load_bars("nifty_5m")
        hist = hist[hist.index.date < today5.index[0].date()]
        hist = hist[hist.index >= hist.index.max() - pd.Timedelta(days=10)] if len(hist) else hist
        vh = load_bars("indiavix_5m")
        vh = vh[vh.index.date < today5.index[0].date()]
        v5 = pd.concat([vh[vh.index >= vh.index.max() - pd.Timedelta(days=10)] if len(vh) else vh,
                        to_5m(self.vix1) if len(self.vix1) else pd.DataFrame()])
        n5 = pd.concat([hist[["open", "high", "low", "close"]], today5])
        nf = store.load("nifty")
        pdaily = pd.Series({pd.Timestamp(now_ist().date()): self.p_daily}) if self.p_daily is not None else None
        news_fn = None
        try:
            from ..semantic.news_impact import NewsFeatureFn
            news_fn = NewsFeatureFn.from_store(self.cfg)
        except Exception:  # noqa: BLE001
            pass
        X = build_frame(n5, v5, nf["close"] if len(nf) else None, pdaily, news_fn, last_only=True)
        if X.empty:
            return None
        groups = (self.intraday_meta or {}).get("groups") or []
        if groups:                      # extra groups: same functions as training, live inputs
            from ..intraday_model import add_groups
            from .live_features import LiveFeatureInputs
            if not hasattr(self, "live_inputs"):
                self.live_inputs = LiveFeatureInputs(self.upstox)
            series = self.live_inputs.series(n5, v5)
            daily = None
            if "daily_ctx" in groups:
                if getattr(self, "_daily_feats_day", None) != now_ist().date():
                    from ..features import build_daily_features
                    self._daily_feats = build_daily_features(store.load_all(), "oc", groups=["base"])
                    self._daily_feats_day = now_ist().date()
                daily = self._daily_feats
            chain = None
            if "chain" in groups:
                row = self.live_inputs.chain_rows(n5["close"], X.index[-1])
                if row:
                    chain = pd.DataFrame([row]).set_index("ts")
            X = add_groups(X, groups, series, daily, chain)
            from ..intraday_model import GROUP_PREFIX
            missing = [g for g in groups if g in ("banknifty", "breadth", "futures", "chain") and
                       not any(c.startswith(GROUP_PREFIX[g]) and pd.notna(X[c].iloc[-1]) for c in X.columns)]
            if missing and not getattr(self, "_warned_missing", False):
                self._warned_missing = True
                self.alert("warn", f"Live data missing for feature groups {missing} (needs UPSTOX_ANALYTICS_TOKEN); "
                                   "predictions use the remaining inputs")
        out = predict_rows(self.intraday, X)
        self.features = {k: (None if pd.isna(v) else float(v)) for k, v in X.iloc[-1].items()
                         if not isinstance(v, (pd.Timestamp, str))}
        return {"p_up_30m": out.get("30m", {}).get("p"), "p_up_close": out.get("close", {}).get("p"),
                "top_drivers": out.get("30m", {}).get("drivers", []), "ts": X.index[-1],
                "voters": {h: out.get(h, {}).get("voters") for h in ("30m", "close")},
                "agreement": {h: out.get(h, {}).get("agreement") for h in ("30m", "close")}}

    def on_five_minute_close(self, b5: pd.DataFrame, now: dt.datetime) -> None:
        try:
            pr = self.predict(b5)
        except Exception as e:  # noqa: BLE001
            pr = None
            self.alert("warn", f"prediction failed: {str(e)[:160]}")
        with self.lock:
            if pr:
                self.pred.update({"p_up_30m": pr["p_up_30m"], "p_up_close": pr["p_up_close"],
                                  "top_drivers": pr["top_drivers"], "updated": now.isoformat(),
                                  "voters": pr["voters"], "agreement": pr["agreement"]})
                self.pred["history"].append({"t_ms": int(pr["ts"].timestamp() * 1000), "p30": pr["p_up_30m"],
                                             "pclose": pr["p_up_close"]})
                self.pred["history"] = [h for h in self.pred["history"]
                                        if dt.datetime.fromtimestamp(h["t_ms"] / 1000, IST).date() == now.date()]
        spot_close = float(b5["close"].iloc[-1])
        for book, p in list(self.positions.items()):
            px = p.get("ltp") or p["state"].entry_price
            reason = on_bar_close(self.rules, p["state"], spot_close, px, now)
            if reason:
                self.close(book, px, reason, p.get("source") or "?")
        if self.auto.get("model_rules") and "model_rules" not in self.positions:
            d = self.rules.entry_direction(self.pred.get("p_up_close"), now, self.trades_today("model_rules"))
            if d:
                self.enter("model_rules", d, note=f"P(up by 15:15) = {self.pred.get('p_up_close'):.2f}")
        self.bar_event.set()
        self.changed.set()

    # ------------------------------------------------------------------ trading (paper)
    def trades_today(self, book: str) -> int:
        tr = self.ledger.trades("live")
        if tr.empty:
            return 0
        return int(((tr["book"] == book) & (tr["date"] == now_ist().date().isoformat())).sum())

    def _expiry(self) -> dt.date:
        from ..calendar import next_weekly_expiry
        today = now_ist().date()
        if self.upstox.available:
            try:
                exps = [e for e in self.upstox.option_expiries() if (e - today).days >= 1]
                if exps:
                    return exps[0]
            except Exception:  # noqa: BLE001
                pass
        return next_weekly_expiry(today, min_days=1)

    def enter(self, book: str, direction: int, note: str = "") -> Dict:
        with self.lock:
            if book in self.positions:
                return {"ok": False, "error": f"{book} already has an open position"}
            spot, vix = self.quote.get("spot"), self.quote.get("vix") or 15.0
            if not spot:
                return {"ok": False, "error": "no NIFTY quote yet"}
            if market_status(now_ist()) != "OPEN":
                return {"ok": False, "error": "market is closed"}
            exp = self._expiry()
            typ = "CE" if direction > 0 else "PE"
            strike = atm(spot)
            leg = Leg(typ, strike, +1)
            from ..paper.live import LivePaperTrader
            lp = LivePaperTrader.__new__(LivePaperTrader)          # reuse its chain pricing without its state
            lp.cfg, lp.upstox, lp._chain_cache = self.cfg, self.upstox, {}
            try:
                price, src = lp.leg_price(leg, exp, spot, vix, buying=True)
            except Exception:  # noqa: BLE001
                T = year_fraction(now_ist(), dt.datetime.combine(exp, dt.time(15, 30), tzinfo=IST))
                mid = bs_price(spot, strike, T, smile_iv(vix, spot, strike), typ)
                price, src = mid + half_spread(mid) + self.cfg.option_slippage_pts, "model_vix"
            qty = self.rules.qty
            rates = rates_from(self.cfg)
            ch = order_charges("option", "BUY", price * qty, rates)["total"]
            now = now_ist()
            legs = [{"typ": typ, "strike": strike, "sign": 1, "entry": round(price, 2), "key": leg.instrument_key,
                     "expiry": exp.isoformat()}]
            tid = self.ledger.insert_trade({
                "book": book, "mode": "live", "date": now.date().isoformat(), "entry_ts": now.isoformat(),
                "direction": direction, "style": "atm_buy", "lots": self.rules.lots, "entry_value": round(price, 2),
                "legs": legs, "charges": round(ch, 2), "p_up": self.pred.get("p_up_close"),
                "model_version": (self.intraday_meta or {}).get("version") if book == "model_rules" else
                (self.cfg.llm_model if book == "agent" else "manual"),
                "status": "OPEN", "price_source": src,
                "rules": {**self.rules.to_public(), "spot_at_entry": spot, "note": note}})
            st = PositionState(direction, price, qty, now, spot)
            trade = self.ledger.trades("live")
            trade = trade[trade["id"] == tid].iloc[0].to_dict()
            self.positions[book] = {"state": st, "trade": trade, "leg": legs[0], "ltp": price, "source": src}
        self.alert("info", f"[{BOOKS[book]}] BUY {strike:.0f} {typ} @ {price:.2f} ({src}) {note}")
        return {"ok": True, "trade_id": tid}

    def close(self, book: str, px: float, reason: str, src: str = "") -> Dict:
        with self.lock:
            p = self.positions.pop(book, None)
            if not p:
                return {"ok": False, "error": f"no open position in {book}"}
            st, t = p["state"], p["trade"]
            qty = st.qty
            rates = rates_from(self.cfg)
            ch = float(t.get("charges") or 0) + order_charges("option", "SELL", px * qty, rates)["total"]
            gross = (px - st.entry_price) * qty
            legs = [dict(p["leg"], exit=round(px, 2))]
            self.ledger.update_trade(int(t["id"]), exit_ts=now_ist().isoformat(), exit_value=round(px, 2),
                                     gross_pnl=round(gross, 2), charges=round(ch, 2), net_pnl=round(gross - ch, 2),
                                     exit_reason=reason, status="CLOSED", legs=legs)
        self.alert("info", f"[{BOOKS[book]}] EXIT ({reason}) @ {px:.2f}: net Rs {gross - ch:,.0f}")
        return {"ok": True, "trade_id": int(t["id"]), "net_pnl": round(gross - ch, 2)}

    def manual_order(self, action: str, book: str = "manual") -> Dict:
        if book not in BOOKS:
            return {"ok": False, "error": f"unknown book {book}"}
        if action in ("buy_ce", "buy_pe"):
            return self.enter(book, 1 if action == "buy_ce" else -1, note="manual (dashboard)")
        if action == "exit":
            p = self.positions.get(book)
            if not p:
                return {"ok": False, "error": f"no open position in {book}"}
            spot, vix = self.quote.get("spot"), self.quote.get("vix")
            px, src = self.option_price(p["leg"], spot, vix) if spot else (p.get("ltp") or p["state"].entry_price, "last")
            return self.close(book, px, "manual", src)
        return {"ok": False, "error": f"unknown action {action}"}

    def set_auto(self, book: str, enabled: bool) -> Dict:
        if book not in BOOKS:
            return {"ok": False, "error": f"unknown book {book}"}
        self.auto[book] = bool(enabled)
        self.ledger.kv_set(f"auto_{book}", bool(enabled))
        self.changed.set()
        return {"ok": True}

    # ------------------------------------------------------------------ agent
    def _agent_loop(self) -> None:
        if not self.bar_event.wait(timeout=60):
            if self.agent is not None and not self.health.get("llm"):
                self.health["llm"] = self.agent.available()
            return
        self.bar_event.clear()
        if self.agent is None or market_status(now_ist()) != "OPEN":
            return
        if not self.agent.available():
            self.health["llm"] = False
            return
        self.health["llm"] = True
        from ..agent.llm_agent import build_state
        snap = self.snapshot(internal=True)
        pos = snap["positions_by_book"].get("agent")
        state = build_state(snap, pos, self.trades_today("agent"), self.rules)
        d = self.agent.decide(state)
        self.agent_last = d
        self.agent_count += 1
        if self.auto.get("agent"):
            a = d["action"]
            if a in ("ENTER_LONG", "ENTER_SHORT") and "agent" not in self.positions:
                self.enter("agent", 1 if a == "ENTER_LONG" else -1, note=f"agent conf {d['confidence']:.2f}")
            elif a == "EXIT" and "agent" in self.positions:
                p = self.positions["agent"]
                self.close("agent", p.get("ltp") or p["state"].entry_price, "agent_exit", p.get("source") or "?")
        self.changed.set()

    # ------------------------------------------------------------------ news
    def _news_loop(self) -> None:
        from ..semantic.news_llm import store_items
        self.health["news_thread"] = True
        items = self.poller.poll()
        if items:
            fresh = []
            for it in items:
                ts = it.get("ts")
                # never treat an old article as news just because we first saw it now
                it["stale"] = bool(ts and (now_ist() - ts) > dt.timedelta(hours=6))
                if not it["stale"]:
                    fresh.append(it)
            n_before = self._count_known()
            n_new = store_items(fresh)
            if n_new:
                by = collections.Counter(i["source"] for i in fresh)
                self.poller.record_new(dict(by))
                self._score_and_react()
            log.debug("news: %d items, %d new (known %d)", len(items), n_new, n_before)
        if time.time() - getattr(self, "_upstox_news_t", 0) > 300 and self.upstox.available:
            self._upstox_news_t = time.time()
            try:
                from ..data.instruments import HEAVYWEIGHTS, equity_keys
                un = self.upstox.news(list(equity_keys(self.cfg.extra.get("breadth_symbols") or HEAVYWEIGHTS).values()))
                if store_items(un):
                    self._score_and_react()
            except Exception as e:  # noqa: BLE001
                log.debug("upstox news failed: %s", e)
        if not self.news or time.time() - getattr(self, "_news_refresh_t", 0) > 60:
            self.refresh_news_view()
        time.sleep(3)

    def _count_known(self) -> int:
        return len(self.news)

    def _score_and_react(self) -> None:
        from ..semantic.news_llm import score_pending
        try:
            res = score_pending(self.cfg, max_items=64, since_days=2)
            self.health["llm"] = bool(res.get("llm_available", self.health.get("llm")))
        except Exception as e:  # noqa: BLE001
            self.alert("warn", f"news scoring failed: {str(e)[:120]}")
        before = {n["id"] for n in self.news}
        self.refresh_news_view()
        validated = bool(self.impact and self.impact.validated)
        for n in self.news:
            if n["id"] in before or (n.get("age_min") or 99) > 10:
                continue
            exp_bps = (n.get("impact") or {}).get("exp_bps_30m")
            for book, p in list(self.positions.items()):
                if on_headline(self.rules, p["state"], exp_bps, validated):
                    self.close(book, p.get("ltp") or p["state"].entry_price, "news", p.get("source") or "?")
                    self.alert("warn", f"[{BOOKS[book]}] exited on adverse headline: {n['title'][:90]}")
            if exp_bps is not None and abs(exp_bps) >= 8:
                self.alert("info", f"Market-moving headline ({exp_bps:+.0f} bps learned impact): {n['title'][:100]}")
        self.changed.set()

    def refresh_news_view(self) -> None:
        from ..semantic.news_llm import scored_frame
        self._news_refresh_t = time.time()
        today = now_ist().date()
        try:
            S = scored_frame(self.cfg, today - dt.timedelta(days=2), today)
        except Exception:  # noqa: BLE001
            return
        if S is None or S.empty:
            return
        S = S.sort_values("first_seen", ascending=False).head(150)
        cats = {name: st.cfg.get("category") for name, st in self.poller.states.items()}
        cats.update({"upstox_news": "broker"})
        out = []
        now = now_ist()
        for r in S.to_dict("records"):
            llm = {"direction": r.get("nifty_direction"), "magnitude": r.get("magnitude"), "surprise": r.get("surprise"),
                   "relevance": r.get("india_relevance"), "confidence": r.get("confidence"), "channel": r.get("channel"),
                   "horizon": r.get("horizon"), "reason": r.get("reason"), "scorer": r.get("scorer")}
            imp = None
            if self.impact is not None and self.impact.n:
                row = dict(r, t_min=(r["first_seen"].hour * 60 + r["first_seen"].minute) - 555)
                imp = {"exp_bps_30m": round(self.impact.predict_bps(row), 1), "p_up_30m": round(self.impact.p_up(row), 3),
                       "basis": f"learned from {self.impact.n} past headlines"
                                + ("" if self.impact.validated else " (not yet validated: information only)")}
            src = str(r.get("source") or "")
            out.append({"id": r["id"], "title": r["title"], "source": src, "category": cats.get(src, "other"),
                        "url": r.get("url"), "first_seen": r["first_seen"].isoformat(),
                        "age_min": round((now - r["first_seen"]).total_seconds() / 60, 1), "llm": llm, "impact": imp})
        with self.lock:
            self.news = out

    # ------------------------------------------------------------------ scheduled jobs
    def _once(self, key: str) -> bool:
        if self.ledger.kv_get(key):
            return False
        self.ledger.kv_set(key, now_ist().isoformat())
        return True

    def _store_is_stale(self) -> bool:
        """True if the stored intraday bars end before the last completed weekday session."""
        from ..intraday_model import load_bars
        b = load_bars("nifty_5m")
        if b.empty:
            return True
        now = now_ist()
        last = now.date() if now.time() >= dt.time(15, 35) else now.date() - dt.timedelta(days=1)
        while last.weekday() >= 5:
            last -= dt.timedelta(days=1)
        return b.index[-1].date() < last

    def _jobs_loop(self) -> None:
        now = now_ist()
        d = now.date().isoformat()
        t = now.time()
        if not getattr(self, "_startup_checked", False):
            self._startup_checked = True
            if self._store_is_stale():
                self.alert("info", "Market data is out of date: refreshing in the background ...")
                self.refresh_data()
                if market_status(now_ist()) != "OPEN":
                    self.bars1 = self.last_session_bars()
                self.changed.set()
        if now.weekday() < 5:
            if dt.time(8, 30) <= t < dt.time(9, 14) and self._once(f"job_refresh_{d}"):
                self.refresh_data()
            if t >= dt.time(9, 20) and t < dt.time(15, 30) and self.p_daily is None:
                self.daily_signal()
            if (self.cfg.record_live and self.upstox.available and dt.time(9, 10) <= t < dt.time(15, 25)
                    and self._once(f"job_recorder_{d}")):
                from ..data.recorder import run_session
                threading.Thread(target=lambda: log.info("recorder: %s", run_session(self.cfg)), daemon=True).start()
            if t >= dt.time(15, 45) and self._once(f"job_eod_{d}"):
                self.end_of_day()
            if now.weekday() == 4 and t >= dt.time(16, 15) and self._once(f"job_retrain_{d}"):
                self.retrain()
        time.sleep(20)

    def refresh_data(self) -> None:
        from ..data import store
        from ..data.global_markets import refresh_hourly
        from .intraday_store import refresh_intraday
        for name, fn in (("daily", store.refresh_daily), ("hourly", refresh_hourly), ("intraday", refresh_intraday)):
            try:
                fn()
            except Exception as e:  # noqa: BLE001
                self.alert("warn", f"{name} data refresh failed: {str(e)[:120]}")

    def daily_signal(self) -> None:
        if getattr(self, "_daily_try", 0) > time.time() - 300:
            return
        self._daily_try = time.time()
        try:
            from ..paper.live import LivePaperTrader
            sig = LivePaperTrader(self.cfg, self.ledger).morning_signal()
            if "error" in sig:
                self.alert("warn", f"daily model: {sig['error']}")
                return
            self.p_daily, self.daily_sig = float(sig["p_up"]), sig
            with self.lock:
                self.pred["p_daily"] = self.p_daily
            if self.cfg.agent_enabled and "llm_analyst" not in (self.cfg.strategy_books or ""):
                from ..semantic.market_brief import build_state, write_brief
                br = write_brief(self.cfg, build_state(self.cfg, sig))
                self.brief = br.get("brief")
        except Exception as e:  # noqa: BLE001
            self.alert("warn", f"daily signal failed: {str(e)[:160]}")

    def end_of_day(self) -> None:
        for book in list(self.positions):
            p = self.positions[book]
            self.close(book, p.get("ltp") or p["state"].entry_price, "time", p.get("source") or "?")
        for book in BOOKS:
            self.ledger.set_equity(now_ist().date().isoformat(), book, self.equity(book))
        try:
            from .intraday_store import refresh_intraday
            refresh_intraday()
        except Exception as e:  # noqa: BLE001
            self.alert("warn", f"EOD intraday store update failed: {e}")
        self.p_daily, self.daily_sig = None, {}

    def retrain(self) -> None:
        from ..intraday_model import train as train_intraday
        from ..pipeline import retrain as retrain_daily
        from ..semantic.news_impact import study
        for name, fn in (("daily model", lambda: retrain_daily(self.cfg)), ("intraday model", lambda: train_intraday(self.cfg)),
                         ("news impact", lambda: study(self.cfg))):
            try:
                fn()
                self.alert("info", f"retrained {name}")
            except Exception as e:  # noqa: BLE001
                self.alert("warn", f"retrain {name} failed: {str(e)[:160]}")
        self._load_models()

    # ------------------------------------------------------------------ snapshot for the API
    def equity(self, book: str) -> float:
        return self.cfg.book_capital + self.ledger.realized_pnl(book, "live")

    def _quality(self) -> Dict:
        out = {}
        im = self.intraday_meta or {}
        for h, key in (("30m", "horizon_30m"), ("close", "horizon_close")):
            r = (im.get("metrics") or {}).get(h)
            if not r or not r.get("n"):
                continue
            out[key] = {"n": r["n"], "period": r["period"], "accuracy": r["accuracy"], "auc": r["auc"],
                        "logloss_skill": r["logloss_skill"], "p_value": r["p_value"], "base_rate": r["base_rate"],
                        "hit_rate_at": [{k: t[k] for k in ("threshold", "coverage", "hit_rate")} for t in r["by_threshold"]],
                        "share_above_80": r["share_above_80"], "has_edge": (im.get("has_edge") or {}).get(h, False),
                        "holdout_accuracy": (r.get("holdout") or {}).get("accuracy")}
        dm = (self.daily_meta or {}).get("metrics")
        if dm:
            out["daily"] = {"n": dm.get("n_days"), "period": dm.get("period"), "accuracy": dm.get("accuracy"),
                            "auc": dm.get("auc"), "logloss_skill": dm.get("skill_vs_baserate"),
                            "p_value": dm.get("skill_p_value"), "base_rate": dm.get("up_rate"),
                            "hit_rate_at": [{"threshold": t["threshold"], "coverage": t["coverage"], "hit_rate": t["hit_rate"]}
                                            for t in dm.get("by_threshold", [])],
                            "has_edge": (self.daily_meta or {}).get("has_edge")}
        return out

    def snapshot(self, internal: bool = False) -> Dict:
        now = now_ist()
        with self.lock:
            q = dict(self.quote)
            bars = self.bars1
            status = market_status(now)
            spot = q.get("spot")
            # The session being shown: today while trading, otherwise the last session in the bars.
            session = bars.index[-1].date() if len(bars) and status != "OPEN" else now.date()
            source = q.get("source", "none")
            if status != "OPEN" and len(bars):
                # Outside market hours show one consistent session: its close, range and change.
                spot, source = float(bars["close"].iloc[-1]), f"last session close ({session.isoformat()})"
            prev_close = None
            try:
                from ..data import store
                nf = store.load("nifty")
                if len(nf):
                    prev = nf[nf.index < pd.Timestamp(session)]
                    prev_close = float(prev["close"].iloc[-1]) if len(prev) else None
            except Exception:  # noqa: BLE001
                pass
            quote = {"spot": spot, "prev_close": prev_close, "vix": q.get("vix"), "source": source,
                     "age_s": round(time.time() - q["fetched"], 1) if q.get("fetched") and status == "OPEN" else None}
            if len(bars):
                quote.update({"open": float(bars["open"].iloc[0]), "high": float(bars["high"].max()),
                              "low": float(bars["low"].min())})
                if not spot:
                    quote["spot"] = spot = float(bars["close"].iloc[-1])
            if spot and prev_close:
                quote["change"] = round(spot - prev_close, 2)
                quote["change_pct"] = round((spot / prev_close - 1) * 100, 2)
            positions, by_book = [], {}
            for book, p in self.positions.items():
                st, leg = p["state"], p["leg"]
                ltp = p.get("ltp")
                pub = {"book": book, "id": int(p["trade"]["id"]), "side": leg["typ"], "direction": st.direction,
                       "strike": leg["strike"], "expiry": leg["expiry"], "lots": self.rules.lots, "qty": st.qty,
                       "entry_ts": st.entry_ts.isoformat(), "entry_price": round(st.entry_price, 2),
                       "ltp": None if ltp is None else round(ltp, 2), "price_source": p.get("source"),
                       "held_minutes": round((now - st.entry_ts).total_seconds() / 60, 1), **st.to_public(self.rules, ltp)}
                positions.append(pub)
                by_book[book] = pub
            tr = self.ledger.trades("live")
            books = []
            for b, label in BOOKS.items():
                t = tr[tr["book"] == b] if len(tr) else tr
                closed = t[t["status"] == "CLOSED"] if len(t) else t
                today = closed[closed["date"] == now.date().isoformat()] if len(closed) else closed
                books.append({"name": b, "label": label, "auto": self.auto.get(b, False), "equity": round(self.equity(b), 2),
                              "realized_pnl": round(float(closed["net_pnl"].sum()), 2) if len(closed) else 0.0,
                              "today_pnl": round(float(today["net_pnl"].sum()), 2) if len(today) else 0.0,
                              "trades": int(len(closed)), "open": b in self.positions,
                              "win_rate": round(float((closed["net_pnl"] > 0).mean()), 3) if len(closed) else None})
            pred = {k: self.pred.get(k) for k in ("p_up_30m", "p_up_close", "updated", "top_drivers", "history",
                                                   "voters", "agreement")}
            pred["p_daily"] = self.p_daily
            pred["model_version"] = (self.intraday_meta or {}).get("version")
            pred["has_edge"] = bool(((self.intraday_meta or {}).get("has_edge") or {}).get("30m"))
            snap = {
                "ts": now.isoformat(),
                "market": {"status": status, "session_date": session.isoformat(),
                           "note": {"OPEN": "closes 15:30", "PRE_OPEN": "opens 09:15"}.get(
                               status, f"showing the last session ({session.strftime('%a %d %b')})")},
                "quote": quote,
                "bars": [{"t_ms": int(ts.timestamp() * 1000), "o": float(r.open), "h": float(r.high), "l": float(r.low),
                          "c": float(r.close)} for ts, r in bars.iterrows()] if len(bars) else [],
                "prediction": pred, "model_quality": self._quality(),
                "position": positions[0] if positions else None, "positions": positions, "books": books,
                "agent": {"model": self.cfg.llm_model, "available": bool(self.health.get("llm")),
                          "enabled": self.agent is not None, "last_decision": self.agent_last,
                          "brief": getattr(self, "brief", None), "decisions_today": self.agent_count},
                "news": self.news[:60], "chain": self._chain(), "sources": self.poller.health(),
                "rules": self.rules.to_public(), "alerts": list(self.alerts),
                "health": {**self.health, "uptime_s": int(time.time() - self.started)},
            }
            if internal:
                snap["_features"] = dict(self.features)
                snap["positions_by_book"] = by_book
            return snap

    def _chain(self) -> Optional[Dict]:
        try:
            from ..data.recorder import latest_chain_metrics
            m = latest_chain_metrics()
        except Exception:  # noqa: BLE001
            m = None
        if not m:
            return None
        g = lambda k: m.get(f"near_{k}")  # noqa: E731
        return {"updated": m.get("ts"), "expiry": m.get("near_expiry"), "pcr": g("pcr_oi"), "max_pain": g("max_pain"),
                "call_wall": g("call_wall"), "put_wall": g("put_wall"), "atm_iv": g("atm_iv"),
                "implied_move_pct": g("implied_move_pct"), "gex_cr": g("gex_cr"), "zero_gamma": g("zero_gamma"),
                "rr25": g("rr25")}
