"""Live auto paper trader. Never sends real orders.

Prices: real broker option-chain bid/ask (Upstox today; any LiveProvider tomorrow) when a
token is configured, otherwise modeled from India VIX. Each trade records its price source.
"""

from __future__ import annotations

import datetime as dt
import logging
from typing import Dict, Optional

import pandas as pd

from ..calendar import IST, next_weekly_expiry, now_ist, year_fraction
from ..config import EngineConfig
from ..data import store
from ..data.providers import UpstoxProvider, YahooProvider
from ..features import build_daily_features
from ..model import load_champion, predict_today
from ..options import Leg, Structure, bs_price, build_structure, half_spread, smile_iv
from ..pipeline import extras
from .charges import order_charges
from .ledger import Ledger
from .sim import decide, rates_from, size_option_trade

log = logging.getLogger(__name__)


class LivePaperTrader:
    def __init__(self, cfg: EngineConfig, ledger: Optional[Ledger] = None):
        self.cfg = cfg
        self.ledger = ledger or Ledger()
        self.upstox = UpstoxProvider()
        self.yahoo = YahooProvider()
        self._chain_cache: Dict = {}

    # ------------------------------------------------------------------ market data
    def spot_and_vix(self) -> Dict[str, Optional[float]]:
        if self.upstox.available:
            try:
                q = self.upstox.ltp([UpstoxProvider.NIFTY, UpstoxProvider.VIX])
                vals = list(q.values())
                spot = q.get(UpstoxProvider.NIFTY) or q.get("NSE_INDEX:Nifty 50") or (vals[0] if vals else None)
                vix = q.get(UpstoxProvider.VIX) or q.get("NSE_INDEX:India VIX") or (vals[1] if len(vals) > 1 else None)
                return {"spot": spot, "vix": vix, "source": "upstox"}
            except Exception as e:  # noqa: BLE001
                log.warning("upstox ltp failed: %s", e)
        return {"spot": self.yahoo.last_price("^NSEI"), "vix": self.yahoo.last_price("^INDIAVIX"), "source": "yahoo"}

    def today_open(self) -> Optional[float]:
        try:
            df = self.yahoo.bars("^NSEI", start=(dt.date.today() - dt.timedelta(days=1)).isoformat(), interval="1m")
            df = df[df.index.date == now_ist().date()]
            if len(df):
                return float(df["open"].iloc[0])
        except Exception:  # noqa: BLE001
            pass
        q = self.spot_and_vix()
        return q["spot"]

    def chain(self, expiry: dt.date) -> Optional[pd.DataFrame]:
        if not self.upstox.available:
            return None
        key = (expiry, now_ist().strftime("%H%M"))
        if key not in self._chain_cache:
            try:
                self._chain_cache = {key: self.upstox.option_chain(expiry)}
            except Exception as e:  # noqa: BLE001
                log.warning("option chain failed: %s", e)
                return None
        return self._chain_cache[key]

    def expiry(self) -> dt.date:
        today = now_ist().date()
        if self.upstox.available:
            try:
                exps = [e for e in self.upstox.option_expiries() if (e - today).days >= 1]
                if exps:
                    return exps[0]
            except Exception:  # noqa: BLE001
                pass
        return next_weekly_expiry(today, min_days=1)

    def leg_price(self, leg: Leg, expiry: dt.date, spot: float, vix: float, buying: bool) -> tuple:
        ch = self.chain(expiry)
        if ch is not None and len(ch):
            r = ch[(ch["strike"] == leg.strike) & (ch["type"] == leg.typ)]
            if len(r):
                r = r.iloc[0]
                quote = r["ask"] if buying else r["bid"]
                if quote and quote > 0:
                    leg.instrument_key = r["instrument_key"] or ""
                    px = float(quote) + (self.cfg.option_slippage_pts if buying else -self.cfg.option_slippage_pts)
                    return max(0.05, px), "broker_bid_ask"
                if r["ltp"] and r["ltp"] > 0:
                    mid = float(r["ltp"])
                    c = half_spread(mid) + self.cfg.option_slippage_pts
                    return max(0.05, mid + c if buying else mid - c), "broker_ltp"
        T = year_fraction(now_ist(), dt.datetime.combine(expiry, dt.time(15, 30), tzinfo=IST))
        mid = bs_price(spot, leg.strike, T, smile_iv(vix, spot, leg.strike), leg.typ)
        c = half_spread(mid) + self.cfg.option_slippage_pts
        return max(0.05, mid + c if buying else mid - c), "model_vix"

    # ------------------------------------------------------------------ equity
    def equity(self, book: str) -> float:
        return self.cfg.capital + self.ledger.realized_pnl(book, "live")

    # ------------------------------------------------------------------ signal
    def morning_signal(self) -> Dict:
        bundle, meta = load_champion()
        if bundle is None:
            return {"error": "no trained model; run `python -m nifty_engine train`"}
        raw = store.load_all()
        today = pd.Timestamp(now_ist().date())
        o = self.today_open()
        nf = raw["nifty"]
        if o and (len(nf) == 0 or nf.index[-1] < today):
            nf = pd.concat([nf, pd.DataFrame({"open": [o], "high": [o], "low": [o], "close": [o], "volume": [0]},
                                             index=pd.DatetimeIndex([today]))])
            raw = dict(raw, nifty=nf)
        F = build_daily_features(raw, meta["target"], extras())
        sig = predict_today(F, bundle, meta)
        sig["direction"] = decide(sig["p_up"], self.cfg)
        sig["stance"] = {1: "BULLISH", -1: "BEARISH", 0: "NO_TRADE"}[sig["direction"]]
        if not meta["has_edge"]:
            sig["stance"] = "NO_EDGE (model does not beat base rate out-of-sample)"
        sig["open"] = o
        sig["ts"] = now_ist().isoformat()
        sig["top_features"] = dict(list(meta.get("explain", {}).items())[:8])
        self.ledger.log_signal(sig["ts"], sig)
        return sig

    # ------------------------------------------------------------------ trading
    def enter(self, sig: Dict, allow_no_edge: bool = False) -> Optional[int]:
        today = now_ist().date().isoformat()
        if self.ledger.kv_get(f"entered_{today}"):
            return None
        if sig.get("direction", 0) == 0 or ("NO_EDGE" in sig.get("stance", "") and not allow_no_edge):
            self.ledger.kv_set(f"entered_{today}", "skip")
            return None
        q = self.spot_and_vix()
        spot, vix = q["spot"], q["vix"] or 15.0
        if not spot:
            return None
        if "options" not in self.cfg.book_list:
            return None
        exp = self.expiry()
        st = build_structure(sig["direction"], spot, self.cfg.option_style, self.cfg.spread_width_pts)
        srcs = set()
        net = 0.0
        for l in st.legs:
            l.entry, src = self.leg_price(l, exp, spot, vix, buying=l.qty_sign > 0)
            srcs.add(src)
            net += l.qty_sign * l.entry
        eq = self.equity("options")
        lots = size_option_trade(net, eq, self.cfg)
        self.ledger.kv_set(f"entered_{today}", "done")
        if lots == 0:
            log.info("skip: insufficient capital for %s (net %.2f)", st.style, net)
            return None
        rates = rates_from(self.cfg)
        ch = sum(order_charges("option", "BUY" if l.qty_sign > 0 else "SELL", l.entry * self.cfg.lot_size * lots, rates)["total"]
                 for l in st.legs)
        tid = self.ledger.insert_trade({
            "book": "options", "mode": "live", "date": today, "entry_ts": now_ist().isoformat(),
            "direction": sig["direction"], "style": st.style, "lots": lots, "entry_value": round(net, 2),
            "legs": [{"typ": l.typ, "strike": l.strike, "sign": l.qty_sign, "entry": l.entry,
                      "key": l.instrument_key, "expiry": exp.isoformat()} for l in st.legs],
            "charges": round(ch, 2), "p_up": sig["p_up"], "model_version": sig.get("model_version"),
            "status": "OPEN", "price_source": ",".join(sorted(srcs)),
        })
        log.info("ENTER #%s %s %s lots=%s net=%.2f", tid, st.style, sig["stance"], lots, net)
        return tid

    def mark_and_manage(self, force_exit: bool = False) -> None:
        import json
        open_t = self.ledger.open_trades("options")
        if open_t.empty:
            return
        q = self.spot_and_vix()
        spot, vix = q["spot"], q["vix"] or 15.0
        if not spot:
            return
        for _, t in open_t.iterrows():
            legs = json.loads(t["legs"])
            exp = dt.date.fromisoformat(legs[0]["expiry"])
            st = Structure(int(t["direction"]), t["style"],
                           [Leg(l["typ"], l["strike"], l["sign"], l["entry"], 0.0, l.get("key", "")) for l in legs])
            val = 0.0
            for l in st.legs:
                l.exit, _ = self.leg_price(l, exp, spot, vix, buying=l.qty_sign < 0)
                val += l.qty_sign * l.exit
            entry = float(t["entry_value"])
            reason = None
            if force_exit:
                reason = "time"
            elif val <= entry * (1 - self.cfg.option_stop_pct / 100):
                reason = "stop"
            elif val >= entry * (1 + self.cfg.option_target_pct / 100):
                reason = "target"
            if reason:
                self.close(t, st, val, reason)

    def close(self, t, st: Structure, val: float, reason: str) -> None:
        lots = int(t["lots"])
        rates = rates_from(self.cfg)
        ch_exit = sum(order_charges("option", "SELL" if l.qty_sign > 0 else "BUY", l.exit * self.cfg.lot_size * lots, rates)["total"]
                      for l in st.legs)
        gross = (val - float(t["entry_value"])) * self.cfg.lot_size * lots
        charges = float(t["charges"] or 0) + ch_exit
        legs = [{"typ": l.typ, "strike": l.strike, "sign": l.qty_sign, "entry": l.entry, "exit": l.exit,
                 "key": l.instrument_key} for l in st.legs]
        self.ledger.update_trade(int(t["id"]), exit_ts=now_ist().isoformat(), exit_value=round(val, 2),
                                 gross_pnl=round(gross, 2), charges=round(charges, 2),
                                 net_pnl=round(gross - charges, 2), exit_reason=reason, status="CLOSED", legs=legs)
        log.info("EXIT #%s %s net=%.0f", t["id"], reason, gross - charges)

    def end_of_day(self) -> None:
        d = now_ist().date().isoformat()
        for b in self.cfg.book_list:
            self.ledger.set_equity(d, b, self.equity(b))
