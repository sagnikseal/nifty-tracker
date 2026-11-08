"""Live inputs for the extra feature groups, built so that live features are computed by exactly the same
functions as in training (intraday_model.add_groups, analytics.chain_intraday.day_features).

Every 5-minute close the runtime asks for:
  * the last ~10 stored sessions + today's 5-minute bars for NIFTY, VIX, Bank Nifty, the near-month
    NIFTY future and the heavyweight stocks (Upstox v3 intraday candles; store/intraday for history);
  * today's 1-minute candles of the at-the-money option strikes (fixed for the day at the first call:
    ATM +/- 300 points, calls and puts, nearest expiry at least one day away) -> option-chain features;
  * hourly global bars (Yahoo; refreshed hourly).
Without an Upstox token only NIFTY/VIX (Yahoo) are available and the other groups stay empty (NaN),
which the dashboard flags.
"""

from __future__ import annotations

import datetime as dt
import logging
import time
from typing import Dict, List, Optional

import pandas as pd

from ..calendar import IST, now_ist
from ..data.providers import UpstoxProvider

log = logging.getLogger(__name__)


class LiveFeatureInputs:
    def __init__(self, up: UpstoxProvider):
        self.up = up
        self.keys: Dict[str, str] = {}
        self.day: Optional[dt.date] = None
        self.opt_keys: List[Dict] = []
        self.opt_expiry: Optional[dt.date] = None
        self._hourly_t = 0.0
        self._hourly: Dict[str, pd.DataFrame] = {}
        self._hist: Dict[str, pd.DataFrame] = {}

    # ------------------------------------------------------------------ keys
    def _resolve(self) -> None:
        if self.keys:
            return
        from ..data.instruments import equity_keys, nifty_futures
        from ..intraday_model import STOCKS
        k = {"banknifty": UpstoxProvider.BANKNIFTY}
        try:
            fut = nifty_futures()
            if fut:
                k["niftyfut"] = fut[0]["key"]
            sym = {s.upper(): s for s in STOCKS}
            sym["BHARTIARTL"] = "bhartiartl"
            for ts, key in equity_keys(list(sym)).items():
                k[sym[ts]] = key
        except Exception as e:  # noqa: BLE001
            log.warning("instrument keys failed: %s", e)
        self.keys = k

    def _history(self, name: str) -> pd.DataFrame:
        if name not in self._hist:
            from ..intraday_model import load_bars
            b = load_bars(f"{name}_5m")
            if len(b):
                b = b[b.index >= b.index.max() - pd.Timedelta(days=21)][["open", "high", "low", "close", "volume", "oi"]]
            self._hist[name] = b
        return self._hist[name]

    def _today5(self, key: str) -> pd.DataFrame:
        try:
            df = self.up.intraday_today(key, "minutes", 5)
        except Exception as e:  # noqa: BLE001
            log.debug("intraday %s failed: %s", key, e)
            return pd.DataFrame()
        if df.empty:
            return df
        now = pd.Timestamp(now_ist())
        return df[(df.index + pd.Timedelta(minutes=5) <= now) & (df.index.date == now.date())]

    # ------------------------------------------------------------------ public
    def series(self, nifty5: pd.DataFrame, vix5: pd.DataFrame) -> Dict[str, object]:
        """Same structure as intraday_model.load_series(), for history + today."""
        today = now_ist().date()
        if self.day != today:
            self.day, self._hist, self.opt_keys = today, {}, []
        out: Dict[str, object] = {"nifty": nifty5, "vix": vix5, "stocks": {}, "hourly": self.hourly()}
        if not self.up.available:
            return out
        self._resolve()
        from ..intraday_model import STOCKS
        for name, key in self.keys.items():
            t = self._today5(key)
            h = self._history(name)
            df = pd.concat([h, t]) if len(h) else t
            if len(df):
                df = df[~df.index.duplicated(keep="last")].sort_index()
            if name in STOCKS:
                if len(df):
                    out["stocks"][name] = df
            else:
                out[name] = df
            time.sleep(0.05)
        return out

    def hourly(self) -> Dict[str, pd.DataFrame]:
        if time.time() - self._hourly_t > 3300 or not self._hourly:
            self._hourly_t = time.time()
            try:
                from ..data.global_markets import load_hourly, refresh_hourly
                refresh_hourly()
                self._hourly = {n: load_hourly(n) for n in ("es", "nq", "usdinr", "crude", "gold_f")}
            except Exception as e:  # noqa: BLE001
                log.debug("hourly refresh failed: %s", e)
        return self._hourly

    def chain_rows(self, spot5: pd.Series, decision: pd.Timestamp) -> Optional[Dict]:
        """Option-chain features at `decision` from today's 1-minute candles of the ATM strikes."""
        if not self.up.available:
            return None
        from ..analytics.chain_intraday import day_features
        if not self.opt_keys:
            try:
                exps = [e for e in self.up.option_expiries() if (e - decision.date()).days >= 1]
                if not exps:
                    return None
                self.opt_expiry = exps[0]
                ch = self.up.option_chain(self.opt_expiry)
                atm = round(float(spot5.iloc[-1]) / 50) * 50
                sel = ch[(ch["strike"] >= atm - 300) & (ch["strike"] <= atm + 300)]
                self.opt_keys = [{"key": r["instrument_key"], "strike": float(r["strike"]), "type": r["type"]}
                                 for _, r in sel.iterrows() if r.get("instrument_key")]
            except Exception as e:  # noqa: BLE001
                log.warning("option keys failed: %s", e)
                return None
        frames = []
        for o in self.opt_keys:
            try:
                df = self.up.intraday_today(o["key"], "minutes", 1)
            except Exception:  # noqa: BLE001
                continue
            if len(df):
                df = df.reset_index().rename(columns={"index": "ts"})
                df["strike"], df["type"] = o["strike"], o["type"]
                frames.append(df[["ts", "close", "volume", "oi", "strike", "type"]])
            time.sleep(0.05)
        if not frames:
            return None
        opt_day = pd.concat(frames)
        decisions = [t + pd.Timedelta(minutes=5) for t in spot5.index if t.date() == decision.date()]
        rows = day_features(opt_day, spot5, [d for d in decisions if d <= decision], self.opt_expiry)
        return rows[-1] if rows else None
