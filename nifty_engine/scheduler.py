"""Always-on runner: data refresh, live paper trading, EOD bookkeeping and self-retraining.

Daily timeline (IST, weekdays):
  08:40  refresh daily data (global markets closed overnight, Asia opened) + collect news
  09:20  morning signal -> paper entry (if signal & edge)
  09:20-15:15  mark-to-market every `poll_seconds`, stop/target exits
  15:15  forced exit (intraday book)
  15:45  EOD equity, refresh data
  16:15  retrain if due (every Friday, or when the champion is older than retrain_every_days)
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import time
import traceback

from .calendar import now_ist, parse_hhmm
from .config import EngineConfig
from .data import store
from .model import REGISTRY
from .news import collect_daily
from .paper.ledger import Ledger
from .paper.live import LivePaperTrader
from .pipeline import retrain

log = logging.getLogger(__name__)


def _once(ledger: Ledger, key: str) -> bool:
    if ledger.kv_get(key):
        return False
    ledger.kv_set(key, now_ist().isoformat())
    return True


def _retrain_due(cfg: EngineConfig) -> bool:
    champ = REGISTRY / "champion.json"
    if not champ.exists():
        return True
    v = json.loads(champ.read_text())["version"]
    age = (dt.datetime.now() - dt.datetime.strptime(v[:15], "%Y%m%d-%H%M%S")).days
    return now_ist().weekday() == 4 or age >= cfg.retrain_every_days


def run_forever(cfg: EngineConfig, allow_no_edge: bool = False) -> None:
    ledger = Ledger()
    trader = LivePaperTrader(cfg, ledger)
    entry_t, exit_t = parse_hhmm(cfg.entry_time), parse_hhmm(cfg.exit_time)
    log.info("scheduler started: capital=%s books=%s style=%s", cfg.capital, cfg.books, cfg.option_style)
    while True:
        try:
            now = now_ist()
            d = now.date().isoformat()
            t = now.time()
            if now.weekday() < 5:
                if dt.time(8, 40) <= t < dt.time(9, 15) and _once(ledger, f"refresh_am_{d}"):
                    store.refresh_daily()
                    try:
                        collect_daily(now.date())
                    except Exception:  # noqa: BLE001
                        log.warning("news collection failed")
                if entry_t <= t < exit_t:
                    if _once(ledger, f"signal_{d}"):
                        sig = trader.morning_signal()
                        log.info("signal: %s", {k: sig.get(k) for k in ("p_up", "stance", "model_version")})
                        if "error" not in sig:
                            trader.enter(sig, allow_no_edge=allow_no_edge or cfg.paper_trade_without_edge)
                        if "condor" in cfg.book_list:
                            rs = trader.range_signal()
                            log.info("range signal: %s", {k: rs.get(k) for k in ("p_inside", "action")})
                            if rs.get("action") == "SELL_CONDOR":
                                trader.enter_condor(rs)
                    trader.mark_and_manage()
                if exit_t <= t < dt.time(15, 35):
                    trader.mark_and_manage(force_exit=True)
                if t >= dt.time(15, 45) and _once(ledger, f"eod_{d}"):
                    trader.mark_and_manage(force_exit=True)
                    trader.end_of_day()
                    store.refresh_daily()
                if t >= dt.time(16, 15) and _retrain_due(cfg) and _once(ledger, f"retrain_{d}"):
                    meta = retrain(cfg)
                    log.info("retrained -> %s (edge=%s)", meta["version"], meta["has_edge"])
        except Exception:  # noqa: BLE001
            log.error("scheduler error:\n%s", traceback.format_exc())
        time.sleep(cfg.poll_seconds)
