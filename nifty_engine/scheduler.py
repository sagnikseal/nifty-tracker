"""Always-on runner: data refresh, news, live recording, paper trading, EOD bookkeeping and
self-retraining. Everything is free: Yahoo/GDELT/RSS data, the Upstox Analytics Token you
already have, and a local Ollama model.

Daily timeline (IST, weekdays):
  08:40  refresh daily + hourly global data, capture and score news (local LLM)
  09:10  start the live recorder (option chain + depth) if an Upstox token is set
  09:20  v4 morning signal -> paper entry; range signal; local-LLM pre-open brief
  09:20-15:25  v5 strategy books (each at its own time), mark-to-market, stops/targets
  15:15  v4 intraday exit (v5 books exit on their own rules)
  every 30 min  news capture
  15:45  EOD equity, outcomes, refresh data, news
  16:15  retrain if due (every Friday, or when the champion is older than retrain_every_days)
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import threading
import time
import traceback

from .calendar import now_ist, parse_hhmm
from .config import EngineConfig
from .data import store
from .model import REGISTRY
from .news import collect_daily
from .paper.books_live import BookRunner
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


def _safe(name, fn, *a, **kw):
    try:
        return fn(*a, **kw)
    except Exception as e:  # noqa: BLE001
        log.warning("%s failed: %s", name, str(e)[:300])
        return None


def news_cycle(cfg: EngineConfig) -> None:
    from .semantic.news_llm import capture, score_pending
    log.info("news capture: %s", _safe("news capture", capture, cfg))
    log.info("news scoring: %s", _safe("news scoring", score_pending, cfg))


def run_forever(cfg: EngineConfig, allow_no_edge: bool = False) -> None:
    ledger = Ledger()
    trader = LivePaperTrader(cfg, ledger)
    runner = BookRunner(cfg, trader)
    entry_t, exit_t = parse_hhmm(cfg.entry_time), parse_hhmm(cfg.exit_time)
    recorder_thread = None
    last_news = 0.0
    log.info("scheduler started: capital=%s books=%s v5 books=%s", cfg.capital, cfg.books, cfg.strategy_books)
    while True:
        try:
            now = now_ist()
            d = now.date().isoformat()
            t = now.time()
            if now.weekday() < 5:
                if dt.time(8, 40) <= t < dt.time(9, 15) and _once(ledger, f"refresh_am_{d}"):
                    store.refresh_daily()
                    from .data.global_markets import refresh_hourly
                    _safe("hourly refresh", refresh_hourly)
                    _safe("lexicon news", collect_daily, now.date())
                    news_cycle(cfg)
                    last_news = time.time()
                if (cfg.record_live and dt.time(9, 10) <= t < dt.time(15, 25) and trader.upstox.available
                        and (recorder_thread is None or not recorder_thread.is_alive()) and _once(ledger, f"recorder_{d}")):
                    from .data.recorder import run_session
                    recorder_thread = threading.Thread(target=lambda: log.info("recorder: %s", _safe("recorder", run_session, cfg)),
                                                       daemon=True)
                    recorder_thread.start()
                if entry_t <= t < exit_t:
                    if _once(ledger, f"signal_{d}"):
                        sig = trader.morning_signal()
                        log.info("signal: %s", {k: sig.get(k) for k in ("p_up", "stance", "model_version")})
                        if "error" not in sig:
                            trader.enter(sig, allow_no_edge=allow_no_edge or cfg.paper_trade_without_edge)
                        if "condor" in cfg.book_list or "iron_condor" in cfg.strategy_list:
                            rs = _safe("range signal", trader.range_signal) or {}
                            if "p_inside" in rs:
                                ledger.kv_set(f"range_{d}", rs)
                            if "condor" in cfg.book_list and rs.get("action") == "SELL_CONDOR":
                                trader.enter_condor(rs)
                        if "llm_analyst" in cfg.strategy_list:
                            from .semantic.market_brief import build_state, write_brief
                            br = _safe("brief", lambda: write_brief(cfg, build_state(cfg, sig)))
                            log.info("brief: %s", (br or {}).get("brief", {}).get("stance") or (br or {}).get("error"))
                    trader.mark_and_manage()
                if dt.time(9, 15) <= t < dt.time(15, 26):
                    runner.step(now)
                if exit_t <= t < dt.time(15, 35):
                    trader.mark_and_manage(force_exit=True)
                if dt.time(9, 15) <= t < dt.time(15, 30) and time.time() - last_news > 1800:
                    news_cycle(cfg)
                    last_news = time.time()
                if t >= dt.time(15, 45) and _once(ledger, f"eod_{d}"):
                    trader.mark_and_manage(force_exit=True)
                    trader.end_of_day()
                    runner.end_of_day()
                    store.refresh_daily()
                    news_cycle(cfg)
                if t >= dt.time(16, 15) and _retrain_due(cfg) and _once(ledger, f"retrain_{d}"):
                    meta = retrain(cfg)
                    log.info("retrained -> %s (edge=%s)", meta["version"], meta["has_edge"])
            elif t >= dt.time(18, 0) and _once(ledger, f"news_weekend_{d}"):
                news_cycle(cfg)
        except Exception:  # noqa: BLE001
            log.error("scheduler error:\n%s", traceback.format_exc())
        time.sleep(cfg.poll_seconds)
