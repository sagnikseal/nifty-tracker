"""CLI: python -m nifty_engine <command>

  update-data       download/refresh daily NIFTY, India VIX and global series (free, no broker)
  train             walk-forward train candidates, promote champion
  backtest          walk-forward predictions + options/futures paper replay -> store/reports/
  signal            print today's signal (after 09:15 IST)
  run               always-on scheduler: data, paper trading, EOD, self-retraining
  serve             API + dashboard at http://localhost:8010/paper
  all               update-data + train + backtest
  fetch-intraday    (Upstox token) download NIFTY & India VIX 5-minute bars since Jan-2022 for the
                    phase-2 intraday model -> store/intraday/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from .config import DATA_DIR, EngineConfig


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="nifty_engine")
    ap.add_argument("command", choices=["update-data", "train", "backtest", "signal", "run", "serve", "all",
                                        "collect-news", "fetch-intraday"])
    ap.add_argument("--capital", type=float)
    ap.add_argument("--books")
    ap.add_argument("--style", choices=["debit_spread", "itm_buy"])
    ap.add_argument("--threshold", type=float)
    ap.add_argument("--target", choices=["oc", "cc"])
    ap.add_argument("--allow-no-edge", action="store_true", help="paper-trade even if model has no proven edge")
    ap.add_argument("--port", type=int, default=8010)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    cfg = EngineConfig.load()
    if a.capital:
        cfg.capital = a.capital
    if a.books:
        cfg.books = a.books
    if a.style:
        cfg.option_style = a.style
    if a.threshold:
        cfg.prob_threshold = a.threshold
    if a.target:
        cfg.target = a.target

    if a.command in ("update-data", "all"):
        from .data.store import refresh_daily
        print(json.dumps(refresh_daily(cfg.extra.get("download_start", "2007-01-01")), indent=1))
    if a.command in ("train", "all"):
        from .pipeline import retrain
        meta = retrain(cfg)
        print(json.dumps({k: meta[k] for k in ("version", "spec_name", "has_edge", "candidates")}, indent=1))
        m = meta["metrics"]
        print(f"OOS accuracy {m['accuracy']:.4f} | AUC {m['auc']:.3f} | logloss {m['logloss']:.4f} "
              f"vs base {m['logloss_baserate']:.4f} | p(coin) {m['p_value_vs_coinflip']:.3g}")
    if a.command in ("backtest", "all"):
        from .pipeline import backtest
        backtest(cfg)
        print(f"Report: {DATA_DIR / 'reports' / 'BACKTEST_REPORT.md'}")
    if a.command == "fetch-intraday":
        import datetime as dt
        from .data.providers import UpstoxProvider
        up = UpstoxProvider()
        if not up.available:
            print("Set UPSTOX_ACCESS_TOKEN (or the TOTP trio) in .env first")
            return 1
        out = DATA_DIR / "intraday"
        out.mkdir(parents=True, exist_ok=True)
        for name, key in (("nifty_5m", up.NIFTY), ("indiavix_5m", up.VIX)):
            df = up.candles(key, "minutes", 5, dt.date(2022, 1, 3), dt.date.today())
            df.to_csv(out / f"{name}.csv")
            print(name, len(df), "bars")
    if a.command == "collect-news":
        from .news import collect_daily
        print(collect_daily())
    if a.command == "signal":
        from .paper.live import LivePaperTrader
        print(json.dumps(LivePaperTrader(cfg).morning_signal(), indent=1, default=str))
    if a.command == "run":
        import threading
        from .scheduler import run_forever
        try:
            import uvicorn
            from .api import app
            threading.Thread(target=lambda: uvicorn.run(app, host="0.0.0.0", port=a.port, log_level="warning"),
                             daemon=True).start()
            print(f"Dashboard: http://localhost:{a.port}/paper")
        except ImportError:
            print("fastapi/uvicorn not installed: running scheduler without dashboard")
        run_forever(cfg, allow_no_edge=a.allow_no_edge)
    if a.command == "serve":
        import uvicorn
        from .api import app
        uvicorn.run(app, host="0.0.0.0", port=a.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
