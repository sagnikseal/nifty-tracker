"""CLI: python -m nifty_engine <command>

  update-data       download/refresh daily NIFTY, India VIX and global series (free, no broker)
  train             walk-forward train candidates, promote champion
  backtest          walk-forward predictions + options/futures paper replay -> store/reports/
  signal            print today's signal (after 09:15 IST)
  run               always-on scheduler: data, paper trading, EOD, self-retraining
  serve             API + dashboard at http://localhost:8010/paper
  all               update-data + train + backtest
  fetch-expired-options  (Upstox Plus) real 5-min prices of the expired weekly options the
                    strategy would have traded since --since -> store/options_hist/
  replay-real       replay the options book on those real prices vs the modeled replay
  zoo               walk-forward every model (logit, hgb, rf, et, mlp, svm, xgb, lgbm, cat, lstm,
                    cnn, transformer) + mean/stacked/best-3 ensembles -> store/zoo/
  variants          run the pre-declared strategy grid (all variants reported) -> VARIANTS.md
  fetch-intraday    (Upstox token) download NIFTY & India VIX 5-minute bars since Jan-2022 for the
                    phase-2 intraday model -> store/intraday/

v5 (all free: Upstox Analytics Token + Yahoo/GDELT/RSS + a local Ollama model)
  research          run every pre-registered experiment (feature groups, strategy books, meta-labelling,
                    forward tests) -> store/reports/RESEARCH_REPORT.md   [--only P1,S4] [--quick]
  record            record today's option chain + order-book depth until 15:31 (Analytics Token)
  news-capture      collect headlines now (RSS, Upstox News API, GDELT)
  news-score        score unscored headlines with the local LLM (lexicon if Ollama is off)
  news-backfill     fill the last ~3 months of headlines from GDELT (run once)
  brief             write today's pre-open brief with the local LLM
  build-option-features  rebuild historical option-chain features from expired-option candles
  books             status of the v5 paper books
  llm-check         check the local Ollama model
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
                                        "collect-news", "fetch-intraday", "variants",
                                        "fetch-expired-options", "replay-real", "zoo",
                                        "research", "record", "news-capture", "news-score", "news-backfill",
                                        "brief", "build-option-features", "books", "llm-check"])
    ap.add_argument("--only", help="research: comma list of experiment ids (e.g. P1,S4)")
    ap.add_argument("--quick", action="store_true", help="research: logistic model only (fast)")
    ap.add_argument("--members", help="comma list for `zoo` (default: every installed model)")
    ap.add_argument("--no-deep", action="store_true", help="zoo without PyTorch models")
    ap.add_argument("--fresh", action="store_true", help="zoo: discard cached member predictions (use after new data)")
    ap.add_argument("--promote", action="store_true", help="zoo: deploy best-3 ensemble if it beats the champion")
    ap.add_argument("--max-expiries", type=int)
    ap.add_argument("--since", default="2024-10-01")
    ap.add_argument("--mode", default="itm", choices=["itm", "wings", "full"])
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
        from .data.global_markets import refresh_hourly
        from .data.store import refresh_daily
        print(json.dumps(refresh_daily(cfg.extra.get("download_start", "2007-01-01")), indent=1))
        print(json.dumps({"hourly": refresh_hourly()}, indent=1))
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
    if a.command == "fetch-expired-options":
        from .data.expired_options import download
        print(json.dumps(download(start=a.since, max_expiries=a.max_expiries, mode=a.mode), indent=1))
    if a.command == "replay-real":
        from .paper.real_replay import replay_real
        res = replay_real(cfg, start=a.since)
        print(json.dumps(res["summary"], indent=1, default=str))
    if a.command == "zoo":
        from .pipeline import build_frame
        from .zoo import run_zoo
        import shutil
        if a.fresh:
            shutil.rmtree(DATA_DIR / "zoo", ignore_errors=True)
        F = build_frame(cfg)
        res = run_zoo(F, cfg, members=a.members.split(",") if a.members else None, include_deep=not a.no_deep)
        print(json.dumps(res["metrics"], indent=1))
        if a.promote:
            from .zoo import promote_best3
            print(json.dumps(promote_best3(F, cfg, res), indent=1, default=str))
    if a.command == "variants":
        from .research import run_grid
        run_grid(cfg)
        print((DATA_DIR / "reports" / "VARIANTS.md").read_text(encoding="utf-8"))
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
    # ------------------------------------------------------------------ v5
    if a.command == "research":
        from .research.experiments import REPORTS, run_all
        out = run_all(cfg, only=a.only.split(",") if a.only else None, quick=a.quick)
        print((REPORTS / "RESEARCH_REPORT.md").read_text(encoding="utf-8"))
        print(f"\nFull results: {REPORTS / 'research.json'}  ({len(out['results'])} experiments)")
    if a.command == "record":
        from .data.recorder import run_session
        print(json.dumps(run_session(cfg), indent=1, default=str))
    if a.command == "news-capture":
        from .semantic.news_llm import capture
        print(json.dumps(capture(cfg), indent=1))
    if a.command == "news-score":
        from .semantic.news_llm import score_pending
        print(json.dumps(score_pending(cfg, max_items=5000), indent=1))
    if a.command == "news-backfill":
        from .semantic.news_llm import backfill_gdelt, score_pending
        print(json.dumps(backfill_gdelt(), indent=1))
        print(json.dumps(score_pending(cfg, max_items=20000), indent=1))
    if a.command == "brief":
        from .semantic.market_brief import build_state, write_brief
        from .paper.ledger import Ledger
        sig = Ledger().last_signal()
        out = write_brief(cfg, build_state(cfg, sig))
        print(json.dumps({k: v for k, v in out.items() if k != "state"}, indent=1, default=str))
    if a.command == "build-option-features":
        from .data import options_hist
        from .intraday import load_5m
        from .research.experiments import Lab
        X = options_hist.build(load_5m("nifty_5m"), Lab(cfg).real_expiries(), cfg.lot_size)
        print(f"{len(X)} sessions -> {options_hist.FEATURES_FILE}")
    if a.command == "books":
        from .paper.books_live import BookRunner
        from .paper.live import LivePaperTrader
        print(json.dumps(BookRunner(cfg, LivePaperTrader(cfg)).scorecard(), indent=1))
    if a.command == "llm-check":
        from .semantic.llm import LocalLLM
        llm = LocalLLM(cfg.llm_model, cfg.llm_url)
        ok = llm.available()
        print(json.dumps({"url": llm.url, "model": llm.model, "available": ok,
                          "hint": None if ok else f"install Ollama, run `ollama serve`, then `ollama pull {llm.model}`"}, indent=1))
        if ok:
            from .semantic.news_llm import BATCH_SCHEMA, SYSTEM
            print(json.dumps(llm.json_chat(SYSTEM.replace("{heavy}", "HDFCBANK, RELIANCE"),
                                           "id=t1: RBI unexpectedly cuts repo rate by 50 bps", BATCH_SCHEMA), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
