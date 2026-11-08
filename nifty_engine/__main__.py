"""CLI: python -m nifty_engine <command>

Everyday
  app               start everything (live data, news, LLM agent, paper books) + open the dashboard
                    at http://localhost:8010   [--no-browser] [--port 8010] [--host 127.0.0.1]
  run               same as `app` without opening a browser
  serve             dashboard + reports only (no live engine)

Models and tests (all walk-forward, out-of-sample)
  intraday-train    movement model: P(up next 30 min) and P(up by 15:15) every 5 minutes -> INTRADAY_REPORT.md
                    [--groups all|tech,chain,...] [--members cat,xgb] [--quick]
  intraday-zoo      stacked zoo: CatBoost, XGBoost + TCN, PatchTST, GRU (+ --tsfm) -> INTRADAY_ZOO_REPORT.md
                    run on the Mac (uses the Apple GPU); [--quick] [--seq tcn,patchtst] [--tsfm chronos-bolt-small]
  intraday-ablation which feature groups help (pre-registered, identical rows) -> INTRADAY_ABLATION_REPORT.md
  atm-replay        your ATM rules on real 1-minute option prices -> ATM_REPLAY_REPORT.md
  news-impact       how NIFTY moved after each LLM-read headline; learned reactions -> NEWS_IMPACT_REPORT.md
  llm-bench         compare local LLMs (Ollama) on labelled headlines + agent scenarios [--models a,b]
  sources-check     test every news source from this machine -> SOURCES_REPORT.md
  data-report       what data exists, what is missing, how to get it -> DATA_REPORT.md
  research          every pre-registered experiment -> RESEARCH_REPORT.md  [--only P1,S4] [--quick]
  train / backtest  daily model (09:20, open->close) walk-forward train / backtest
  zoo               daily model zoo [--fresh] [--promote]

Data
  update-data       daily NIFTY, India VIX and global series (free)
  refresh-intraday  append recent 1-/5-minute bars to store/intraday (Upstox, else Yahoo)
  fetch-intraday    (Upstox) 5-minute NIFTY + VIX since 2022
  fetch-history     (Upstox) Bank Nifty, 10 heavyweights (5-min since 2022) + continuous NIFTY futures (volume, OI)
  fetch-expired-options  (Upstox Plus) real prices of expired weekly options [--since] [--mode]
  news-capture / news-score / news-backfill   headlines now / score with the LLM / ~3 months of GDELT
  record            record today's option chain + depth (Upstox)
  llm-check         is the local LLM reachable?
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from .config import DATA_DIR, EngineConfig

COMMANDS = ["app", "run", "serve", "intraday-train", "intraday-zoo", "intraday-ablation", "fetch-history", "atm-replay", "news-impact", "llm-bench", "sources-check",
            "data-report", "refresh-intraday", "update-data", "train", "backtest", "signal", "all", "collect-news",
            "fetch-intraday", "variants", "fetch-expired-options", "replay-real", "zoo", "research", "record",
            "news-capture", "news-score", "news-backfill", "brief", "build-option-features", "books", "llm-check"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="nifty_engine", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=COMMANDS)
    ap.add_argument("--only", help="research: comma list of experiment ids (e.g. P1,S4)")
    ap.add_argument("--quick", action="store_true", help="research / intraday-train: faster, fewer refits")
    ap.add_argument("--members", help="zoo / intraday-train: comma list of model members")
    ap.add_argument("--models", help="llm-bench: comma list of Ollama models")
    ap.add_argument("--tsfm", help="intraday-zoo: foundation-model features, e.g. chronos-bolt-small,timesfm-2.5")
    ap.add_argument("--seq", default="tcn,patchtst,gru", help="intraday-zoo: sequence members ('' = none)")
    ap.add_argument("--groups", help="intraday-train: extra feature groups, 'all' or comma list (see INTRADAY_GROUPS)")
    ap.add_argument("--no-deep", action="store_true", help="zoo without PyTorch models")
    ap.add_argument("--fresh", action="store_true", help="zoo: discard cached member predictions")
    ap.add_argument("--promote", action="store_true", help="zoo: deploy best-3 ensemble if it beats the champion")
    ap.add_argument("--max-expiries", type=int)
    ap.add_argument("--since", default="2024-10-01")
    ap.add_argument("--mode", default="itm", choices=["itm", "wings", "full"])
    ap.add_argument("--capital", type=float)
    ap.add_argument("--threshold", type=float)
    ap.add_argument("--port", type=int, default=8010)
    ap.add_argument("--host", default="127.0.0.1", help="keep 127.0.0.1 (this machine only)")
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    for noisy in ("urllib3", "httpx", "uvicorn.access"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    cfg = EngineConfig.load()
    if a.capital:
        cfg.capital = a.capital
    if a.threshold:
        cfg.prob_threshold = a.threshold
    c = a.command

    if c in ("app", "run"):
        from .api import serve
        from .live.runtime import Runtime
        print("Starting the NIFTY engine (paper trading only; no real orders can be placed) ...")
        rt = Runtime(cfg)
        print(f"Dashboard: http://localhost:{a.port}/   (Ctrl+C to stop)")
        serve(rt, host=a.host, port=a.port, open_browser=(c == "app" and not a.no_browser))
        return 0
    if c == "serve":
        from .api import serve
        print(f"Reports-only dashboard: http://localhost:{a.port}/")
        serve(None, host=a.host, port=a.port, open_browser=not a.no_browser)
        return 0
    if c == "intraday-train":
        from .intraday_model import INTRADAY_GROUPS, train
        members = tuple(a.members.split(",")) if a.members else ("cat", "xgb")
        groups = list(INTRADAY_GROUPS) if a.groups == "all" else (a.groups.split(",") if a.groups else [])
        meta = train(cfg, members=members, quick=a.quick, groups=groups)
        print(json.dumps({"version": meta["version"], "deployed": meta["deployed_member"], "has_edge": meta["has_edge"]}, indent=1))
        print(f"Report: {DATA_DIR / 'reports' / 'INTRADAY_REPORT.md'}")
    if c == "intraday-zoo":
        from .intraday_stack import run as run_zoo
        tab = tuple(a.members.split(",")) if a.members else ("cat", "xgb")
        run_zoo(cfg, tab_members=tab, seq_archs=tuple(x for x in a.seq.split(",") if x),
                tsfm=a.tsfm.split(",") if a.tsfm else None, quick=a.quick)
        print((DATA_DIR / "reports" / "INTRADAY_ZOO_REPORT.md").read_text(encoding="utf-8"))
    if c == "intraday-ablation":
        from .intraday_model import ablation, build_training_frame_groups
        members = tuple(a.members.split(",")) if a.members else ("cat", "xgb")
        ablation(build_training_frame_groups(cfg), members=members, refit_days=63 if a.quick else 21)
        print((DATA_DIR / "reports" / "INTRADAY_ABLATION_REPORT.md").read_text(encoding="utf-8"))
    if c == "fetch-history":
        from .data.upstox_history import fetch_all
        print(json.dumps(fetch_all(), indent=1, default=str))
    if c == "atm-replay":
        from .paper.atm_replay import run_all
        res = run_all(cfg)
        print(json.dumps({e: {k: {x: v.get(x) for x in ("trades", "win_rate", "net_per_trade", "t_stat")}
                              for k, v in r["variants"].items()} for e, r in res.items()}, indent=1, default=str))
        print(f"Report: {DATA_DIR / 'reports' / 'ATM_REPLAY_REPORT.md'}")
    if c == "news-impact":
        from .semantic.news_impact import REPORT, study
        out = study(cfg)
        print(json.dumps({k: out.get(k) for k in ("status", "events", "days", "oos")}, indent=1, default=str))
        print(f"Report: {REPORT}")
    if c == "llm-bench":
        from .agent.bench import REPORT, run
        out = run(cfg, a.models.split(",") if a.models else None)
        print((REPORT).read_text(encoding="utf-8"))
        if not out["results"]:
            return 1
    if c == "sources-check":
        from .data.news_sources import check_all, write_check_report
        rows = check_all()
        p = DATA_DIR / "reports" / "SOURCES_REPORT.md"
        write_check_report(rows, p)
        print(p.read_text(encoding="utf-8"))
    if c == "data-report":
        from .data_report import write
        write(cfg)
        print((DATA_DIR / "reports" / "DATA_REPORT.md").read_text(encoding="utf-8"))
    if c == "refresh-intraday":
        from .live.intraday_store import refresh_intraday
        print(json.dumps(refresh_intraday(), indent=1))

    # ------------------------------------------------------------------ daily model + data
    if c in ("update-data", "all"):
        from .data.global_markets import refresh_hourly
        from .data.store import refresh_daily
        print(json.dumps(refresh_daily(cfg.extra.get("download_start", "2007-01-01")), indent=1))
        print(json.dumps({"hourly": refresh_hourly()}, indent=1))
    if c in ("train", "all"):
        from .pipeline import retrain
        meta = retrain(cfg)
        m = meta["metrics"]
        print(f"OOS accuracy {m['accuracy']:.4f} | AUC {m['auc']:.3f} | logloss {m['logloss']:.4f} "
              f"vs base {m['logloss_baserate']:.4f} | p(coin) {m['p_value_vs_coinflip']:.3g}")
    if c in ("backtest", "all"):
        from .pipeline import backtest
        backtest(cfg)
        print(f"Report: {DATA_DIR / 'reports' / 'BACKTEST_REPORT.md'}")
    if c == "fetch-intraday":
        import datetime as dt
        from .data.providers import UpstoxProvider
        up = UpstoxProvider()
        if not up.available:
            print("Set UPSTOX_ANALYTICS_TOKEN in .env first")
            return 1
        out = DATA_DIR / "intraday"
        out.mkdir(parents=True, exist_ok=True)
        for name, key, mins in (("nifty_5m", up.NIFTY, 5), ("indiavix_5m", up.VIX, 5), ("nifty_1m", up.NIFTY, 1)):
            df = up.candles(key, "minutes", mins, dt.date(2022, 1, 3), dt.date.today())
            df.to_csv(out / f"{name}.csv")
            print(name, len(df), "bars")
    if c == "fetch-expired-options":
        from .data.expired_options import download
        print(json.dumps(download(start=a.since, max_expiries=a.max_expiries, mode=a.mode), indent=1))
    if c == "replay-real":
        from .paper.real_replay import replay_real
        print(json.dumps(replay_real(cfg, start=a.since)["summary"], indent=1, default=str))
    if c == "zoo":
        import shutil
        from .pipeline import build_frame
        from .zoo import promote_best3, run_zoo
        if a.fresh:
            shutil.rmtree(DATA_DIR / "zoo", ignore_errors=True)
        F = build_frame(cfg)
        res = run_zoo(F, cfg, members=a.members.split(",") if a.members else None, include_deep=not a.no_deep)
        print(json.dumps(res["metrics"], indent=1))
        if a.promote:
            print(json.dumps(promote_best3(F, cfg, res), indent=1, default=str))
    if c == "variants":
        from .research import run_grid
        run_grid(cfg)
        print((DATA_DIR / "reports" / "VARIANTS.md").read_text(encoding="utf-8"))
    if c == "collect-news":
        from .news import collect_daily
        print(collect_daily())
    if c == "signal":
        from .paper.live import LivePaperTrader
        print(json.dumps(LivePaperTrader(cfg).morning_signal(), indent=1, default=str))
    if c == "research":
        from .research.experiments import REPORTS, run_all
        out = run_all(cfg, only=a.only.split(",") if a.only else None, quick=a.quick)
        print((REPORTS / "RESEARCH_REPORT.md").read_text(encoding="utf-8"))
        print(f"\nFull results: {REPORTS / 'research.json'}  ({len(out['results'])} experiments)")
    if c == "record":
        from .data.recorder import run_session
        print(json.dumps(run_session(cfg), indent=1, default=str))
    if c == "news-capture":
        from .semantic.news_llm import capture
        print(json.dumps(capture(cfg), indent=1))
    if c == "news-score":
        from .semantic.news_llm import score_pending
        print(json.dumps(score_pending(cfg, max_items=5000), indent=1))
    if c == "news-backfill":
        from .semantic.news_llm import backfill_gdelt, score_pending
        print(json.dumps(backfill_gdelt(), indent=1))
        print(json.dumps(score_pending(cfg, max_items=20000), indent=1))
    if c == "brief":
        from .paper.ledger import Ledger
        from .semantic.market_brief import build_state, write_brief
        out = write_brief(cfg, build_state(cfg, Ledger().last_signal()))
        print(json.dumps({k: v for k, v in out.items() if k != "state"}, indent=1, default=str))
    if c == "build-option-features":
        from .data import options_hist
        from .intraday import load_5m
        from .research.experiments import Lab
        X = options_hist.build(load_5m("nifty_5m"), Lab(cfg).real_expiries(), cfg.lot_size)
        print(f"{len(X)} sessions -> {options_hist.FEATURES_FILE}")
    if c == "books":
        from .paper.books_live import BookRunner
        from .paper.live import LivePaperTrader
        print(json.dumps(BookRunner(cfg, LivePaperTrader(cfg)).scorecard(), indent=1))
    if c == "llm-check":
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
