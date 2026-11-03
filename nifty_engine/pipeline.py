"""High-level jobs used by the CLI, scheduler and API."""

from __future__ import annotations

import json
import logging
import os
from typing import Dict, Optional

import pandas as pd

from .config import DATA_DIR, EngineConfig
from .data import store
from .data.providers import load_participant_oi
from .features import build_daily_features
from .model import ModelSpec, evaluate, load_champion, train_and_register, walk_forward
from .news import load_news_series
from .paper.replay import replay

log = logging.getLogger(__name__)
REPORTS = DATA_DIR / "reports"


def extras() -> Dict[str, pd.Series]:
    ex = dict(load_news_series())
    poi = load_participant_oi(os.environ.get("NE_PARTICIPANT_OI_DIR", str(DATA_DIR / "nse" / "participant_oi")))
    for c in poi.columns if len(poi) else []:
        ex[c] = poi[c]
    return ex


def build_frame(cfg: EngineConfig, raw: Optional[dict] = None) -> pd.DataFrame:
    raw = raw or store.load_all()
    if raw.get("nifty") is None or raw["nifty"].empty:
        raise RuntimeError("No NIFTY data in store. Run: python -m nifty_engine update-data")
    return build_daily_features(raw, cfg.target, extras())


def build_range_frame(cfg: EngineConfig, raw: Optional[dict] = None) -> pd.DataFrame:
    raw = raw or store.load_all()
    return build_daily_features(raw, "range", extras(), range_k=cfg.condor_k)


def retrain(cfg: EngineConfig) -> Dict:
    F = build_frame(cfg)
    meta = train_and_register(F, cfg)
    if "condor" in cfg.book_list:
        import copy
        c2 = copy.deepcopy(cfg)
        c2.target = "range"
        mr = train_and_register(build_range_frame(cfg), c2, name="champion_range")
        log.info("range model %s acc=%.4f base=%.4f", mr["version"], mr["metrics"]["accuracy"], mr["metrics"]["up_rate"])
        meta["range_model"] = {k: mr[k] for k in ("version", "has_edge")}
    log.info("retrained %s edge=%s acc=%.4f", meta["version"], meta["has_edge"], meta["metrics"]["accuracy"])
    return meta


def backtest(cfg: EngineConfig, last_days: int = 22, capitals=(50_000, 100_000, 200_000, 500_000),
             spec_name: Optional[str] = None) -> Dict:
    """Walk-forward predictions + options/futures paper replay. Writes a JSON+Markdown report."""
    F = build_frame(cfg)
    bundle, meta = load_champion()
    spec = ModelSpec(spec_name or (meta["spec_name"] if meta else "ens_v1"))
    if meta and not spec_name:
        from .model import CANDIDATES
        spec = next((s for s in CANDIDATES if s.name == meta["spec_name"]), spec)
    oos = walk_forward(F, cfg, spec)
    metrics = evaluate(oos)
    out = {"config": cfg.to_dict(), "model_spec": spec.__dict__, "prediction_metrics": metrics,
           "replays": {}}
    last_start = str(oos.index[-last_days].date()) if len(oos) > last_days else None
    for book in cfg.book_list + (["futures"] if "futures" not in cfg.book_list else []):
        for cap in capitals:
            full = replay(F, oos, cfg, book, capital=cap)
            month = replay(F, oos, cfg, book, start=last_start, capital=cap)
            out["replays"][f"{book}_{cap}"] = {"full": full["summary"], "last_month": month["summary"]}
            if cap == cfg.capital or (book == "futures" and cap == 500_000):
                REPORTS.mkdir(parents=True, exist_ok=True)
                full["trades"].to_csv(REPORTS / f"trades_{book}_{cap}.csv", index=False)
                full["equity"].to_csv(REPORTS / f"equity_{book}_{cap}.csv")
    REPORTS.mkdir(parents=True, exist_ok=True)
    oos.to_csv(REPORTS / "oos_predictions.csv")
    (REPORTS / "backtest.json").write_text(json.dumps(out, indent=2, default=str))
    from .report import write_backtest_markdown
    write_backtest_markdown(out, REPORTS / "BACKTEST_REPORT.md")
    return out
