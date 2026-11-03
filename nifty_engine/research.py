"""Pre-declared strategy grid. Every variant is reported (no cherry-picking), so the number
of trials is explicit when judging the best one (see Bailey & Lopez de Prado, deflated SR)."""

from __future__ import annotations

import copy
import itertools
import json
from typing import Dict, List

import pandas as pd

from .config import EngineConfig
from .model import CANDIDATES, evaluate, walk_forward
from .paper.replay import replay
from .pipeline import REPORTS, build_frame


def run_grid(cfg: EngineConfig) -> Dict:
    out: Dict = {"prediction": {}, "strategies": []}
    spec = CANDIDATES[0]
    frames = {}
    for target in ("oc", "cc"):
        c = copy.deepcopy(cfg)
        c.target = target
        F = build_frame(c)
        oos = walk_forward(F, c, spec)
        frames[target] = (F, oos)
        out["prediction"][target] = {k: v for k, v in evaluate(oos).items()
                                     if k not in ("by_year", "calibration")}
    F, oos = frames["oc"]
    rows: List[Dict] = []
    grid = list(itertools.product(["debit_spread", "itm_buy"], [45.0, 100.0], [0.56, 0.60]))
    for style, stop, thr in grid:
        c = copy.deepcopy(cfg)
        c.option_style, c.option_stop_pct, c.prob_threshold = style, stop, thr
        c.max_drawdown_kill_pct, c.max_lots = 100.0, 1
        rows.append(_row("options", f"{style}|stop={'none' if stop >= 100 else int(stop)}|thr={thr}",
                         replay(F, oos, c, "options", capital=100_000)))
    for thr in (0.56, 0.60):
        c = copy.deepcopy(cfg)
        c.prob_threshold, c.max_drawdown_kill_pct, c.max_lots = thr, 100.0, 1
        rows.append(_row("futures", f"future|thr={thr}", replay(F, oos, c, "futures", capital=500_000)))
    out["strategies"] = rows
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "variants.json").write_text(json.dumps(out, indent=2, default=str))
    md = ["| Book | Variant | Trades | Win% | Dir. acc% | Gross/trade | Charges/trade | Net/trade | Net total | Max DD% | Recent 3y net |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['book']} | {r['variant']} | {r['trades']} | {r['win_pct']} | {r['dir_acc_pct']} | "
                  f"{r['gross_per_trade']} | {r['charges_per_trade']} | {r['net_per_trade']} | {r['net_total']} | "
                  f"{r['max_dd_pct']} | {r['net_last_3y']} |")
    (REPORTS / "VARIANTS.md").write_text("\n".join(md), encoding="utf-8")
    return out


def _row(book: str, name: str, res: Dict) -> Dict:
    T, s = res["trades"], res["summary"]
    if T.empty:
        return {"book": book, "variant": name, "trades": 0}
    recent = T[pd.to_datetime(T["date"]) >= pd.to_datetime(T["date"]).max() - pd.Timedelta(days=3 * 365)]
    return {"book": book, "variant": name, "trades": int(len(T)),
            "win_pct": round(float((T["net_pnl"] > 0).mean() * 100), 1),
            "dir_acc_pct": round(float(T["correct_direction"].mean() * 100), 1),
            "gross_per_trade": round(float(T["gross_pnl"].mean()), 0),
            "charges_per_trade": round(float(T["charges"].mean()), 0),
            "net_per_trade": round(float(T["net_pnl"].mean()), 0),
            "net_total": round(float(T["net_pnl"].sum()), 0),
            "max_dd_pct": s.get("max_drawdown_pct"),
            "net_last_3y": round(float(recent["net_pnl"].sum()), 0)}
