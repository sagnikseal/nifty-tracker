"""The local-LLM trading agent (PAPER ONLY).

Every 5 minutes it receives a compact JSON state (price action, India VIX, the movement model's
probabilities and their MEASURED accuracy, the open position and its rule state, fresh headlines
with the LLM's reading and their learned impact, the option chain) and returns a structured
decision. Guardrails in code always win: the agent cannot trade outside the entry window, exceed
the daily trade limit, size up, or override the stop/target/time rules.

Every decision is logged with its p_up_30m and scored 30 minutes later against what NIFTY did,
so the agent's accuracy is measured, shown in the dashboard and compared with the model.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from typing import Dict, List, Optional

import pandas as pd

from ..calendar import IST, now_ist
from ..config import DATA_DIR, EngineConfig
from ..semantic.llm import LocalLLM, clamp, pick
from ..strategies.atm_rules import ATMRules
from .prompts import agent_system

log = logging.getLogger(__name__)
LOG_DIR = DATA_DIR / "agent"
ACTIONS = ["ENTER_LONG", "ENTER_SHORT", "EXIT", "HOLD", "STAND_ASIDE"]

SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ACTIONS},
        "confidence": {"type": "number"},
        "p_up_30m": {"type": "number"},
        "reasoning": {"type": "string"},
        "key_factors": {"type": "array", "items": {"type": "string"}},
        "risks": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["action", "confidence", "p_up_30m", "reasoning", "key_factors", "risks"],
}


def _r(v, nd=4):
    try:
        return None if v is None or v != v else round(float(v), nd)
    except (TypeError, ValueError):
        return None


def build_state(snap: Dict, position: Optional[Dict], trades_today: int, rules: ATMRules) -> Dict:
    """Compact, number-rounded state for the prompt, from the runtime's snapshot."""
    q = snap.get("quote") or {}
    pr = snap.get("prediction") or {}
    mq = snap.get("model_quality") or {}
    feats = snap.get("_features") or {}
    now = now_ist()
    st = {
        "time_ist": now.strftime("%H:%M"), "minutes_to_exit": max(0, int((dt.datetime.combine(now.date(), dt.time(15, 15), tzinfo=IST) - now).total_seconds() // 60)),
        "nifty": {"spot": _r(q.get("spot"), 1), "change_vs_prev_close_pct": _r(q.get("change_pct"), 2),
                  "move_since_open_pct": _r((feats.get("r_open_T") or 0) * 100, 2),
                  "last_30min_pct": _r((feats.get("r_30m") or 0) * 100, 2), "last_60min_pct": _r((feats.get("r_60m") or 0) * 100, 2),
                  "position_in_day_range_0_1": _r(feats.get("range_pos"), 2), "opening_gap_pct": _r((feats.get("gap") or 0) * 100, 2)},
        "india_vix": {"level": _r(q.get("vix"), 2), "change_pct": _r(q.get("vix_change_pct"), 2)},
        "quant_model": {"p_up_next_30m": _r(pr.get("p_up_30m"), 3), "p_up_by_1515": _r(pr.get("p_up_close"), 3),
                        "daily_model_p_close_above_open": _r(pr.get("p_daily"), 3),
                        "measured_accuracy_30m": _r((mq.get("horizon_30m") or {}).get("accuracy"), 3),
                        "has_proven_edge_30m": bool((mq.get("horizon_30m") or {}).get("has_edge")),
                        "note": "accuracy is out-of-sample over 2023-2026; ~0.50 means no better than a coin flip"},
        "position": None, "trades_today": trades_today, "max_trades_per_day": rules.max_trades_per_day,
        "entry_window_open": rules.entry_start <= now.strftime("%H:%M") <= rules.entry_end,
    }
    if position:
        st["position"] = {k: position.get(k) for k in ("side", "strike", "entry_price", "ltp", "pnl_inr", "target_hit",
                                                        "extend_until", "against_count", "rule_state")}
        st["position"]["held_minutes"] = position.get("held_minutes")
    news = []
    for n in (snap.get("news") or [])[:40]:
        llm = n.get("llm") or {}
        if (n.get("age_min") or 999) > 180 or (llm.get("relevance") or 0) < 0.4:
            continue
        news.append({"age_min": n.get("age_min"), "source": n.get("source"), "headline": n.get("title"),
                     "llm_direction": _r(llm.get("direction"), 2), "magnitude": _r(llm.get("magnitude"), 2),
                     "surprise": _r(llm.get("surprise"), 2), "channel": llm.get("channel"),
                     "learned_30m_impact_bps": _r((n.get("impact") or {}).get("exp_bps_30m"), 1)})
        if len(news) >= 12:
            break
    st["fresh_news"] = news
    ch = snap.get("chain")
    if ch:
        st["option_chain"] = {k: ch.get(k) for k in ("pcr", "max_pain", "call_wall", "put_wall", "atm_iv", "implied_move_pct")}
    return st


class TradingAgent:
    def __init__(self, cfg: EngineConfig, rules: ATMRules):
        self.cfg, self.rules = cfg, rules
        self.llm = LocalLLM(cfg.llm_model, cfg.llm_url, timeout=90)
        self.system = agent_system(rules)

    def available(self) -> bool:
        self.llm._ok = None        # re-check (Ollama may have been started after the engine)
        return self.llm.available()

    def decide(self, state: Dict) -> Dict:
        ts = now_ist().isoformat()
        res, meta = self.llm.json_chat_timed(self.system, "Market state (JSON):\n" + json.dumps(state, default=str),
                                             SCHEMA, temperature=0.1, num_ctx=8192)
        if not res:
            d = {"ts": ts, "action": "STAND_ASIDE", "confidence": 0.0, "p_up_30m": 0.5,
                 "reasoning": "LLM unavailable or returned nothing usable", "key_factors": [], "risks": [],
                 "guardrail": "llm_error", "latency_s": meta.get("seconds")}
        else:
            d = {"ts": ts, "action": pick(res.get("action"), ACTIONS, "STAND_ASIDE"),
                 "confidence": clamp(res.get("confidence"), 0, 1), "p_up_30m": clamp(res.get("p_up_30m"), 0, 1, 0.5),
                 "reasoning": str(res.get("reasoning", ""))[:600],
                 "key_factors": [str(x)[:140] for x in (res.get("key_factors") or [])][:4],
                 "risks": [str(x)[:140] for x in (res.get("risks") or [])][:4], "guardrail": None,
                 "latency_s": meta.get("seconds"), "tokens_per_s": meta.get("tokens_per_s")}
        d = self.apply_guardrails(d, state)
        d["spot"] = (state.get("nifty") or {}).get("spot")
        self._log(d)
        return d

    def apply_guardrails(self, d: Dict, state: Dict) -> Dict:
        pos = state.get("position")
        a = d["action"]
        why = None
        if a in ("ENTER_LONG", "ENTER_SHORT"):
            if pos:
                why = "already in a position"
            elif not state.get("entry_window_open"):
                why = "outside the entry window"
            elif state.get("trades_today", 0) >= self.rules.max_trades_per_day:
                why = "daily trade limit reached"
            elif d["confidence"] < 0.6:
                why = "confidence below 0.60"
            elif (a == "ENTER_LONG" and d["p_up_30m"] < 0.5) or (a == "ENTER_SHORT" and d["p_up_30m"] > 0.5):
                why = "p_up_30m contradicts the action"
        elif a in ("EXIT", "HOLD") and not pos:
            d["action"] = "STAND_ASIDE"
        if why:
            d["guardrail"] = why
            d["action"] = "HOLD" if pos else "STAND_ASIDE"
        return d

    def _log(self, d: Dict) -> None:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with (LOG_DIR / f"decisions-{now_ist().date().isoformat()}.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(d, default=str) + "\n")


def load_decisions(days: int = 60) -> pd.DataFrame:
    rows: List[Dict] = []
    if LOG_DIR.exists():
        for f in sorted(LOG_DIR.glob("decisions-*.jsonl"))[-days:]:
            for line in f.read_text(encoding="utf-8").splitlines():
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue
    df = pd.DataFrame(rows)
    if len(df):
        df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert(IST)
    return df


def score_decisions(bars_1m: pd.DataFrame, days: int = 60) -> Dict:
    """Agent p_up_30m vs NIFTY 30 minutes later (and vs the quant model's, when logged)."""
    D = load_decisions(days)
    if D.empty or bars_1m.empty or "p_up_30m" not in D:
        return {"n": 0}
    close = bars_1m["close"]
    idx = close.index
    ys = []
    for t in D["ts"]:
        i = idx.searchsorted(t, side="right") - 1
        j = idx.searchsorted(t + pd.Timedelta(minutes=30), side="right") - 1
        ok = i >= 0 and j > i and idx[j].date() == t.date() and (idx[j] - t) >= pd.Timedelta(minutes=25)
        ys.append(float(close.iloc[j] > close.iloc[i]) if ok else float("nan"))
    D["y"] = ys
    D = D.dropna(subset=["y"])
    if D.empty:
        return {"n": 0}
    D = D[(D["p_up_30m"] - 0.5).abs() > 1e-9] if (D["p_up_30m"] != 0.5).any() else D
    hit = ((D["p_up_30m"] >= 0.5).astype(float) == D["y"])
    return {"n": int(len(D)), "days": int(D["ts"].dt.date.nunique()), "accuracy": float(hit.mean()),
            "actions": D["action"].value_counts().to_dict()}
