"""Pre-open market brief by the local LLM: it reads the numbers (model probability, overnight
global moves, Asia, ADRs, volatility regime, option chain, events) and the top headlines, and
writes a calibrated view in plain English. The brief is logged as the `llm_analyst` book's
signal and scored every evening, so its accuracy is measured instead of assumed."""

from __future__ import annotations

import datetime as dt
import json
import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

from ..calendar import IST, now_ist
from ..config import DATA_DIR, EngineConfig
from .llm import LocalLLM, clamp, pick

log = logging.getLogger(__name__)
BRIEFS = DATA_DIR / "briefs"

BRIEF_SCHEMA = {
    "type": "object",
    "properties": {
        "stance": {"type": "string", "enum": ["bullish", "bearish", "neutral"]},
        "p_up": {"type": "number"},
        "confidence": {"type": "number"},
        "expected_range_pct": {"type": "number"},
        "drivers": {"type": "array", "items": {"type": "string"}},
        "risks": {"type": "array", "items": {"type": "string"}},
        "narrative": {"type": "string"},
    },
    "required": ["stance", "p_up", "confidence", "expected_range_pct", "drivers", "risks", "narrative"],
}

SYSTEM = """You are an experienced NIFTY 50 index-options trader writing the 09:10 IST pre-open brief for today's session (09:15-15:30 IST). You are given numbers and headlines. Decide whether NIFTY is more likely to close above or below its opening price today.

Rules:
- Weigh the evidence like a professional: overnight US futures and Asian opens, the ADR premium, the volatility regime, the option chain, scheduled events, and news. Say which items matter most.
- The quantitative model's historical out-of-sample accuracy is given; treat it as one input, not the truth.
- Be calibrated. Daily index direction is hard to predict: keep p_up between 0.40 and 0.60 unless the evidence is unusually strong. Use "neutral" when it is mixed.
- expected_range_pct: your estimate of today's high-low range in percent of the open.
- drivers and risks: at most 4 short items each. narrative: at most 120 words."""


def _fmt(v, nd=3):
    return None if v is None or (isinstance(v, float) and v != v) else (round(float(v), nd) if isinstance(v, (int, float, np.floating)) else v)


def build_state(cfg: EngineConfig, sig: Optional[Dict] = None, day: Optional[dt.date] = None) -> Dict:
    """Snapshot of everything known at ~09:10 IST, all numbers rounded for the prompt."""
    from ..data import store
    from ..data.events import event_features
    from ..data.global_markets import overnight_features
    from ..model import load_champion
    day = day or now_ist().date()
    raw = store.load_all()
    nf = raw["nifty"]
    sessions = nf.index[nf.index < pd.Timestamp(day)].append(pd.DatetimeIndex([pd.Timestamp(day)]))
    st: Dict = {"date": day.isoformat()}
    _, meta = load_champion()
    if meta:
        st["model"] = {"accuracy_oos": _fmt(meta["metrics"].get("accuracy")),
                       "confident_hit_rate": _fmt(next((r["hit_rate"] for r in meta["metrics"].get("by_threshold", [])
                                                        if abs(r["threshold"] - cfg.prob_threshold) < 1e-9), None)),
                       "has_proven_edge": meta.get("has_edge")}
    if sig:
        st["model_today"] = {"p_up": _fmt(sig.get("p_up")), "stance": sig.get("stance"), "open": _fmt(sig.get("open"), 1)}
    if len(nf):
        last = nf.iloc[-1]
        st["nifty_prev"] = {"close": _fmt(last["close"], 1), "day_change_pct": _fmt((last["close"] / nf["close"].iloc[-2] - 1) * 100, 2)
                            if len(nf) > 1 else None}
    vix = raw.get("indiavix")
    if vix is not None and len(vix):
        st["india_vix"] = _fmt(vix["close"].iloc[-1], 2)
    try:
        on = overnight_features(sessions).iloc[-1]
        st["overnight_moves_pct"] = {k.replace("on_", ""): _fmt(v * 100, 2) for k, v in on.items()}
    except Exception:  # noqa: BLE001
        pass
    asia = {}
    for name in ("n225", "hsi", "kospi", "twii", "shanghai"):
        df = raw.get(name)
        if df is not None and len(df) and df.index[-1] == pd.Timestamp(day) and len(df) > 1:
            asia[name] = _fmt((df["open"].iloc[-1] / df["close"].iloc[-2] - 1) * 100, 2)
    if asia:
        st["asia_opening_gap_pct"] = asia
    for name in ("spx", "ndx", "brent", "dxy", "us10y"):
        df = raw.get(name)
        if df is not None and len(df) > 1:
            st.setdefault("us_and_macro_last_session_pct", {})[name] = _fmt((df["close"].iloc[-1] / df["close"].iloc[-2] - 1) * 100, 2)
    try:
        ev = event_features(sessions).iloc[-1]
        st["events"] = [k[3:] for k, v in ev.items() if v and k.startswith("ev_") and k not in ("ev_any", "ev_any_next")]
    except Exception:  # noqa: BLE001
        pass
    try:
        from ..data.recorder import latest_chain_metrics
        m = latest_chain_metrics(day)
        if m:
            st["option_chain"] = {k.replace("near_", ""): _fmt(v) for k, v in m.items()
                                  if k.startswith("near_") and any(s in k for s in ("pcr", "wall", "max_pain", "atm_iv", "rr25", "implied_move", "gex"))}
    except Exception:  # noqa: BLE001
        pass
    try:
        from .news_llm import preopen_window, top_headlines, scored_frame, window_features
        prev = sessions[-2].date() if len(sessions) > 1 else day - dt.timedelta(days=1)
        s, e = preopen_window(day, prev)
        st["news_summary"] = {k: _fmt(v) for k, v in window_features(scored_frame(cfg, prev, day), s, e).items()}
        st["top_headlines"] = top_headlines(cfg, s, e)
    except Exception as ex:  # noqa: BLE001
        log.warning("news state failed: %s", ex)
    return st


def write_brief(cfg: EngineConfig, state: Dict) -> Dict:
    llm = LocalLLM(cfg.llm_model, cfg.llm_url)
    out = {"date": state.get("date"), "ts": now_ist().isoformat(), "model": llm.model, "state": state}
    if not llm.available():
        out["error"] = "local LLM not available (start Ollama and `ollama pull %s`)" % llm.model
        return out
    res = llm.json_chat(SYSTEM, "Data (JSON):\n" + json.dumps(state, default=str), BRIEF_SCHEMA, temperature=0.1)
    if not res:
        out["error"] = "LLM returned nothing usable"
        return out
    out["brief"] = {"stance": pick(res.get("stance"), ["bullish", "bearish", "neutral"], "neutral"),
                    "p_up": clamp(res.get("p_up"), 0.0, 1.0, 0.5), "confidence": clamp(res.get("confidence"), 0, 1),
                    "expected_range_pct": clamp(res.get("expected_range_pct"), 0, 10, float("nan")),
                    "drivers": [str(x)[:160] for x in (res.get("drivers") or [])][:4],
                    "risks": [str(x)[:160] for x in (res.get("risks") or [])][:4],
                    "narrative": str(res.get("narrative", ""))[:1200]}
    BRIEFS.mkdir(parents=True, exist_ok=True)
    (BRIEFS / f"{state.get('date')}.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    return out


def load_briefs() -> pd.DataFrame:
    rows = []
    for f in sorted(BRIEFS.glob("*.json")) if BRIEFS.exists() else []:
        js = json.loads(f.read_text(encoding="utf-8"))
        b = js.get("brief")
        if b:
            rows.append({"date": pd.Timestamp(js["date"]), "p_up": b["p_up"], "stance": b["stance"],
                         "confidence": b["confidence"], "expected_range_pct": b.get("expected_range_pct")})
    return pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame()
