"""`python -m nifty_engine llm-bench [--models a,b,c]`: choose the local model by measurement.

For each Ollama model it measures, on the labelled set in config/llm_bench.json:
  * JSON reliability (share of headlines returned in valid, schema-conforming form),
  * direction accuracy on clear-cut headlines and neutrality on noise headlines,
  * channel accuracy, consistency (same batch scored twice), speed (seconds per 8 headlines,
    generation tokens/s) and memory (from Ollama's /api/ps),
  * trading-agent sanity: the expected action in three scenarios.
Writes store/reports/LLM_BENCH_REPORT.md. Nothing here touches trading.
"""

from __future__ import annotations

import json
import logging
from typing import Dict, List

import requests

from ..config import DATA_DIR, ROOT, EngineConfig
from ..semantic.llm import LocalLLM
from ..strategies.atm_rules import ATMRules
from .prompts import agent_system, headline_system

log = logging.getLogger(__name__)
REPORT = DATA_DIR / "reports" / "LLM_BENCH_REPORT.md"
DEFAULT_MODELS = ["qwen3.5:9b", "llama3.1:8b", "gemma4:12b", "qwen3:8b"]


def _installed(url: str) -> List[str]:
    try:
        return [m["name"] for m in requests.get(url + "/api/tags", timeout=3).json().get("models", [])]
    except Exception:  # noqa: BLE001
        return []


def _mem_gb(url: str, model: str):
    try:
        for m in requests.get(url + "/api/ps", timeout=3).json().get("models", []):
            if m.get("name") == model or m.get("model") == model:
                return round((m.get("size") or 0) / 1e9, 1)
    except Exception:  # noqa: BLE001
        pass
    return None


def bench_model(model: str, cfg: EngineConfig, data: Dict, batch: int = 8) -> Dict:
    from ..data.instruments import HEAVYWEIGHTS
    from ..semantic.news_llm import BATCH_SCHEMA
    llm = LocalLLM(model, cfg.llm_url, timeout=240)
    system = headline_system(", ".join(HEAVYWEIGHTS))
    H = data["headlines"]
    got: Dict[str, Dict] = {}
    secs, tps = [], []
    for rep in range(2):
        for i in range(0, len(H), batch):
            chunk = H[i:i + batch]
            res, meta = llm.json_chat_timed(system, "\n".join(f"id={h['id']}: {h['text']}" for h in chunk), BATCH_SCHEMA)
            if rep == 0:
                secs.append(meta.get("seconds") or 0)
                if meta.get("tokens_per_s"):
                    tps.append(meta["tokens_per_s"])
            for it in (res or {}).get("items", []):
                if it.get("id") in {h["id"] for h in chunk}:
                    got.setdefault(it["id"], {})[rep] = it
    n = len(H)
    valid = [h for h in H if 0 in got.get(h["id"], {})]
    clear = [h for h in valid if h["sign"] != 0]
    noise = [h for h in valid if h["sign"] == 0]

    def dirv(h, rep=0):
        try:
            return float(got[h["id"]][rep].get("nifty_direction") or 0)
        except (TypeError, ValueError, KeyError):
            return 0.0

    dir_acc = sum(1 for h in clear if dirv(h) * h["sign"] > 0.1) / len(clear) if clear else None
    neutral_ok = sum(1 for h in noise if abs(dirv(h)) < 0.3) / len(noise) if noise else None
    ch_acc = sum(1 for h in valid if got[h["id"]][0].get("channel") == h["channel"]) / len(valid) if valid else None
    both = [h for h in valid if 1 in got[h["id"]]]
    consist = sum(1 for h in both if abs(dirv(h, 0) - dirv(h, 1)) <= 0.2) / len(both) if both else None
    # agent scenarios
    ag = LocalLLM(model, cfg.llm_url, timeout=240)
    from .llm_agent import SCHEMA
    asys = agent_system(ATMRules.from_cfg(cfg))
    agent_rows = []
    for sc in data.get("agent_scenarios", []):
        res, meta = ag.json_chat_timed(asys, "Market state (JSON):\n" + json.dumps(sc["state"]), SCHEMA, temperature=0.1)
        act = (res or {}).get("action")
        agent_rows.append({"id": sc["id"], "expect": sc["expect"], "got": act, "ok": act in sc["expect"],
                           "seconds": meta.get("seconds"), "reason": ((res or {}).get("reasoning") or "")[:160]})
    return {"model": model, "json_valid": len(valid) / n, "direction_accuracy": dir_acc, "neutral_handling": neutral_ok,
            "channel_accuracy": ch_acc, "consistency": consist,
            "sec_per_batch": round(sum(secs) / len(secs), 1) if secs else None,
            "tokens_per_s": round(sum(tps) / len(tps), 1) if tps else None, "memory_gb": _mem_gb(cfg.llm_url, model),
            "agent_ok": sum(r["ok"] for r in agent_rows), "agent_n": len(agent_rows), "agent": agent_rows}


def run(cfg: EngineConfig, models: List[str] = None) -> Dict:
    data = json.loads((ROOT / "config" / "llm_bench.json").read_text(encoding="utf-8"))
    inst = _installed(cfg.llm_url)
    models = models or DEFAULT_MODELS
    rows, missing = [], []
    for m in models:
        if not any(i == m or i.split(":")[0] == m.split(":")[0] and (":" not in m or i == m) for i in inst):
            missing.append(m)
            continue
        log.info("benchmarking %s ...", m)
        rows.append(bench_model(m, cfg, data))
    out = {"installed": inst, "results": rows, "missing": missing, "ollama_running": bool(inst) or _ping(cfg.llm_url)}
    write_report(out)
    return out


def _ping(url: str) -> bool:
    try:
        requests.get(url + "/api/tags", timeout=2)
        return True
    except Exception:  # noqa: BLE001
        return False


def write_report(out: Dict) -> None:
    f = lambda v: "—" if v is None else (f"{v:.0%}" if isinstance(v, float) and v <= 1 else str(v))  # noqa: E731
    L = ["# Local LLM benchmark", "",
         "Measured on this machine with the labelled set in `config/llm_bench.json` (clear-cut headlines, noise "
         "headlines that should score neutral, and three trading-agent scenarios).", ""]
    if not out["ollama_running"]:
        L += ["**Ollama is not running.** Install it from https://ollama.com, start it, pull the candidates "
              "(`ollama pull qwen3.5:9b`, `ollama pull llama3.1:8b`, `ollama pull gemma4:12b`) and run "
              "`python -m nifty_engine llm-bench` again."]
    if out["results"]:
        L += ["| Model | JSON valid | Direction (clear) | Neutral on noise | Channel | Consistency | s / 8 headlines | tokens/s | Memory | Agent scenarios |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for r in out["results"]:
            L.append(f"| {r['model']} | {f(r['json_valid'])} | {f(r['direction_accuracy'])} | {f(r['neutral_handling'])} | "
                     f"{f(r['channel_accuracy'])} | {f(r['consistency'])} | {r['sec_per_batch']} | {r['tokens_per_s']} | "
                     f"{r['memory_gb'] or '—'} GB | {r['agent_ok']}/{r['agent_n']} |")
        L += ["", "Agent scenario details:", ""]
        for r in out["results"]:
            for a in r["agent"]:
                L.append(f"* {r['model']} {a['id']}: expected {a['expect']}, got {a['got']} ({a['seconds']} s) — {a['reason']}")
        best = max(out["results"], key=lambda r: ((r["json_valid"] or 0) >= 0.95, (r["direction_accuracy"] or 0)
                                                    + (r["neutral_handling"] or 0) + 0.5 * (r["agent_ok"] / max(r["agent_n"], 1))))
        L += ["", f"Suggested: **{best['model']}** (needs ≥ 95% valid JSON; then direction + neutrality + agent). Set it in "
              f"`config/engine.json` as `llm_model`, and set `llm_knowledge_cutoff` to that model's training cutoff "
              "(or its release date if unknown)."]
    if out["missing"]:
        L += ["", "Not installed (skipped): " + ", ".join(out["missing"]) + ". Pull with `ollama pull <name>`."]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(L), encoding="utf-8")
