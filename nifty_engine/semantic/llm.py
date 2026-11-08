"""Minimal client for a local Ollama server (https://ollama.com), free and on-device.

Setup on a Mac:  brew install ollama && ollama serve   (or the Ollama app)
                 ollama pull llama3.1:8b               (~5 GB; any instruct model works)
Structured output: Ollama's `format` field takes a JSON schema and constrains the reply to it.
Replies are still validated and clamped here, because small models can drift."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional

import requests

log = logging.getLogger(__name__)


class LocalLLM:
    def __init__(self, model: Optional[str] = None, url: Optional[str] = None, timeout: float = 180.0):
        self.model = model or os.environ.get("NE_LLM_MODEL", "llama3.1:8b")
        self.url = (url or os.environ.get("NE_LLM_URL", "http://localhost:11434")).rstrip("/")
        self.timeout = timeout
        self._ok: Optional[bool] = None

    def available(self) -> bool:
        if self._ok is None:
            try:
                r = requests.get(self.url + "/api/tags", timeout=3)
                names = [m.get("name", "") for m in r.json().get("models", [])]
                base = self.model.split(":")[0]
                self._ok = any(n == self.model or n.split(":")[0] == base for n in names)
                if not self._ok:
                    log.warning("Ollama is running but model %s is not pulled (ollama pull %s)", self.model, self.model)
            except Exception:  # noqa: BLE001
                self._ok = False
        return self._ok

    def json_chat(self, system: str, user: str, schema: Dict[str, Any], temperature: float = 0.0,
                  num_ctx: int = 8192) -> Optional[Dict[str, Any]]:
        """One chat turn constrained to `schema`. Returns the parsed object, or None on failure."""
        obj, _ = self.json_chat_timed(system, user, schema, temperature, num_ctx)
        return obj

    def json_chat_timed(self, system: str, user: str, schema: Dict[str, Any], temperature: float = 0.0,
                        num_ctx: int = 8192):
        """Like json_chat, plus timing: {seconds, eval_tokens, tokens_per_s, prompt_tokens}.
        Thinking is off by default (decisions must be fast); set NE_LLM_THINK=1 to allow it.
        keep_alive keeps the model in memory between the engine's calls."""
        import time as _t
        body = {"model": self.model, "stream": False, "format": schema, "keep_alive": "30m",
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "options": {"temperature": temperature, "num_ctx": num_ctx}}
        if os.environ.get("NE_LLM_THINK", "").lower() not in ("1", "true", "yes"):
            body["think"] = False
        t0 = _t.time()
        meta: Dict[str, Any] = {}
        try:
            r = requests.post(self.url + "/api/chat", json=body, timeout=self.timeout)
            if r.status_code == 400 and "think" in body:      # model without a thinking switch
                body.pop("think")
                r = requests.post(self.url + "/api/chat", json=body, timeout=self.timeout)
            r.raise_for_status()
            js = r.json()
            ev, dur = js.get("eval_count") or 0, (js.get("eval_duration") or 0) / 1e9
            meta = {"seconds": round(_t.time() - t0, 2), "eval_tokens": ev, "prompt_tokens": js.get("prompt_eval_count"),
                    "tokens_per_s": round(ev / dur, 1) if dur else None}
            content = js.get("message", {}).get("content", "")
            return json.loads(content), meta
        except Exception as e:  # noqa: BLE001
            log.warning("local LLM call failed: %s", str(e)[:200])
            meta["seconds"] = round(_t.time() - t0, 2)
            meta["error"] = str(e)[:200]
            return None, meta


def clamp(v, lo: float, hi: float, default: float = 0.0) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return default
    return default if x != x else max(lo, min(hi, x))


def pick(v, allowed, default):
    return v if v in allowed else default
