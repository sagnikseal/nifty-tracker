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
        body = {"model": self.model, "stream": False, "format": schema,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "options": {"temperature": temperature, "num_ctx": num_ctx}}
        if os.environ.get("NE_LLM_THINK", "").lower() in ("0", "false", "no"):
            body["think"] = False      # for thinking models (e.g. qwen3) when speed matters
        try:
            r = requests.post(self.url + "/api/chat", json=body, timeout=self.timeout)
            r.raise_for_status()
            content = r.json().get("message", {}).get("content", "")
            return json.loads(content)
        except Exception as e:  # noqa: BLE001
            log.warning("local LLM call failed: %s", str(e)[:200])
            return None


def clamp(v, lo: float, hi: float, default: float = 0.0) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return default
    return default if x != x else max(lo, min(hi, x))


def pick(v, allowed, default):
    return v if v in allowed else default
