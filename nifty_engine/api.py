"""FastAPI service for the dashboard (contract: docs/API.md). Paper trading only.

`create_app(runtime)` serves the dashboard at `/`, static files at `/static`, the JSON API at
`/api/*`, and a Server-Sent Events stream of the live snapshot at `/api/stream`.
Without a runtime (e.g. `serve` for reports only) the state endpoints say the engine is not running.
"""

from __future__ import annotations

import asyncio
import json
import math
from typing import Any, Optional

from fastapi import Body, FastAPI
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .config import DATA_DIR, ROOT
from .paper.ledger import Ledger

WEB = ROOT / "nifty_engine" / "web"
REPORTS = DATA_DIR / "reports"
REPORT_FILES = {"intraday": "INTRADAY_REPORT.md", "atm_replay": "ATM_REPLAY_REPORT.md",
                "news_impact": "NEWS_IMPACT_REPORT.md", "research": "RESEARCH_REPORT.md", "data": "DATA_REPORT.md",
                "llm_bench": "LLM_BENCH_REPORT.md", "sources": "SOURCES_REPORT.md", "backtest": "BACKTEST_REPORT.md"}
HINTS = {"intraday": "python -m nifty_engine intraday-train", "atm_replay": "python -m nifty_engine atm-replay",
         "news_impact": "python -m nifty_engine news-impact", "research": "python -m nifty_engine research",
         "data": "python -m nifty_engine data-report", "llm_bench": "python -m nifty_engine llm-bench",
         "sources": "python -m nifty_engine sources-check", "backtest": "python -m nifty_engine backtest"}


def clean(o: Any) -> Any:
    """JSON-safe: NaN/inf -> None, timestamps -> ISO strings, numpy scalars -> Python."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, float):
        return None if math.isnan(o) or math.isinf(o) else o
    if hasattr(o, "isoformat"):
        return o.isoformat()
    if hasattr(o, "item"):
        try:
            return clean(o.item())
        except Exception:  # noqa: BLE001
            return str(o)
    return o


def _json(o: Any, status: int = 200) -> JSONResponse:
    return JSONResponse(clean(o), status_code=status)


def create_app(runtime=None) -> FastAPI:
    app = FastAPI(title="NIFTY engine")
    ledger = runtime.ledger if runtime is not None else Ledger()
    if WEB.exists():
        app.mount("/static", StaticFiles(directory=str(WEB)), name="static")

    @app.get("/")
    def index():
        f = WEB / "index.html"
        return FileResponse(str(f)) if f.exists() else PlainTextResponse("dashboard files missing (nifty_engine/web)")

    @app.get("/api/state")
    def state():
        if runtime is None:
            return _json({"error": "engine not running: start it with `python -m nifty_engine app`"}, 503)
        return _json(runtime.snapshot())

    @app.get("/api/stream")
    async def stream():
        if runtime is None:
            return _json({"error": "engine not running"}, 503)

        async def gen():
            last = 0.0
            loop = asyncio.get_running_loop()
            while True:
                changed = await loop.run_in_executor(None, runtime.changed.wait, 2.0)
                runtime.changed.clear()
                now = loop.time()
                if not changed and now - last < 2.0:
                    continue
                last = now
                yield f"event: state\ndata: {json.dumps(clean(runtime.snapshot()), allow_nan=False)}\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/api/trades")
    def trades(limit: int = 200):
        t = ledger.trades()
        if len(t):
            t = t.sort_values("id", ascending=False).head(limit)
            for c in ("legs", "rules"):
                if c in t:
                    t[c] = [json.loads(x) if isinstance(x, str) and x else None for x in t[c]]
        return _json(json.loads(t.to_json(orient="records")) if len(t) else [])

    @app.get("/api/equity")
    def equity():
        t = ledger.trades("live")
        out = {}
        cap = runtime.cfg.book_capital if runtime is not None else 100_000.0
        if len(t):
            c = t[t["status"] == "CLOSED"].copy()
            c["d"] = c["exit_ts"].fillna(c["date"]).astype(str).str[:10]
            for b, g in c.groupby("book"):
                s = g.groupby("d")["net_pnl"].sum().sort_index().cumsum() + cap
                out[b] = [{"date": d, "equity": round(float(v), 2)} for d, v in s.items()]
        return _json(out)

    @app.get("/api/news")
    def news(limit: int = 100):
        if runtime is None:
            return _json([])
        if not runtime.news:
            runtime.refresh_news_view()
        return _json(runtime.news[:limit])

    @app.get("/api/report/{name}")
    def report(name: str):
        f = REPORT_FILES.get(name)
        if not f:
            return PlainTextResponse(f"unknown report {name}", status_code=404)
        p = REPORTS / f
        if not p.exists():
            return PlainTextResponse(f"# Not generated yet\n\nRun `{HINTS[name]}` in the engine's terminal.")
        return PlainTextResponse(p.read_text(encoding="utf-8"))

    @app.get("/api/glossary")
    def glossary():
        f = WEB / "glossary.json"
        return Response(f.read_bytes(), media_type="application/json") if f.exists() else _json({})

    @app.post("/api/paper/order")
    def order(body: dict = Body(...)):
        if runtime is None:
            return _json({"ok": False, "error": "engine not running"}, 503)
        return _json(runtime.manual_order(str(body.get("action", "")), str(body.get("book", "manual"))))

    @app.post("/api/auto")
    def auto(body: dict = Body(...)):
        if runtime is None:
            return _json({"ok": False, "error": "engine not running"}, 503)
        return _json(runtime.set_auto(str(body.get("book", "")), bool(body.get("enabled"))))

    @app.get("/api/health")
    def health():
        return _json({"ok": True, "runtime": runtime is not None})

    return app


def serve(runtime=None, host: str = "127.0.0.1", port: int = 8010, open_browser: bool = False) -> None:
    import threading
    import webbrowser

    import uvicorn
    if open_browser:
        threading.Timer(1.5, lambda: webbrowser.open(f"http://localhost:{port}/")).start()
    uvicorn.run(create_app(runtime), host=host, port=port, log_level="warning")


app: Optional[FastAPI] = None   # created by the CLI (`python -m nifty_engine app`)
