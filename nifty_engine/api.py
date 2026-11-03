"""FastAPI service: signal, paper-trading status and reports. `python -m nifty_engine serve`."""

from __future__ import annotations

import json

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, PlainTextResponse

from .config import EngineConfig
from .model import load_champion
from .paper.ledger import Ledger
from .pipeline import REPORTS
from .report import live_month_report

app = FastAPI(title="NIFTY v4 engine", version="4.0.0")
cfg = EngineConfig.load()
ledger = Ledger()


@app.get("/api/v4/health")
def health():
    _, meta = load_champion()
    return {"ok": True, "model": meta and {k: meta[k] for k in ("version", "spec_name", "has_edge", "trained_on")},
            "capital": cfg.capital, "books": cfg.books}


@app.get("/api/v4/signal")
def signal():
    return ledger.last_signal() or {"stance": "no signal yet today"}


@app.get("/api/v4/paper/status")
def status():
    tr = ledger.trades("live")
    closed = tr[tr["status"] == "CLOSED"] if len(tr) else tr
    return {"capital": cfg.capital, "equity": cfg.capital + (float(closed["net_pnl"].sum()) if len(closed) else 0.0),
            "trades_closed": int(len(closed)), "open": json.loads(ledger.open_trades().to_json(orient="records")),
            "win_rate": float((closed["net_pnl"] > 0).mean()) if len(closed) else None}


@app.get("/api/v4/paper/scorecard")
def scorecard():
    from .paper.live import LivePaperTrader
    return LivePaperTrader(cfg, ledger).prediction_scorecard()


@app.get("/api/v4/paper/trades")
def trades():
    return json.loads(ledger.trades().to_json(orient="records"))


@app.get("/api/v4/model")
def model():
    _, meta = load_champion()
    return meta or {}


@app.get("/api/v4/report/live", response_class=PlainTextResponse)
def live_report():
    return live_month_report(ledger.trades("live"), ledger.equity_curve(), cfg.capital)


@app.get("/api/v4/report/backtest", response_class=PlainTextResponse)
def bt_report():
    p = REPORTS / "BACKTEST_REPORT.md"
    return p.read_text(encoding="utf-8") if p.exists() else "Run `python -m nifty_engine backtest` first."


@app.get("/paper", response_class=HTMLResponse)
def dashboard():
    return """<!doctype html><html><head><meta charset=utf-8><title>NIFTY v4 paper trader</title>
<style>body{font:14px system-ui;margin:24px;background:#0f1115;color:#e6e6e6}table{border-collapse:collapse}
td,th{border:1px solid #333;padding:4px 8px}h2{margin-top:24px}.k{color:#9aa}</style></head><body>
<h1>NIFTY v4 &mdash; auto paper trader</h1><div id=s></div><h2>Prediction scorecard (every day, traded or not)</h2><pre id=c></pre><h2>Today's signal</h2><pre id=g></pre>
<h2>Trades</h2><table id=t></table><script>
async function load(){const s=await (await fetch('/api/v4/paper/status')).json();
document.getElementById('s').innerHTML=`<b>Equity</b> Rs ${s.equity.toFixed(0)} <span class=k>(capital ${s.capital}, closed ${s.trades_closed}, win rate ${s.win_rate==null?'-':(s.win_rate*100).toFixed(1)+'%'})</span>`;
document.getElementById('c').textContent=JSON.stringify(await (await fetch('/api/v4/paper/scorecard')).json(),null,2);
document.getElementById('g').textContent=JSON.stringify(await (await fetch('/api/v4/signal')).json(),null,2);
const t=await (await fetch('/api/v4/paper/trades')).json();
document.getElementById('t').innerHTML='<tr><th>Date</th><th>Dir</th><th>Style</th><th>Lots</th><th>Entry</th><th>Exit</th><th>Net</th><th>Reason</th><th>Status</th><th>Prices</th></tr>'+
t.reverse().map(r=>`<tr><td>${r.date}</td><td>${r.direction>0?'LONG':'SHORT'}</td><td>${r.style}</td><td>${r.lots}</td><td>${r.entry_value}</td><td>${r.exit_value??''}</td><td>${r.net_pnl??''}</td><td>${r.exit_reason??''}</td><td>${r.status}</td><td>${r.price_source}</td></tr>`).join('');}
load();setInterval(load,30000);</script></body></html>"""
