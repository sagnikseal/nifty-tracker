# Engine API (dashboard contract)

The engine runs in a terminal (`python -m nifty_engine app`, or the one-click launchers). It serves the
dashboard at `http://localhost:8010/` and the JSON API below. Every number shown in the UI comes from
this API. The UI never invents data: when a value is missing it shows "—" and the reason.

All timestamps are ISO-8601 in IST (`+05:30`) unless the field name ends in `_ms` (epoch milliseconds).

## Live state

`GET /api/state`: one snapshot of everything.
`GET /api/stream`: Server-Sent Events. Sends `event: state` with the same JSON every ~2 s (and
immediately after a trade or a new headline). The UI should use SSE and fall back to polling
`/api/state` every 5 s if the stream drops.

```jsonc
{
  "ts": "2026-11-09T10:42:05+05:30",
  "market": {"status": "OPEN",            // OPEN | PRE_OPEN | CLOSED | HOLIDAY
             "session_date": "2026-11-09", "note": "closes 15:30"},
  "quote": {"spot": 23810.85, "prev_close": 23731.85, "open": 23750.1, "high": 23842.0, "low": 23720.4,
            "change": 79.0, "change_pct": 0.33, "vix": 15.66, "vix_change_pct": -1.2,
            "source": "upstox",          // upstox | yahoo | none
            "age_s": 3.1},               // seconds since the quote was fetched
  "bars": [{"t_ms": 1762661700000, "o": 0, "h": 0, "l": 0, "c": 0}],   // today's 1-minute NIFTY bars (<= 375)
  "prediction": {
    "p_up_30m": 0.57,                    // P(NIFTY higher 30 min from now), calibrated, null if unavailable
    "p_up_close": 0.55,                  // P(NIFTY higher at 15:15 than now)
    "p_daily": 0.53,                     // 09:20 daily model P(close > open)
    "model_version": "intraday-20261108-2210",
    "updated": "2026-11-09T10:40:01+05:30",
    "has_edge": true,                    // walk-forward skill significant (p < 0.05)
    "top_drivers": [{"feature": "r_open_T", "label": "Move since open", "value": 0.0021,
                     "contribution": 0.031}],    // signed contribution to log-odds (+ = bullish)
    "history": [{"t_ms": 0, "p30": 0.5, "pclose": 0.5}]   // today's predictions, one per 5-min bar
  },
  "model_quality": {                     // out-of-sample walk-forward results (from the last training run)
    "horizon_30m":  {"n": 80000, "period": ["2022-06-01", "2026-11-03"], "accuracy": 0.53, "auc": 0.54,
                     "logloss_skill": 0.002, "p_value": 0.001, "base_rate": 0.51,
                     "hit_rate_at": [{"threshold": 0.6, "coverage": 0.08, "hit_rate": 0.58}],
                     "share_above_80": 0.001},
    "horizon_close": {"...": "same fields"},
    "daily":         {"...": "same fields"}
  },
  "position": null,                      // or the object below (the book shown is the first open one)
  "positions": [{
    "book": "model_rules",               // model_rules | agent | manual
    "id": 12, "side": "CE", "direction": 1, "strike": 23800, "expiry": "2026-11-11", "lots": 1, "qty": 65,
    "entry_ts": "...", "entry_price": 142.5, "ltp": 171.0, "pnl_inr": 1852.5,
    "target_inr": 5000, "stop_inr": 2000, "target_hit": false,
    "extend_until": null,                // ISO time while in a 30-minute extension
    "against_count": 1, "against_limit": 2,
    "rule_state": "WAITING_TARGET",      // WAITING_TARGET | EXTENDED | EXITING
    "rule_note": "1 of 2 consecutive 5-min closes against the trade",
    "price_source": "broker_bid_ask"     // broker_bid_ask | broker_ltp | model_vix
  }],
  "books": [{"name": "model_rules", "label": "Model + your rules", "auto": true, "equity": 100000,
             "realized_pnl": 0, "today_pnl": 0, "trades": 0, "win_rate": null, "open": false}],
  "agent": {"model": "qwen3.5:9b", "available": true,
            "last_decision": {"ts": "...", "action": "HOLD",   // ENTER_LONG | ENTER_SHORT | EXIT | HOLD | STAND_ASIDE
                              "confidence": 0.6, "p_up_30m": 0.58, "reasoning": "...",
                              "key_factors": ["..."], "risks": ["..."], "guardrail": null},
            "brief": {"stance": "bullish", "p_up": 0.55, "narrative": "...", "drivers": [], "risks": []},
            "decisions_today": 14},
  "news": [{"id": "ab12", "title": "...", "source": "rbi_press", "category": "regulator", "url": "...",
            "first_seen": "...", "age_min": 4,
            "llm": {"direction": 0.6, "magnitude": 0.5, "surprise": 0.7, "relevance": 0.9, "confidence": 0.7,
                    "channel": "rates", "horizon": "days", "reason": "...", "scorer": "qwen3.5:9b"},
            "impact": {"exp_bps_30m": 6.5, "p_up_30m": 0.56, "basis": "learned from 412 past headlines"}}],
  "chain": {"updated": "...", "expiry": "2026-11-11", "pcr": 1.12, "max_pain": 23800, "call_wall": 24000,
            "put_wall": 23500, "atm_iv": 13.2, "implied_move_pct": 0.62, "gex_cr": 1250.0, "zero_gamma": 23650,
            "rr25": -2.1},                       // or null
  "sources": [{"name": "rbi_press", "category": "regulator", "ok": true, "last_ok": "...",
               "latency_ms": 420, "items_new_today": 3, "error": null, "poll_s": 60}],
  "rules": {"target_inr": 5000, "stop_inr": 2000, "extend_prob": 0.8, "extend_minutes": 30,
            "against_bars": 2, "against_scope": "in_profit", "entry_threshold": 0.6,
            "entry_window": ["09:20", "13:30"], "exit_time": "15:15", "lots": 1},
  "alerts": [{"ts": "...", "level": "warn", "text": "Upstox token missing: option prices are modelled"}],
  "health": {"engine": "running", "loop_lag_s": 0.4, "llm": true, "upstox": true, "news_thread": true,
             "last_bar": "...", "uptime_s": 3600}
}
```

## Other endpoints

| Method | Path | Returns |
|---|---|---|
| GET | `/api/trades?limit=200` | list of trades (newest first): `id, book, date, entry_ts, exit_ts, direction, style, legs, entry_value, exit_value, net_pnl, charges, exit_reason, status, price_source` |
| GET | `/api/equity` | `{book: [{"date": "...", "equity": 0}]}` |
| GET | `/api/news?limit=100` | same items as `state.news`, more of them |
| GET | `/api/report/{name}` | Markdown text. `name` = `intraday` \| `atm_replay` \| `news_impact` \| `research` \| `data` \| `llm_bench` |
| GET | `/api/glossary` | `{term: {"short": "...", "long": "..."}}` (served from `nifty_engine/web/glossary.json`) |
| POST | `/api/paper/order` | body `{"action": "buy_ce" \| "buy_pe" \| "exit", "book": "manual"}` → `{"ok": true, "trade_id": 1}` or `{"ok": false, "error": "..."}` |
| POST | `/api/auto` | body `{"book": "model_rules" \| "agent", "enabled": true}` → `{"ok": true}` |

Paper only: no endpoint can place a real order.
