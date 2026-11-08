"""Upstox collection for the GitHub Actions data collector (read-only Analytics Token).

  snapshot  (every 15 min, 09:15-15:45 IST): quotes with 5-level depth for NIFTY, India VIX,
            Bank Nifty, the near-month NIFTY future and the heavyweights; the full option chain of
            the nearest two expiries plus derived metrics (recorder format); one news poll.
  eod       (after 15:35 IST): today's 1- and 5-minute candles for the same instruments; 1-minute
            candles of the options of a weekly expiry that ended in the last few days (ATM +/- 800,
            Upstox Plus; skipped with a log line when the plan lacks it); one news poll.
  news      : one news poll only.

Every Upstox call is guarded: one bad instrument is logged in the run record and skipped. A 401
stops all Upstox calls for the run (token expired) without failing the workflow. The token is
never logged.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd
import requests

from ..calendar import IST, now_ist, year_fraction
from ..config import EngineConfig
from ..data.providers import UpstoxProvider
from ..data.recorder import CHAIN_COLS
from . import layout
from .layout import (CANDLE_COLS, CHAIN_EXTRA, EXPIRED_COLS, QUOTE_COLS, append_gz_csv, append_jsonl, day_dir,
                     read_csv_any, ts_iso, ts_minute, upsert_gz_csv)

log = logging.getLogger(__name__)

OPEN, CLOSE = dt.time(9, 15), dt.time(15, 30)
SNAPSHOT_END = dt.time(15, 45)     # cron can be late; a post-close snapshot still has the closing chain
EOD_START = dt.time(15, 35)
STEP = 50
EXPIRED_HALF_WIDTH = 800           # ATM +/- 800 points -> 33 strikes x CE/PE
EXPIRED_LOOKBACK_DAYS = 8          # candles from 8 days before expiry (the whole weekly cycle)
EXPIRED_RECENT_DAYS = 4            # look for expiries that ended up to 4 days ago (weekends, holidays)


# ----------------------------------------------------------------------------- gates
def market_gate(now: dt.datetime, end: dt.time = SNAPSHOT_END) -> Tuple[bool, str]:
    """Is this a time at which a snapshot makes sense? Holidays are detected later from the data
    (stale last-trade time), because no network-free holiday list exists."""
    now = now.astimezone(IST)
    if now.weekday() >= 5:
        return False, "weekend"
    if now.time() < OPEN:
        return False, "before 09:15 IST"
    if now.time() > end:
        return False, f"after {end.strftime('%H:%M')} IST"
    return True, "market hours"


def eod_gate(now: dt.datetime) -> Tuple[bool, str]:
    now = now.astimezone(IST)
    if now.weekday() >= 5:
        return False, "weekend"
    if now.time() < EOD_START:
        return False, "before 15:35 IST (session not finished)"
    return True, "after close"


# ----------------------------------------------------------------------------- context
def token_from_env() -> str:
    return (os.environ.get("UPSTOX_ANALYTICS_TOKEN", "") or os.environ.get("UPSTOX_ACCESS_TOKEN", "")).strip()


def get_provider() -> Optional[UpstoxProvider]:
    tok = token_from_env()
    return UpstoxProvider(token=tok) if tok else None


class Context:
    """Run state: where to write (`out`), where existing data lives (`base`), the provider, the
    clock and the error list that ends up in the run record."""

    def __init__(self, out: Path, base: Optional[Path] = None, up: Optional[UpstoxProvider] = None,
                 clock: Callable[[], dt.datetime] = now_ist, cfg: Optional[EngineConfig] = None,
                 session: Optional[requests.Session] = None, sources: Optional[List[Dict]] = None,
                 pause: float = 0.25, news_timeout: float = 8.0):
        self.out = Path(out)
        self.base = Path(base) if base else self.out
        self.up = up
        self.clock = clock
        self.cfg = cfg or EngineConfig.load()
        self.session = session
        self.sources = sources
        self.pause = pause
        self.news_timeout = news_timeout
        self.errors: List[Dict] = []
        self.token_invalid = False
        self.meta: Optional[Dict] = None

    def upstox_ok(self) -> bool:
        return self.up is not None and not self.token_invalid

    def _scrub(self, msg: str) -> str:
        tok = getattr(self.up, "token", "") if self.up else ""
        return msg.replace(tok, "***") if tok else msg

    def err(self, what: str, e) -> None:
        msg = self._scrub(f"{type(e).__name__}: {e}" if isinstance(e, BaseException) else str(e))[:240]
        log.warning("%s: %s", what, msg)
        self.errors.append({"what": what, "error": msg})

    def guard(self, what: str, fn: Callable, *a, **k):
        """Call an Upstox function; never raise. A 401 disables Upstox for the rest of the run."""
        if not self.upstox_ok():
            return None
        try:
            return fn(*a, **k)
        except PermissionError as e:
            self.token_invalid = True
            self.err(what, f"token rejected (401): {e}")
            return None
        except Exception as e:  # noqa: BLE001
            self.err(what, e)
            return None
        finally:
            if self.pause:
                time.sleep(self.pause)


# ----------------------------------------------------------------------------- instruments
def resolve_instruments(ctx: Context) -> Dict:
    """Keys for the near-month future, the heavyweights and the next option expiries. Resolved
    from the public instrument master once per day and cached in meta/instruments.json."""
    today = ctx.clock().astimezone(IST).date()
    for root in (ctx.out, ctx.base):
        p = Path(root) / "meta" / "instruments.json"
        try:
            m = json.loads(p.read_text(encoding="utf-8"))
            if m.get("date") == today.isoformat() and m.get("expiries"):
                ctx.meta = m
                return m
        except (OSError, ValueError):
            pass
    from ..data import instruments
    syms = ctx.cfg.extra.get("breadth_symbols") or instruments.HEAVYWEIGHTS
    m: Dict = {"date": today.isoformat(), "future": None, "equities": {}, "expiries": [], "resolved_at": ts_iso(ctx.clock())}
    try:
        master = instruments.load_master()
        futs = instruments.nifty_futures(master)
        if futs:
            f = futs[0]
            m["future"] = {"key": f["key"], "symbol": f.get("symbol"), "expiry": str(f["expiry"])}
        m["equities"] = instruments.equity_keys(list(syms), master)
        m["expiries"] = [e.isoformat() for e in instruments.option_expiries(master)[:4]]
    except Exception as e:  # noqa: BLE001
        ctx.err("instrument_master", e)
    if not m["expiries"] and ctx.upstox_ok():
        ex = ctx.guard("option_expiries", ctx.up.option_expiries) or []
        m["expiries"] = [e.isoformat() for e in ex if e >= today][:4]
    p = ctx.out / "meta" / "instruments.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(m, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    ctx.meta = m
    return m


def candle_targets(meta: Dict) -> Dict[str, str]:
    """Series name -> instrument key. Names match store/intraday/<name>_<1m|5m>.csv."""
    t = {"nifty": UpstoxProvider.NIFTY, "indiavix": UpstoxProvider.VIX, "banknifty": UpstoxProvider.BANKNIFTY}
    if (meta.get("future") or {}).get("key"):
        t["niftyfut"] = meta["future"]["key"]
    for sym, key in sorted((meta.get("equities") or {}).items()):
        t[series_name(sym)] = key
    return t


def series_name(symbol: str) -> str:
    """Same naming as data/upstox_history.py: lower case, '&' removed (M&M -> mm)."""
    return symbol.lower().replace("&", "")


# ----------------------------------------------------------------------------- quotes
def parse_trade_time(v) -> Optional[dt.datetime]:
    if v is None or v == "":
        return None
    try:
        s = str(v).strip()
        if s.isdigit():
            n = int(s)
            return dt.datetime.fromtimestamp(n / 1000 if n > 10 ** 11 else n, IST)
        t = pd.Timestamp(s)
        return (t.tz_localize(IST) if t.tzinfo is None else t.tz_convert(IST)).to_pydatetime()
    except (ValueError, OverflowError, OSError):
        return None


def flatten_quote(key: str, v: Dict, ts: str) -> Dict:
    ohlc = v.get("ohlc") or {}
    depth = v.get("depth") or {}
    ltt = parse_trade_time(v.get("last_trade_time"))
    row = {"ts": ts, "key": key, "symbol": v.get("symbol") or v.get("trading_symbol"),
           "last_price": v.get("last_price"), "open": ohlc.get("open"), "high": ohlc.get("high"),
           "low": ohlc.get("low"), "close": ohlc.get("close"), "last_trade_time": ltt.isoformat() if ltt else None}
    for f in ("volume", "average_price", "oi", "oi_day_high", "oi_day_low", "net_change", "total_buy_quantity",
              "total_sell_quantity", "lower_circuit_limit", "upper_circuit_limit"):
        row[f] = v.get(f)
    for side, name in (("buy", "bid"), ("sell", "ask")):
        levels = depth.get(side) or []
        for i in range(5):
            lv = levels[i] if i < len(levels) and isinstance(levels[i], dict) else {}
            row[f"{name}{i + 1}_price"] = lv.get("price")
            row[f"{name}{i + 1}_qty"] = lv.get("quantity")
            row[f"{name}{i + 1}_orders"] = lv.get("orders")
    return row


def quotes_fresh(quotes: Dict[str, Dict], now: dt.datetime) -> Tuple[Optional[bool], Optional[str]]:
    """True if any instrument traded today, False if all last trades are older (holiday), None if
    no last-trade time could be read (then the data is kept)."""
    times = [parse_trade_time(v.get("last_trade_time")) for v in quotes.values() if isinstance(v, dict)]
    times = [t for t in times if t]
    if not times:
        return None, None
    newest = max(times)
    return newest.astimezone(IST).date() == now.astimezone(IST).date(), newest.isoformat()


# ----------------------------------------------------------------------------- chains
def last_snapshot(path: Path, before: Optional[str] = None) -> Optional[pd.DataFrame]:
    """The most recent stored snapshot of one expiry (for OI-change / build-up metrics)."""
    df = read_csv_any(path)
    if df.empty or "ts" not in df:
        return None
    t = layout.parse_ts(df["ts"])
    if before is not None:
        df, t = df[t < pd.Timestamp(before)], t[t < pd.Timestamp(before)]
    if df.empty:
        return None
    return df[t == t.max()].copy()


def snapshot_chains(ctx: Context, expiries: List[dt.date], ts: dt.datetime, spot_hint: Optional[float]) -> Dict:
    from ..analytics.options_micro import chain_metrics
    today = ts.astimezone(IST).date()
    tss = ts_iso(ts)
    metrics: Dict = {"ts": tss}
    stats = {"chains": 0, "chain_rows": 0}
    for i, e in enumerate(expiries):
        ch = ctx.guard(f"option_chain {e}", ctx.up.option_chain, e)
        if ch is None or ch.empty:
            continue
        spot_s = pd.to_numeric(ch.get("spot"), errors="coerce").dropna() if "spot" in ch else pd.Series(dtype=float)
        spot = float(spot_s.iloc[0]) if len(spot_s) else spot_hint
        ch = ch.assign(ts=tss, expiry=e.isoformat())
        rel = f"chain_{e.isoformat()}.csv.gz"
        prev = last_snapshot(day_dir(ctx.base, today) / rel, before=tss)
        stats["chain_rows"] += append_gz_csv(day_dir(ctx.out, today) / rel, ch, CHAIN_COLS + CHAIN_EXTRA,
                                             keys=["ts", "strike", "type"])
        stats["chains"] += 1
        if not spot:
            continue
        try:
            T = year_fraction(ts, dt.datetime.combine(e, CLOSE, tzinfo=IST))
            m = chain_metrics(ch, spot, T, ctx.cfg.lot_size, prev=prev, with_gex=(i == 0))
            tag = "near" if i == 0 else "next"
            metrics.update({f"{tag}_{k}": v for k, v in m.items()})
            metrics[f"{tag}_expiry"] = e.isoformat()
        except Exception as ex:  # noqa: BLE001
            ctx.err(f"chain_metrics {e}", ex)
    if "near_atm_iv" in metrics and "next_atm_iv" in metrics:
        a, b = metrics["near_atm_iv"], metrics["next_atm_iv"]
        metrics["iv_term"] = a - b if a == a and b == b else None
    if len(metrics) > 1:
        metrics["source"] = "collector"
        clean = {k: (None if isinstance(v, float) and v != v else v) for k, v in metrics.items()}
        stats["metrics"] = append_jsonl(day_dir(ctx.out, today) / "chain_metrics.jsonl", [clean], key="ts")
    return stats


# ----------------------------------------------------------------------------- jobs
def snapshot(ctx: Context, force: bool = False) -> Dict:
    rec: Dict = {}
    now = ctx.clock()
    ok, why = market_gate(now)
    if ctx.up is None:
        rec["market"] = "skipped: no UPSTOX_ANALYTICS_TOKEN"
    elif not ok and not force:
        rec["market"] = f"skipped: {why}"
    else:
        meta = resolve_instruments(ctx)
        keys = [UpstoxProvider.NIFTY, UpstoxProvider.VIX, UpstoxProvider.BANKNIFTY]
        if (meta.get("future") or {}).get("key"):
            keys.append(meta["future"]["key"])
        keys += list((meta.get("equities") or {}).values())
        ts = ctx.clock()
        q = ctx.guard("quotes", ctx.up.quotes, keys) or {}
        fresh, newest = quotes_fresh(q, ts)
        rec["last_trade"] = newest
        if q and fresh is False and not force:
            rec["market"] = f"skipped: no trade today (holiday?), newest trade {newest}"
        elif ctx.upstox_ok():
            today = ts.astimezone(IST).date()
            rows = [flatten_quote(k, v, ts_iso(ts)) for k, v in q.items() if isinstance(v, dict)]
            rec["quotes"] = append_gz_csv(day_dir(ctx.out, today) / "quotes.csv.gz", pd.DataFrame(rows),
                                          QUOTE_COLS, keys=["ts", "key"]) if rows else 0
            spot = (q.get(UpstoxProvider.NIFTY) or {}).get("last_price")
            n = int(ctx.cfg.extra.get("record_expiries", 2))
            exps = [dt.date.fromisoformat(e) for e in meta.get("expiries", [])]
            exps = [e for e in exps if e >= today][:n]
            rec.update(snapshot_chains(ctx, exps, ts, float(spot) if spot else None))
            rec["market"] = "collected"
    rec["news"] = _news(ctx)
    return rec


def _news(ctx: Context) -> Dict:
    from . import news
    if ctx.upstox_ok() and ctx.meta is None:
        try:
            resolve_instruments(ctx)
        except Exception as e:  # noqa: BLE001
            ctx.err("instruments", e)
    return news.poll(ctx)


def news_only(ctx: Context, force: bool = False) -> Dict:
    return {"news": _news(ctx)}


def eod(ctx: Context, force: bool = False, expired: bool = True) -> Dict:
    rec: Dict = {}
    now = ctx.clock()
    today = now.astimezone(IST).date()
    ok, why = eod_gate(now)
    if ctx.up is None:
        rec["market"] = "skipped: no UPSTOX_ANALYTICS_TOKEN"
    elif not ok and not force:
        rec["market"] = f"skipped: {why}"
    else:
        meta = resolve_instruments(ctx)
        rec["candles"] = {}
        nifty_close = None
        for name, key in candle_targets(meta).items():
            for n in (1, 5):
                df = ctx.guard(f"candles {name} {n}m", ctx.up.intraday_today, key, "minutes", n)
                if df is None or df.empty:
                    continue
                df = df[[t.astimezone(IST).date() == today for t in df.index]]
                if df.empty:
                    continue
                if name == "nifty" and n == 1:
                    nifty_close = float(df["close"].iloc[-1])
                out = df.reset_index()
                out["ts"] = [ts_minute(t) for t in out["ts"]]
                rec["candles"][f"{name}_{n}m"] = upsert_gz_csv(
                    day_dir(ctx.out, today) / "candles" / f"{name}_{n}m.csv.gz", out.reindex(columns=CANDLE_COLS),
                    CANDLE_COLS, keys=["ts"])
        rec["market"] = "collected" if rec["candles"] else "no candles today (holiday?)"
        if expired and ctx.upstox_ok():
            rec["expired_options"] = expired_options(ctx, today, nifty_close)
    rec["news"] = _news(ctx)
    return rec


# ----------------------------------------------------------------------------- expired options
def _reference_spot(ctx: Context, e: dt.date, today: dt.date, nifty_close: Optional[float]) -> Optional[float]:
    if e == today and nifty_close:
        return nifty_close
    for root in (ctx.out, ctx.base):
        df = read_csv_any(day_dir(root, e) / "candles" / "nifty_1m.csv.gz")
        if len(df):
            v = pd.to_numeric(df["close"], errors="coerce").dropna()
            if len(v):
                return float(v.iloc[-1])
    if nifty_close:
        return nifty_close
    ltp = ctx.guard("ltp nifty", ctx.up.ltp, [UpstoxProvider.NIFTY]) or {}
    return next(iter(ltp.values()), None)


def expired_options(ctx: Context, today: dt.date, nifty_close: Optional[float]) -> Dict:
    """1-minute candles of a just-expired weekly expiry, strikes ATM +/- 800 (CE and PE).
    Needs Upstox Plus; a 401/403 from these endpoints is logged once and the step is skipped."""
    from ..data import expired_options as xo
    out: Dict = {}

    def plus_call(what, fn, *a, **k):
        try:
            return fn(*a, **k), None
        except PermissionError:
            return None, "401"
        except requests.HTTPError as e:
            code = getattr(e.response, "status_code", None)
            if code in (401, 403):
                return None, str(code)
            ctx.err(what, e)
            return None, "error"
        except Exception as e:  # noqa: BLE001
            ctx.err(what, e)
            return None, "error"
        finally:
            if ctx.pause:
                time.sleep(ctx.pause)

    exps, code = plus_call("expired_expiries", xo.expired_expiries, ctx.up)
    if code in ("401", "403"):
        log.warning("expired-instrument endpoints answered %s: Upstox Plus not active; skipped", code)
        return {"skipped": f"HTTP {code}: Upstox Plus not available"}
    recent = [e for e in (exps or []) if 0 <= (today - e).days <= EXPIRED_RECENT_DAYS]
    for e in recent:
        side = day_dir(ctx.base, e) / "expired_options_1m.json"
        try:
            if json.loads(side.read_text(encoding="utf-8")).get("complete"):
                out[e.isoformat()] = "already complete"
                continue
        except (OSError, ValueError):
            pass
        contracts, code = plus_call(f"expired_contracts {e}", xo.expired_contracts, ctx.up, e)
        if code in ("401", "403"):
            return {**out, "skipped": f"HTTP {code}: Upstox Plus not available"}
        if contracts is None or contracts.empty:
            out[e.isoformat()] = "no contracts"
            continue
        spot = _reference_spot(ctx, e, today, nifty_close)
        if not spot:
            out[e.isoformat()] = "no reference spot"
            continue
        atm = round(float(spot) / STEP) * STEP
        k = pd.to_numeric(contracts["strike"], errors="coerce")
        sel = contracts[(k >= atm - EXPIRED_HALF_WIDTH) & (k <= atm + EXPIRED_HALF_WIDTH)
                        & contracts["type"].isin(["CE", "PE"])]
        have = read_csv_any(day_dir(ctx.base, e) / "expired_options_1m.csv.gz")
        have_keys = set(have["key"]) if "key" in have else set()
        frames, failed = [], 0
        for _, c in sel.iterrows():
            if c["instrument_key"] in have_keys:
                continue
            df, code = plus_call(f"expired_candles {c['instrument_key']}", xo.expired_candles, ctx.up,
                                 c["instrument_key"], e - dt.timedelta(days=EXPIRED_LOOKBACK_DAYS), e, "1minute")
            if code in ("401", "403"):
                break
            if df is None or df.empty:
                failed += 1
                continue
            df = df.assign(strike=float(c["strike"]), type=c["type"], key=c["instrument_key"])
            df["ts"] = [ts_minute(t) for t in df["ts"]]
            frames.append(df.reindex(columns=EXPIRED_COLS))
        rows = 0
        if frames:
            rows = upsert_gz_csv(day_dir(ctx.out, e) / "expired_options_1m.csv.gz", pd.concat(frames),
                                 EXPIRED_COLS, keys=["key", "ts"], sort=["strike", "type", "ts"])
        got = len(have_keys) + len(frames)
        info = {"expiry": e.isoformat(), "atm": atm, "wanted": int(len(sel)), "got": int(got), "failed": failed,
                "complete": bool(len(sel)) and len(sel) - got <= 2, "rows_added": int(rows),
                "updated": ts_iso(ctx.clock())}
        p = day_dir(ctx.out, e) / "expired_options_1m.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(info, indent=1) + "\n", encoding="utf-8")
        out[e.isoformat()] = info
        if code in ("401", "403"):
            return {**out, "skipped": f"HTTP {code}: Upstox Plus not available"}
    if not recent:
        out["note"] = "no weekly expiry in the last few days"
    return out


# ----------------------------------------------------------------------------- run
JOBS = {"snapshot": snapshot, "eod": eod, "news": news_only}


def gh_annotation(level: str, title: str, msg: str) -> None:
    if os.environ.get("GITHUB_ACTIONS") == "true":
        print(f"::{level} title={title}::{msg}", flush=True)


def run(mode: str, out: Optional[Path] = None, data_branch_dir: Optional[Path] = None, force: bool = False,
        up: Optional[UpstoxProvider] = None, clock: Callable[[], dt.datetime] = now_ist,
        session: Optional[requests.Session] = None, sources: Optional[List[Dict]] = None,
        pause: float = 0.25, expired: bool = True, use_env_token: bool = True) -> Dict:
    """Collect into `out` (staging); if `data_branch_dir` is given, merge staging into it."""
    from .merge import merge_tree
    tmp = None
    if out is None:
        if data_branch_dir is None:
            from ..config import DATA_DIR
            out = DATA_DIR / "collect_out"
        else:
            tmp = out = Path(tempfile.mkdtemp(prefix="ne_collect_"))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if up is None and use_env_token:
        up = get_provider()
    if up is None:
        log.warning("UPSTOX_ANALYTICS_TOKEN is not set: Upstox data skipped, collecting free news only")
        gh_annotation("warning", "No Upstox token", "Repository secret UPSTOX_ANALYTICS_TOKEN is not set; "
                      "only free news feeds were collected.")
    ctx = Context(out, data_branch_dir, up, clock=clock, session=session, sources=sources, pause=pause)
    started = clock()
    rec: Dict = {"run_id": layout.new_run_id(), "mode": mode, "started": ts_iso(started), "token": up is not None}
    try:
        if mode == "eod":
            rec.update(eod(ctx, force=force, expired=expired))
        else:
            rec.update(JOBS[mode](ctx, force=force))
    except Exception as e:  # noqa: BLE001 - a bug: keep what was collected, report it, exit non-zero
        log.exception("collector failed")
        rec["fatal"] = ctx._scrub(f"{type(e).__name__}: {e}")[:300]
    finally:
        rec["finished"] = ts_iso(clock())
        rec["errors"] = ctx.errors[:60]
        rec["n_errors"] = len(ctx.errors)
        if ctx.token_invalid:
            rec["token_invalid"] = True
            gh_annotation("error", "Upstox token rejected", "Upstox answered 401: renew UPSTOX_ANALYTICS_TOKEN "
                          "(Analytics Tokens last one year). News collection continues.")
        append_jsonl(layout.run_record_path(out, started), [rec], key="run_id")
    if data_branch_dir is not None:
        rec["merge"] = merge_tree(out, Path(data_branch_dir))
    if tmp is not None:
        shutil.rmtree(tmp, ignore_errors=True)
    return rec
