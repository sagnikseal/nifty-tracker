"""Live market recorder (Upstox Analytics Token; Upstox Plus for 30-level depth).

Option-chain snapshots, order-book depth and OI build-up cannot be downloaded later, so every
recorded session becomes history nobody else has. Two parts run side by side:

  * ChainPoller (REST, needs only the token): full NIFTY option chain with OI, volume, bid/ask
    and greeks for the nearest two expiries every `chain_poll_seconds`, plus derived metrics
    (PCR, walls, max pain, GEX, skew, implied move). -> store/live/<date>/chain_<expiry>.csv,
    store/live/<date>/chain_metrics.csv
  * DepthStreamer (WebSocket v3 via the official `upstox-python-sdk`): NIFTY futures and the
    ATM +/- N strikes in `full_d30` (Plus) or `full` mode, heavyweight stocks in `ltpc` mode for
    breadth; snapshotted every `depth_snapshot_seconds`. -> store/live/<date>/depth.csv

Files are compressed after the close (store/live/<date>/*.csv.gz).
"""

from __future__ import annotations

import csv
import datetime as dt
import gzip
import json
import logging
import shutil
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..analytics.options_micro import chain_metrics, depth_imbalance
from ..calendar import IST, now_ist, year_fraction
from ..config import DATA_DIR, EngineConfig
from .providers import UpstoxProvider

log = logging.getLogger(__name__)
LIVE = DATA_DIR / "live"

CHAIN_COLS = ["ts", "expiry", "strike", "type", "ltp", "bid", "ask", "bid_qty", "ask_qty", "oi", "prev_oi",
              "volume", "iv", "delta", "gamma", "theta", "vega", "spot"]
DEPTH_COLS = ["ts", "key", "ltp", "cp", "vtt", "oi", "tbq", "tsq", "bid1", "ask1", "bidq1", "askq1",
              "bidq_sum", "askq_sum", "imbalance", "levels", "iv", "delta", "gamma", "atp"]


def day_dir(d: Optional[dt.date] = None) -> Path:
    p = LIVE / (d or now_ist().date()).isoformat()
    p.mkdir(parents=True, exist_ok=True)
    return p


def _append_csv(path: Path, cols: List[str], rows: List[Dict]) -> None:
    if not rows:
        return
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerows(rows)


def compress_day(d: Optional[dt.date] = None) -> int:
    n = 0
    for f in day_dir(d).glob("*.csv"):
        with f.open("rb") as src, gzip.open(str(f) + ".gz", "ab") as dst:
            shutil.copyfileobj(src, dst)
        f.unlink()
        n += 1
    return n


def load_day(d: dt.date, name: str) -> pd.DataFrame:
    """Read one recorded table (plain or compressed) for a session."""
    base = LIVE / d.isoformat()
    frames = [pd.read_csv(p) for p in (base / f"{name}.csv.gz", base / f"{name}.csv") if p.exists()]
    if (base / f"{name}.jsonl").exists():
        frames.append(pd.read_json(base / f"{name}.jsonl", lines=True))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    if "ts" in df:
        df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert(IST)
    return df


def recorded_days() -> List[dt.date]:
    if not LIVE.exists():
        return []
    return sorted(dt.date.fromisoformat(p.name) for p in LIVE.iterdir() if p.is_dir() and p.name[:2] == "20")


# ----------------------------------------------------------------------------- REST chain poller
class ChainPoller:
    def __init__(self, cfg: EngineConfig, up: Optional[UpstoxProvider] = None):
        self.cfg, self.up = cfg, up or UpstoxProvider()
        self.prev: Dict[dt.date, pd.DataFrame] = {}
        self.latest: Dict[str, float] = {}

    def expiries(self) -> List[dt.date]:
        today = now_ist().date()
        try:
            ex = [e for e in self.up.option_expiries() if e >= today]
        except Exception as e:  # noqa: BLE001
            log.warning("expiries failed: %s", e)
            return []
        return ex[: int(self.cfg.extra.get("record_expiries", 2))]

    def poll_once(self) -> Dict[str, float]:
        ts = now_ist()
        out: Dict[str, float] = {"ts": ts.isoformat()}
        exps = self.expiries()
        for i, e in enumerate(exps):
            try:
                ch = self.up.option_chain(e)
            except Exception as ex:  # noqa: BLE001
                log.warning("chain %s failed: %s", e, ex)
                continue
            if ch.empty:
                continue
            spot = float(pd.to_numeric(ch["spot"], errors="coerce").dropna().iloc[0])
            ch = ch.assign(ts=ts.isoformat(), expiry=e.isoformat())
            _append_csv(day_dir() / f"chain_{e.isoformat()}.csv", CHAIN_COLS, ch.to_dict("records"))
            T = year_fraction(ts, dt.datetime.combine(e, dt.time(15, 30), tzinfo=IST))
            m = chain_metrics(ch, spot, T, self.cfg.lot_size, prev=self.prev.get(e), with_gex=(i == 0))
            self.prev[e] = ch
            tag = "near" if i == 0 else "next"
            out.update({f"{tag}_{k}": v for k, v in m.items()})
            out[f"{tag}_expiry"] = e.isoformat()
        if "near_atm_iv" in out and "next_atm_iv" in out:
            out["iv_term"] = out["near_atm_iv"] - out["next_atm_iv"]
        if len(out) > 1:
            with (day_dir() / "chain_metrics.jsonl").open("a", encoding="utf-8") as fh:
                clean = {k: (None if isinstance(v, float) and v != v else v) for k, v in out.items()}
                fh.write(json.dumps(clean, default=float) + "\n")
            self.latest = out
        return out


# ----------------------------------------------------------------------------- WebSocket depth
def flatten_feed(key: str, feed: dict) -> Dict:
    """Normalise one instrument's v3 feed (dict decoded from protobuf; field names vary slightly
    between SDK versions, so both camelCase v3 and short v2-style names are accepted)."""
    ff = feed.get("fullFeed") or feed.get("ff") or {}
    body = ff.get("marketFF") or ff.get("indexFF") or feed.get("firstLevelWithGreeks") or {}
    ltpc = body.get("ltpc") or feed.get("ltpc") or {}
    row = {"key": key, "ltp": ltpc.get("ltp"), "cp": ltpc.get("cp")}
    ml = body.get("marketLevel") or {}
    quotes = ml.get("bidAskQuote") or ([body["firstDepth"]] if body.get("firstDepth") else [])
    bq = [float(q.get("bidQ", q.get("bq", 0)) or 0) for q in quotes]
    aq = [float(q.get("askQ", q.get("aq", 0)) or 0) for q in quotes]
    bp = [float(q.get("bidP", q.get("bp", 0)) or 0) for q in quotes]
    ap = [float(q.get("askP", q.get("ap", 0)) or 0) for q in quotes]
    if quotes:
        row.update({"bid1": bp[0], "ask1": ap[0], "bidq1": bq[0], "askq1": aq[0], "bidq_sum": sum(bq),
                    "askq_sum": sum(aq), "imbalance": depth_imbalance(sum(bq), sum(aq)), "levels": len(quotes)})
    g = body.get("optionGreeks") or {}
    row.update({"vtt": body.get("vtt"), "oi": body.get("oi"), "tbq": body.get("tbq"), "tsq": body.get("tsq"),
                "iv": body.get("iv"), "delta": g.get("delta"), "gamma": g.get("gamma"), "atp": body.get("atp")})
    return row


class DepthStreamer:
    """Runs the official SDK streamer in a background thread and snapshots the latest state."""

    def __init__(self, cfg: EngineConfig, up: Optional[UpstoxProvider] = None):
        self.cfg, self.up = cfg, up or UpstoxProvider()
        self.state: Dict[str, Dict] = {}
        self.lock = threading.Lock()
        self.streamers = []

    def plan(self) -> List[Dict]:
        """Subscriptions: [{'mode':..., 'keys':[...]}] within Upstox limits (50 keys per full_d30 socket)."""
        from . import instruments
        plus = bool(self.cfg.extra.get("upstox_plus", True))
        n = int(self.cfg.extra.get("depth_strikes_each_side", 10))
        master = instruments.load_master()
        futs = instruments.nifty_futures(master)[:2]
        exps = instruments.option_expiries(master)
        spot = self.up.ltp([UpstoxProvider.NIFTY]).get(UpstoxProvider.NIFTY) if self.up.available else None
        keys_a = [f["key"] for f in futs]
        if exps and spot:
            opt = instruments.nifty_options(exps[0], master)
            atm = round(spot / 50) * 50
            sel = opt[(opt["strike"] >= atm - 50 * n) & (opt["strike"] <= atm + 50 * n)]
            keys_a += sel["instrument_key"].tolist()
        syms = self.cfg.extra.get("breadth_symbols") or instruments.HEAVYWEIGHTS
        eq = list(instruments.equity_keys(syms, master).values())
        return [{"mode": "full_d30" if plus else "full", "keys": keys_a[:50]},
                {"mode": "ltpc", "keys": [UpstoxProvider.NIFTY, UpstoxProvider.VIX, UpstoxProvider.BANKNIFTY] + eq}]

    def _on_message(self, msg) -> None:
        feeds = (msg or {}).get("feeds") or {}
        with self.lock:
            for k, f in feeds.items():
                self.state[k] = {**self.state.get(k, {}), **{a: b for a, b in flatten_feed(k, f).items() if b is not None}}

    def start(self) -> bool:
        try:
            import upstox_client  # official SDK: pip install upstox-python-sdk
        except ImportError:
            log.warning("upstox-python-sdk not installed: depth recording off (chain polling still runs)")
            return False
        if not self.up.token:
            return False
        conf = upstox_client.Configuration()
        conf.access_token = self.up.token
        for sub in self.plan():
            if not sub["keys"]:
                continue
            s = upstox_client.MarketDataStreamerV3(upstox_client.ApiClient(conf), sub["keys"], sub["mode"])
            s.on("message", self._on_message)
            s.on("error", lambda e: log.warning("feed error: %s", e))
            threading.Thread(target=s.connect, daemon=True).start()
            self.streamers.append(s)
        return bool(self.streamers)

    def snapshot(self) -> int:
        ts = now_ist().isoformat()
        with self.lock:
            rows = [{"ts": ts, **v} for v in self.state.values()]
        _append_csv(day_dir() / "depth.csv", DEPTH_COLS, rows)
        return len(rows)

    def stop(self) -> None:
        for s in self.streamers:
            try:
                s.disconnect()
            except Exception:  # noqa: BLE001
                pass
        self.streamers = []


def run_session(cfg: EngineConfig, until: dt.time = dt.time(15, 31), stop_event: Optional[threading.Event] = None) -> Dict:
    """Record one session (blocking). Use `python -m nifty_engine record`."""
    up = UpstoxProvider()
    if not up.available:
        return {"error": "set UPSTOX_ANALYTICS_TOKEN in .env"}
    poller, depth = ChainPoller(cfg, up), DepthStreamer(cfg, up)
    ws = depth.start() if cfg.extra.get("record_depth", True) else False
    poll_s = int(cfg.extra.get("chain_poll_seconds", 30))
    snap_s = int(cfg.extra.get("depth_snapshot_seconds", 5))
    last_poll = 0.0
    stats = {"chain_polls": 0, "depth_snapshots": 0, "websocket": ws}
    while now_ist().time() < until and not (stop_event and stop_event.is_set()):
        t = time.time()
        if now_ist().time() >= dt.time(9, 15) and t - last_poll >= poll_s:
            poller.poll_once()
            stats["chain_polls"] += 1
            last_poll = t
        if ws and now_ist().time() >= dt.time(9, 15):
            depth.snapshot()
            stats["depth_snapshots"] += 1
        time.sleep(snap_s if ws else 1.0)
    depth.stop()
    stats["files_compressed"] = compress_day()
    return stats


def latest_chain_metrics(d: Optional[dt.date] = None, at: Optional[dt.datetime] = None) -> Dict[str, float]:
    """Most recent recorded chain metrics at/before `at` (default now)."""
    df = load_day(d or now_ist().date(), "chain_metrics")
    if df.empty:
        return {}
    if at is not None:
        df = df[df["ts"] <= pd.Timestamp(at)]
    if df.empty:
        return {}
    r = df.iloc[-1].to_dict()
    return {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in r.items()}
