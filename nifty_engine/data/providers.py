"""Broker-agnostic market data providers.

The model is trained and scored only on data any provider can supply (daily OHLC of
NIFTY, India VIX and global assets). Brokers are interchangeable *adapters* used for
(1) live quotes / option prices during paper trading and (2) optional intraday history.
Adding a broker = implementing `LiveProvider` (Zerodha, Dhan, Fyers, AngelOne ...).
"""

from __future__ import annotations

import datetime as dt
import json
import os
import time
import urllib.parse
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests

from ..calendar import IST

UA = {"User-Agent": "Mozilla/5.0 (nifty-engine)", "Accept": "application/json"}

# Daily series used by the model. Timing rule (enforced in features.py): anything that
# trades after 09:15 IST on day t is only used with a one-session lag.
YAHOO_SYMBOLS: Dict[str, str] = {
    "nifty": "^NSEI",
    "indiavix": "^INDIAVIX",
    "banknifty": "^NSEBANK",
    "spx": "^GSPC",
    "ndx": "^NDX",
    "usvix": "^VIX",
    "n225": "^N225",
    "hsi": "^HSI",
    "dxy": "DX-Y.NYB",
    "usdinr": "INR=X",
    "brent": "BZ=F",
    "us10y": "^TNX",
    "gold": "GC=F",
    # ---- related markets (feature groups "asia_extra" and "adr"; off in the base model) ----
    "kospi": "^KS11",          # Seoul opens 05:30 IST -> same-day opening gap
    "twii": "^TWII",           # Taipei opens 06:30 IST
    "shanghai": "000001.SS",   # Shanghai opens 07:00 IST
    "inda": "INDA",            # US-listed India ETF: how US investors repriced India overnight
    "hdb": "HDB", "ibn": "IBN", "infy_adr": "INFY", "wit": "WIT", "rdy": "RDY",   # NIFTY-heavyweight ADRs
    "hdfcbank": "HDFCBANK.NS", "icicibank": "ICICIBANK.NS", "infy_ns": "INFY.NS",
    "wipro_ns": "WIPRO.NS", "drreddy_ns": "DRREDDY.NS",
    "copper": "HG=F",
}

# Hourly series used for the "overnight_futures" group: their move between NIFTY's close (15:30 IST)
# and 09:00 IST the next morning is known before the open. Yahoo keeps ~730 days of hourly bars,
# so the engine stores them and the history grows over time. ES/NQ futures trade through the Asian
# night, which makes them the best free stand-in for GIFT Nifty.
YAHOO_HOURLY: Dict[str, str] = {
    "es": "ES=F", "nq": "NQ=F", "crude": "CL=F", "gold_f": "GC=F", "btc": "BTC-USD", "usdinr": "INR=X",
}


# --------------------------------------------------------------------------- Yahoo
class YahooProvider:
    """Free daily/intraday bars via Yahoo's public chart endpoint (no API key)."""

    BASE = "https://query1.finance.yahoo.com/v8/finance/chart/"

    def __init__(self, session: Optional[requests.Session] = None):
        self.s = session or requests.Session()
        self.s.headers.update(UA)

    def bars(self, symbol: str, start: str = "2007-01-01", end: Optional[str] = None,
             interval: str = "1d") -> pd.DataFrame:
        p1 = int(pd.Timestamp(start, tz="UTC").timestamp())
        p2 = int(pd.Timestamp(end, tz="UTC").timestamp()) if end else int(time.time()) + 86400
        url = f"{self.BASE}{urllib.parse.quote(symbol)}"
        params = {"period1": p1, "period2": p2, "interval": interval, "includePrePost": "false",
                  "events": "div,splits"}
        last_err = None
        for attempt in range(3):
            try:
                r = self.s.get(url, params=params, timeout=20)
                r.raise_for_status()
                res = r.json()["chart"]["result"][0]
                break
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(1.5 * (attempt + 1))
        else:
            raise RuntimeError(f"Yahoo fetch failed for {symbol}: {last_err}")
        ts = res.get("timestamp") or []
        if not ts:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        q = res["indicators"]["quote"][0]
        tz = res["meta"].get("exchangeTimezoneName", "UTC")
        idx = pd.to_datetime(ts, unit="s", utc=True).tz_convert(tz)
        df = pd.DataFrame({k: q.get(k) for k in ["open", "high", "low", "close", "volume"]}, index=idx)
        df = df.apply(pd.to_numeric, errors="coerce")
        if interval.endswith("d") or interval.endswith("wk"):
            df.index = pd.DatetimeIndex(df.index.date, name="date")
            df = df[~df.index.duplicated(keep="last")]
        df = df.dropna(subset=["open", "high", "low", "close"])
        df = df[(df[["open", "high", "low", "close"]] > 0).all(axis=1)]
        return df

    def last_price(self, symbol: str) -> Optional[float]:
        try:
            df = self.bars(symbol, start=(dt.date.today() - dt.timedelta(days=5)).isoformat(), interval="1m")
            return float(df["close"].iloc[-1]) if len(df) else None
        except Exception:  # noqa: BLE001
            return None


# --------------------------------------------------------------------------- Upstox
def totp_now(base32_secret: str, step: int = 30, digits: int = 6) -> str:
    """RFC 6238 TOTP (SHA1) without extra dependencies."""
    import base64
    import hashlib
    import hmac
    import struct
    sec = base32_secret.replace(" ", "").upper()
    key = base64.b32decode(sec + "=" * (-len(sec) % 8))
    msg = struct.pack(">Q", int(time.time()) // step)
    h = hmac.new(key, msg, hashlib.sha1).digest()
    o = h[-1] & 0x0F
    code = (struct.unpack(">I", h[o:o + 4])[0] & 0x7FFFFFFF) % (10 ** digits)
    return str(code).zfill(digits)


class UpstoxProvider:
    """Upstox v2/v3 REST adapter (market data, option chains, news; never orders).

    Token, in order of preference:
      * UPSTOX_ANALYTICS_TOKEN: the read-only Analytics Token (valid 1 year, GET endpoints only:
        quotes, candles, option chain, expired instruments with Upstox Plus, news, WebSocket feed).
        No daily login needed. Generate it in the Upstox developer console.
      * UPSTOX_ACCESS_TOKEN: a daily OAuth token (valid until ~03:30 IST next day), or the
        UPSTOX_API_KEY + UPSTOX_PIN + UPSTOX_TOTP_SECRET trio for automatic daily refresh.
    """

    BASE = "https://api.upstox.com"
    NIFTY = "NSE_INDEX|Nifty 50"
    VIX = "NSE_INDEX|India VIX"
    BANKNIFTY = "NSE_INDEX|Nifty Bank"

    def __init__(self, token: Optional[str] = None):
        self.token = (token or os.environ.get("UPSTOX_ANALYTICS_TOKEN", "")
                      or os.environ.get("UPSTOX_ACCESS_TOKEN", "")).strip()
        self.s = requests.Session()

    @property
    def available(self) -> bool:
        return bool(self.token) or bool(os.environ.get("UPSTOX_TOTP_SECRET"))

    @property
    def is_analytics_token(self) -> bool:
        return bool(os.environ.get("UPSTOX_ANALYTICS_TOKEN", "").strip()) and \
            self.token == os.environ.get("UPSTOX_ANALYTICS_TOKEN", "").strip()

    def _get(self, path: str, params: Optional[dict] = None, _retry: bool = True) -> dict:
        if not self.token and _retry and self.refresh_via_totp():
            pass
        r = self.s.get(self.BASE + path, params=params, timeout=10, headers={
            "Accept": "application/json", "Authorization": f"Bearer {self.token}"})
        if r.status_code == 401:
            if _retry and not self.is_analytics_token and self.refresh_via_totp():
                return self._get(path, params, _retry=False)
            raise PermissionError("Upstox token expired/invalid (401). Refresh UPSTOX_ANALYTICS_TOKEN "
                                  "or UPSTOX_ACCESS_TOKEN.")
        r.raise_for_status()
        return r.json()

    def refresh_via_totp(self) -> bool:
        """Official TOTP login (POST /v2/totp-login/authorization/token, documented on Upstox's
        Get Token page). Needs UPSTOX_API_KEY, UPSTOX_PIN, UPSTOX_TOTP_SECRET in .env."""
        key, pin, secret = (os.environ.get(k, "").strip() for k in ("UPSTOX_API_KEY", "UPSTOX_PIN", "UPSTOX_TOTP_SECRET"))
        if not (key and pin and secret):
            return False
        try:
            r = requests.post(self.BASE + "/v2/totp-login/authorization/token", timeout=10,
                              headers={"accept": "application/json", "Content-Type": "application/json", "x-api-key": key},
                              json={"totp": totp_now(secret), "pin": pin})
            js = r.json()
            tok = js.get("access_token") or (js.get("data") or {}).get("access_token")
            if tok:
                self.token = tok
                os.environ["UPSTOX_ACCESS_TOKEN"] = tok
                return True
        except Exception:  # noqa: BLE001
            pass
        return False

    def ltp(self, keys: List[str]) -> Dict[str, float]:
        data = self._get("/v2/market-quote/ltp", {"instrument_key": ",".join(keys)}).get("data", {})
        out = {}
        for v in data.values():
            k = v.get("instrument_token") or v.get("instrument_key")
            out[k] = float(v.get("last_price"))
        return out

    def quotes(self, keys: List[str]) -> Dict[str, dict]:
        """Full market quotes (OHLC, depth, OI, volume) keyed by instrument key. Max ~500 keys."""
        data = self._get("/v2/market-quote/quotes", {"instrument_key": ",".join(keys)}).get("data", {})
        return {(v.get("instrument_token") or v.get("instrument_key") or k): v for k, v in data.items()}

    def news(self, keys: List[str]) -> List[Dict]:
        """GET /v2/news (launched Apr-2026): articles for up to 30 instrument keys per call."""
        out = []
        for i in range(0, len(keys), 30):
            js = self._get("/v2/news", {"category": "instrument_keys", "instrument_keys": ",".join(keys[i:i + 30])})
            data = js.get("data") or []
            groups = data if isinstance(data, list) else [data]
            for g in groups:
                items = g.get("news") or g.get("articles") or g.get("items") or ([g] if g.get("headline") else [])
                for it in items:
                    ts = it.get("published_at") or it.get("publishedAt") or it.get("published_time")
                    try:
                        tsd = dt.datetime.fromtimestamp(int(ts) / 1000, IST) if ts and str(ts).isdigit() else (
                            pd.Timestamp(ts).tz_convert(IST).to_pydatetime() if ts else None)
                    except Exception:  # noqa: BLE001
                        tsd = None
                    out.append({"source": "upstox_news", "title": (it.get("headline") or it.get("title") or "").strip(),
                                "summary": (it.get("summary") or "").strip(), "url": it.get("link") or it.get("url"),
                                "ts": tsd, "instrument_key": g.get("instrument_key") or it.get("instrument_key")})
            time.sleep(0.2)
        return [o for o in out if o["title"]]

    def option_expiries(self) -> List[dt.date]:
        rows = self._get("/v2/option/contract", {"instrument_key": self.NIFTY}).get("data", [])
        return sorted({dt.date.fromisoformat(r["expiry"][:10]) for r in rows if r.get("expiry")})

    def option_chain(self, expiry: dt.date) -> pd.DataFrame:
        rows = self._get("/v2/option/chain", {"instrument_key": self.NIFTY,
                                              "expiry_date": expiry.isoformat()}).get("data", [])
        recs = []
        for r in rows:
            for side, key in (("CE", "call_options"), ("PE", "put_options")):
                o = r.get(key) or {}
                md, gk = o.get("market_data") or {}, o.get("option_greeks") or {}
                recs.append({
                    "strike": float(r["strike_price"]), "type": side,
                    "instrument_key": o.get("instrument_key"),
                    "ltp": md.get("ltp"), "bid": md.get("bid_price"), "ask": md.get("ask_price"),
                    "bid_qty": md.get("bid_qty"), "ask_qty": md.get("ask_qty"),
                    "oi": md.get("oi"), "prev_oi": md.get("prev_oi"), "volume": md.get("volume"),
                    "close_price": md.get("close_price"),
                    "iv": gk.get("iv"), "delta": gk.get("delta"), "gamma": gk.get("gamma"),
                    "theta": gk.get("theta"), "vega": gk.get("vega"), "pop": gk.get("pop"),
                    "spot": r.get("underlying_spot_price"), "pcr": r.get("pcr"),
                })
        return pd.DataFrame(recs)

    def intraday_today(self, key: str, unit: str = "minutes", interval: int = 5) -> pd.DataFrame:
        """Today's candles so far (v3 intraday endpoint; the historical endpoint ends yesterday)."""
        path = f"/v3/historical-candle/intraday/{urllib.parse.quote(key, safe='')}/{unit}/{interval}"
        c = self._get(path).get("data", {}).get("candles", [])
        if not c:
            return pd.DataFrame()
        df = pd.DataFrame(c, columns=["ts", "open", "high", "low", "close", "volume", "oi"][: len(c[0])])
        df["ts"] = pd.to_datetime(df["ts"]).dt.tz_convert(IST)
        return df.drop_duplicates("ts").set_index("ts").sort_index()

    def candles(self, key: str, unit: str, interval: int, start: dt.date, end: dt.date) -> pd.DataFrame:
        """v3 historical candles. unit: minutes|hours|days. Minute data is chunked monthly."""
        frames, cur = [], start
        step = dt.timedelta(days=28 if unit == "minutes" else 3650)
        while cur <= end:
            chunk_end = min(end, cur + step)
            path = (f"/v3/historical-candle/{urllib.parse.quote(key, safe='')}/{unit}/{interval}/"
                    f"{chunk_end.isoformat()}/{cur.isoformat()}")
            try:
                c = self._get(path).get("data", {}).get("candles", [])
            except requests.HTTPError:
                c = []
            if c:
                f = pd.DataFrame(c, columns=["ts", "open", "high", "low", "close", "volume", "oi"][: len(c[0])])
                frames.append(f)
            cur = chunk_end + dt.timedelta(days=1)
            time.sleep(0.25)
        if not frames:
            return pd.DataFrame()
        df = pd.concat(frames)
        df["ts"] = pd.to_datetime(df["ts"]).dt.tz_convert(IST)
        return df.drop_duplicates("ts").set_index("ts").sort_index()


# --------------------------------------------------------------------------- local NSE files
def load_participant_oi(folder: str) -> pd.DataFrame:
    """Load NSE 'Participant wise Open Interest' CSVs saved manually from
    nseindia.com > All Reports > Derivatives (file fao_participant_oi_DDMMYYYY.csv).
    NSE's terms prohibit automated scraping, so this is a *loader*, not a crawler.
    Returns per-date FII index-futures long share and net options positioning."""
    rows = []
    if not folder or not os.path.isdir(folder):
        return pd.DataFrame()
    for fn in sorted(os.listdir(folder)):
        if not fn.lower().startswith("fao_participant_oi_") or not fn.lower().endswith(".csv"):
            continue
        try:
            d = dt.datetime.strptime(fn[-12:-4], "%d%m%Y").date()
            df = pd.read_csv(os.path.join(folder, fn), skiprows=1)
            df.columns = [c.strip() for c in df.columns]
            fii = df[df.iloc[:, 0].astype(str).str.strip().str.upper() == "FII"].iloc[0]
            fl, fs = float(fii["Future Index Long"]), float(fii["Future Index Short"])
            cl, cs = float(fii["Option Index Call Long"]), float(fii["Option Index Call Short"])
            pl, ps = float(fii["Option Index Put Long"]), float(fii["Option Index Put Short"])
            rows.append({"date": d, "fii_fut_long_share": fl / max(fl + fs, 1.0),
                         "fii_opt_net": ((cl - cs) - (pl - ps)) / max(cl + cs + pl + ps, 1.0)})
        except Exception:  # noqa: BLE001
            continue
    return pd.DataFrame(rows).set_index("date").sort_index() if rows else pd.DataFrame()


# --------------------------------------------------------------------------- news
class GdeltError(RuntimeError):
    pass


class GdeltProvider:
    """GDELT DOC 2.0 API: free global news tone and headlines (rolling ~3 month window).

    GDELT allows ONE request every 5 seconds per IP and answers HTTP 429 (or a plain-text
    "Please limit requests" page) when that is exceeded. Several engine jobs call it within
    seconds of each other (08:40 lexicon tone + news capture), so every request goes through
    `_request`, which:
      * throttles to >= MIN_INTERVAL seconds between calls, shared across threads AND processes
        (the timestamp lives in store/cache/gdelt/_last_call),
      * retries 429 / throttle pages with backoff (Retry-After honoured, else 8s, 16s, 32s),
      * after repeated throttling, pauses all GDELT calls for COOLDOWN seconds (circuit breaker),
      * caches responses on disk (TTL: 20 min for live windows, forever for past windows),
        so re-running a command does not re-hit the API.
    """

    URL = "https://api.gdeltproject.org/api/v2/doc/doc"
    MIN_INTERVAL = 6.0
    MAX_RETRIES = 3
    COOLDOWN = 15 * 60
    LIVE_TTL = 20 * 60
    _lock = __import__("threading").Lock()

    def __init__(self, session: Optional[requests.Session] = None, cache_dir=None, sleep=time.sleep,
                 max_retries: Optional[int] = None, timeout: float = 40.0):
        """Live captures should fail fast (max_retries=1, timeout=15): RSS and Upstox cover live news,
        and a slow GDELT must never hold up anything else. Backfills keep the patient defaults."""
        from ..config import DATA_DIR
        if max_retries is not None:
            self.MAX_RETRIES = max_retries
        self.timeout = timeout
        self.s = session or requests.Session()
        self.s.headers.update({"User-Agent": "nifty-engine (research; +https://github.com/sagnikseal/nifty-tracker)"})
        self.dir = Path(cache_dir or (DATA_DIR / "cache" / "gdelt"))
        self.dir.mkdir(parents=True, exist_ok=True)
        self._sleep = sleep

    # ---- shared state (works across processes) --------------------------------------
    def _read_float(self, name: str) -> float:
        try:
            return float((self.dir / name).read_text().strip() or 0)
        except (OSError, ValueError):
            return 0.0

    def _write_float(self, name: str, v: float) -> None:
        try:
            (self.dir / name).write_text(repr(v))
        except OSError:
            pass

    def _cache_path(self, params: Dict) -> Path:
        import hashlib
        key = json.dumps(params, sort_keys=True)
        return self.dir / (hashlib.sha1(key.encode()).hexdigest()[:20] + ".json")

    @staticmethod
    def _throttled(r: requests.Response) -> bool:
        if r.status_code == 429:
            return True
        ctype = r.headers.get("Content-Type", "")
        return r.status_code == 200 and "json" not in ctype and "limit requests" in r.text[:300].lower()

    def _request(self, params: Dict, ttl: Optional[float]) -> Dict:
        cp = self._cache_path(params)
        if cp.exists() and (ttl is None or time.time() - cp.stat().st_mtime < ttl):
            try:
                return json.loads(cp.read_text(encoding="utf-8"))
            except ValueError:
                pass
        if time.time() < self._read_float("_cooldown_until"):
            raise GdeltError("GDELT cooling down after repeated rate limiting; skipped")
        delay = 8.0
        for attempt in range(self.MAX_RETRIES + 1):
            with self._lock:
                wait = self.MIN_INTERVAL - (time.time() - self._read_float("_last_call"))
                if wait > 0:
                    self._sleep(wait)
                self._write_float("_last_call", time.time())
                r = self.s.get(self.URL, params=params, timeout=self.timeout)
            if self._throttled(r):
                if attempt == self.MAX_RETRIES:
                    self._write_float("_cooldown_until", time.time() + self.COOLDOWN)
                    raise GdeltError(f"GDELT rate limit persisted after {self.MAX_RETRIES} retries; "
                                     f"pausing GDELT for {self.COOLDOWN // 60} min")
                ra = r.headers.get("Retry-After")
                self._sleep(float(ra) if ra and ra.replace(".", "", 1).isdigit() else delay)
                delay *= 2
                continue
            r.raise_for_status()
            try:
                js = r.json()
            except ValueError:            # plain-text error page for a bad query
                raise GdeltError("GDELT returned non-JSON: " + r.text[:120].strip())
            try:
                cp.write_text(json.dumps(js), encoding="utf-8")
            except OSError:
                pass
            return js
        raise GdeltError("unreachable")

    # ---- endpoints ----------------------------------------------------------------------
    def daily_tone(self, query: str, timespan: str = "3months") -> pd.Series:
        js = self._request({"query": query, "mode": "timelinetone", "format": "json", "timespan": timespan},
                           ttl=self.LIVE_TTL)
        pts = (js.get("timeline") or [{}])[0].get("data", [])
        if not pts:
            return pd.Series(dtype=float)
        s = pd.Series({pd.to_datetime(p["date"], format="%Y%m%dT%H%M%SZ", utc=True).tz_convert(IST).date():
                       float(p["value"]) for p in pts})
        return s.groupby(level=0).mean().sort_index()

    def articles(self, query: str, start: dt.datetime, end: dt.datetime, maxrecords: int = 250) -> List[Dict]:
        """Headlines with the time GDELT first saw them (`seendate`, UTC). Only the last ~3 months
        are searchable, which is enough to backfill news history after a local LLM's training cutoff."""
        fmt = "%Y%m%d%H%M%S"
        live = (dt.datetime.now(dt.timezone.utc) - end.astimezone(dt.timezone.utc)).total_seconds() < 6 * 3600
        js = self._request({
            "query": query, "mode": "artlist", "format": "json", "maxrecords": maxrecords, "sort": "datedesc",
            "startdatetime": start.astimezone(dt.timezone.utc).strftime(fmt),
            "enddatetime": end.astimezone(dt.timezone.utc).strftime(fmt)}, ttl=self.LIVE_TTL if live else None)
        out = []
        for a in js.get("articles", []):
            try:
                ts = pd.to_datetime(a["seendate"], format="%Y%m%dT%H%M%SZ", utc=True).tz_convert(IST).to_pydatetime()
            except Exception:  # noqa: BLE001
                continue
            out.append({"source": "gdelt:" + str(a.get("domain", "")), "title": (a.get("title") or "").strip(),
                        "url": a.get("url"), "ts": ts, "first_seen": ts})
        return [o for o in out if o["title"]]
