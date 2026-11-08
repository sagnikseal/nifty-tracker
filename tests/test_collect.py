"""GitHub Actions data collector (nifty_engine.collect): no network, the Upstox provider and the
HTTP session are fakes."""

from __future__ import annotations

import datetime as dt
import gzip
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import requests

os.environ.setdefault("NE_DATA_DIR", tempfile.mkdtemp(prefix="ne_test_"))

from nifty_engine.calendar import IST  # noqa: E402
from nifty_engine.collect import layout, market, merge, news, sync  # noqa: E402
from nifty_engine.data import recorder  # noqa: E402
from nifty_engine.data.providers import UpstoxProvider  # noqa: E402
from nifty_engine.semantic.news_llm import headline_id  # noqa: E402

DAY = dt.date(2026, 11, 10)            # a Tuesday (weekly expiry day)
NEAR, NEXT = dt.date(2026, 11, 10), dt.date(2026, 11, 17)
FUT = "NSE_FO|FUTNIFTY"
EQ = {"HDFCBANK": "NSE_EQ|INE040A01034", "M&M": "NSE_EQ|INE101A01026"}
TOKEN = "SECRET-TOKEN-123"


def at(h, m, d=DAY):
    return dt.datetime.combine(d, dt.time(h, m), tzinfo=IST)


class Clock:
    def __init__(self, t):
        self.t = t

    def __call__(self):
        return self.t


def fake_chain(expiry: dt.date, oi_shift: int = 0) -> pd.DataFrame:
    rows = []
    spot = 24512.0
    for k in range(24000, 25050, 50):
        for side in ("CE", "PE"):
            intrinsic = max(0.0, spot - k) if side == "CE" else max(0.0, k - spot)
            ltp = round(intrinsic + 60 * np.exp(-abs(k - spot) / 300) + 2, 2)
            rows.append({"strike": float(k), "type": side, "instrument_key": f"NSE_FO|{expiry:%d%m}{k}{side}",
                         "ltp": ltp, "bid": ltp - 0.5, "ask": ltp + 0.5, "bid_qty": 650, "ask_qty": 975,
                         "oi": 100000 + (k % 500) * 10 + oi_shift, "prev_oi": 90000, "volume": 50000,
                         "close_price": ltp, "iv": 12.5 + abs(k - spot) / 400, "delta": None, "gamma": None,
                         "theta": None, "vega": None, "pop": None, "spot": spot, "pcr": 1.05})
    return pd.DataFrame(rows)


def bars(day: dt.date, n_min: int, start=dt.time(9, 15), count=10, base=24500.0, oi=False):
    idx = pd.DatetimeIndex([dt.datetime.combine(day, start, tzinfo=IST) + dt.timedelta(minutes=n_min * i)
                            for i in range(count)], name="ts")
    c = base + np.arange(count, dtype=float)
    df = pd.DataFrame({"open": c, "high": c + 2, "low": c - 2, "close": c + 1, "volume": 1000.0,
                       "oi": 5000.0 if oi else 0.0}, index=idx)
    return df


class FakeUpstox:
    """Mimics UpstoxProvider's public methods (and `_get` for the expired-instrument endpoints)."""

    NIFTY, VIX, BANKNIFTY = UpstoxProvider.NIFTY, UpstoxProvider.VIX, UpstoxProvider.BANKNIFTY

    def __init__(self, day=DAY, fail_quotes_401=False, plus=True, last_trade_day=None):
        self.token = TOKEN
        self.day = day
        self.calls = []
        self.fail_quotes_401 = fail_quotes_401
        self.plus = plus
        self.last_trade_day = last_trade_day or day
        self.oi_shift = 0

    def quotes(self, keys):
        self.calls.append(("quotes", tuple(keys)))
        if self.fail_quotes_401:
            raise PermissionError("Upstox token expired/invalid (401). Bearer " + TOKEN)
        ltt = int(dt.datetime.combine(self.last_trade_day, dt.time(10, 0), tzinfo=IST).timestamp() * 1000)
        out = {}
        for k in keys:
            depth = {"buy": [{"price": 100 - i, "quantity": 10 * (i + 1), "orders": i + 1} for i in range(5)],
                     "sell": [{"price": 101 + i, "quantity": 12 * (i + 1), "orders": i + 1} for i in range(5)]}
            out[k] = {"instrument_token": k, "symbol": k.split("|")[-1], "last_price": 24512.0 if "Nifty 50" in k else 100.5,
                      "ohlc": {"open": 24400, "high": 24550, "low": 24390, "close": 24450}, "volume": 12345,
                      "average_price": 100.2, "oi": 777 if k == FUT else 0, "net_change": 62.0,
                      "total_buy_quantity": 5000, "total_sell_quantity": 4000, "last_trade_time": str(ltt),
                      "depth": depth if "NSE_INDEX" not in k else {}}
        return out

    def option_chain(self, expiry):
        self.calls.append(("chain", expiry))
        return fake_chain(expiry, self.oi_shift)

    def option_expiries(self):
        return [NEAR, NEXT]

    def intraday_today(self, key, unit="minutes", interval=5):
        self.calls.append(("intraday", key, interval))
        if key == EQ["M&M"] and interval == 5:
            raise requests.HTTPError("500 Server Error")       # one bad instrument
        return bars(self.day, interval, count=8, oi=(key == FUT))

    def news(self, keys):
        self.calls.append(("news", tuple(keys)))
        return [{"source": "upstox_news", "title": "HDFC Bank Q2 profit beats estimates", "summary": "",
                 "url": "https://x/1", "ts": at(9, 50), "instrument_key": EQ["HDFCBANK"]}]

    def ltp(self, keys):
        return {UpstoxProvider.NIFTY: 24512.0}

    def _get(self, path, params=None):
        self.calls.append(("get", path))
        if not self.plus:
            r = requests.Response()
            r.status_code = 403
            raise requests.HTTPError("403 Forbidden", response=r)
        if path.endswith("/expiries"):
            return {"data": [(NEAR - dt.timedelta(days=7)).isoformat(), NEAR.isoformat()]}
        if path.endswith("/option/contract"):
            rows = [{"instrument_key": f"NSE_FO|X{k}{t}", "strike_price": float(k), "instrument_type": t}
                    for k in range(23000, 26050, 50) for t in ("CE", "PE")]
            return {"data": rows}
        if "historical-candle" in path:
            idx = [dt.datetime.combine(NEAR, dt.time(9, 15), tzinfo=IST) + dt.timedelta(minutes=i) for i in range(3)]
            return {"data": {"candles": [[t.isoformat(), 10, 11, 9, 10.5, 100, 2000] for t in idx]}}
        return {"data": []}


RSS = b"""<?xml version="1.0"?><rss><channel>
<item><title>RBI keeps repo rate unchanged at 5.5%</title><link>https://x/rbi</link>
<pubDate>Tue, 10 Nov 2026 09:40:00 +0530</pubDate><description>Policy</description></item>
<item><title>Sensex, Nifty open higher; IT stocks gain</title><link>https://x/open</link>
<pubDate>Tue, 10 Nov 2026 09:20:00 +0530</pubDate></item>
</channel></rss>"""


class FakeResp:
    def __init__(self, content=RSS, status=200):
        self.status_code, self.content, self.headers = status, content, {}
        self.text = content.decode()
        self.ok = status < 400

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


class FakeSession:
    def __init__(self, content=RSS):
        self.headers = {}
        self.content = content
        self.n = 0

    def get(self, url, headers=None, timeout=None):
        self.n += 1
        return FakeResp(self.content)


SOURCES = [{"name": "fake_feed", "category": "media_in", "url": "https://example.invalid/rss"}]


def seed_meta(root: Path, day=DAY):
    p = Path(root) / "meta" / "instruments.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"date": day.isoformat(), "future": {"key": FUT, "symbol": "NIFTY FUT", "expiry": "2026-11-24"},
                             "equities": EQ, "expiries": [NEAR.isoformat(), NEXT.isoformat()]}), encoding="utf-8")


def collect(mode, db, t, up, staging=None, **kw):
    staging = staging or Path(tempfile.mkdtemp(prefix="stg_"))
    return market.run(mode, out=staging, data_branch_dir=db, up=up, clock=Clock(t), session=FakeSession(),
                      sources=SOURCES, pause=0, use_env_token=False, **kw)


# ============================================================================= tests
class TestGates(unittest.TestCase):
    def test_market_hours_gate(self):
        self.assertTrue(market.market_gate(at(9, 15))[0])
        self.assertTrue(market.market_gate(at(12, 0))[0])
        self.assertTrue(market.market_gate(at(15, 44))[0])            # late cron still gets the close
        self.assertFalse(market.market_gate(at(9, 14))[0])
        self.assertFalse(market.market_gate(at(15, 46))[0])
        self.assertEqual(market.market_gate(at(11, 0, dt.date(2026, 11, 14)))[1], "weekend")   # Saturday
        # a UTC timestamp is converted: 04:00 UTC = 09:30 IST
        self.assertTrue(market.market_gate(dt.datetime(2026, 11, 10, 4, 0, tzinfo=dt.timezone.utc))[0])
        self.assertFalse(market.eod_gate(at(15, 0))[0])
        self.assertTrue(market.eod_gate(at(16, 10))[0])
        self.assertFalse(market.eod_gate(at(16, 10, dt.date(2026, 11, 15)))[0])

    def test_quotes_freshness_detects_holiday(self):
        q = FakeUpstox(last_trade_day=DAY - dt.timedelta(days=1)).quotes([UpstoxProvider.NIFTY])
        self.assertIs(market.quotes_fresh(q, at(10, 0))[0], False)
        q = FakeUpstox().quotes([UpstoxProvider.NIFTY])
        self.assertIs(market.quotes_fresh(q, at(10, 0))[0], True)
        self.assertIsNone(market.quotes_fresh({"x": {"last_price": 1}}, at(10, 0))[0])

    def test_snapshot_skips_outside_hours_and_on_holiday(self):
        db = Path(tempfile.mkdtemp())
        seed_meta(db)
        up = FakeUpstox()
        rec = collect("snapshot", db, at(8, 30), up)
        self.assertTrue(rec["market"].startswith("skipped"))
        self.assertFalse(any(c[0] in ("quotes", "chain") for c in up.calls))
        self.assertGreater(rec["news"]["new"], 0)                     # news is collected anyway
        up = FakeUpstox(last_trade_day=DAY - dt.timedelta(days=1))
        rec = collect("snapshot", db, at(10, 0), up)
        self.assertIn("holiday", rec["market"])
        self.assertFalse((db / "days" / DAY.isoformat() / "quotes.csv.gz").exists())
        self.assertFalse(any(c[0] == "chain" for c in up.calls))


class TestGzipStore(unittest.TestCase):
    def test_append_dedupe_and_multimember(self):
        p = Path(tempfile.mkdtemp()) / "x" / "t.csv.gz"
        cols = ["ts", "strike", "type", "ltp"]
        a = pd.DataFrame({"ts": ["2026-11-10T10:00:00.000000+05:30"] * 2, "strike": [24500.0, 24500.0],
                          "type": ["CE", "PE"], "ltp": [100.0, 90.0]})
        self.assertEqual(layout.append_gz_csv(p, a, cols, ["ts", "strike", "type"]), 2)
        size1 = p.stat().st_size
        # same keys spelled differently (int strike, ts without fraction / in UTC) + one new row
        b = pd.DataFrame({"ts": ["2026-11-10T04:30:00+00:00", "2026-11-10T10:15:00.000000+05:30"],
                          "strike": [24500, 24500], "type": ["CE", "CE"], "ltp": [101.0, 102.0]})
        self.assertEqual(layout.append_gz_csv(p, b, cols, ["ts", "strike", "type"]), 1)
        self.assertEqual(layout.append_gz_csv(p, b, cols, ["ts", "strike", "type"]), 0)
        raw = p.read_bytes()
        self.assertEqual(raw.count(b"\x1f\x8b\x08"), 2)              # two gzip members, first untouched
        self.assertTrue(raw.startswith(raw[:size1]))
        text = gzip.decompress(raw).decode()
        self.assertEqual(text.count("ts,strike,type,ltp"), 1)
        df = pd.read_csv(p)
        self.assertEqual(len(df), 3)
        self.assertEqual(df["ltp"].tolist(), [100.0, 90.0, 102.0])   # existing rows are never overwritten

    def test_upsert_rewrites_only_on_change(self):
        p = Path(tempfile.mkdtemp()) / "c.csv.gz"
        df = bars(DAY, 1, count=5).reset_index()
        df["ts"] = [layout.ts_minute(t) for t in df["ts"]]
        self.assertEqual(layout.upsert_gz_csv(p, df, layout.CANDLE_COLS, ["ts"]), 5)
        b1 = p.read_bytes()
        self.assertEqual(layout.upsert_gz_csv(p, df, layout.CANDLE_COLS, ["ts"]), 0)
        self.assertEqual(p.read_bytes(), b1)                          # deterministic, untouched
        df2 = df.copy()
        df2.loc[4, "close"] = 1.0                                     # a corrected last bar + 1 new bar
        extra = df2.iloc[[4]].copy()
        extra["ts"] = layout.ts_minute(at(9, 20))
        self.assertEqual(layout.upsert_gz_csv(p, pd.concat([extra, df2]), layout.CANDLE_COLS, ["ts"]), 2)
        out = pd.read_csv(p)
        self.assertEqual(len(out), 6)
        self.assertTrue(out["ts"].is_monotonic_increasing)
        self.assertEqual(float(out.iloc[4]["close"]), 1.0)

    def test_ts_formats_are_ist_iso(self):
        s = layout.ts_iso(dt.datetime(2026, 11, 10, 4, 30, tzinfo=dt.timezone.utc))
        self.assertEqual(s, "2026-11-10T10:00:00.000000+05:30")
        self.assertEqual(layout.ts_minute(pd.Timestamp("2026-11-10 09:15", tz=IST)), "2026-11-10T09:15:00+05:30")


class TestNews(unittest.TestCase):
    def test_ids_match_news_llm_and_dedupe(self):
        items = [{"title": "Sensex, Nifty open higher; IT stocks gain", "source": "a", "ts": at(9, 20)},
                 {"title": "SENSEX NIFTY OPEN HIGHER IT STOCKS GAIN", "source": "b"},       # same normalised text
                 {"title": "  ", "source": "c"}]
        recs = news.to_records(items, at(9, 31))
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0]["id"], headline_id("Sensex, Nifty open higher; IT stocks gain"))
        self.assertEqual(list(recs[0]), layout.NEWS_FIELDS)
        self.assertEqual(recs[0]["first_seen"], "2026-11-10T09:31:00+05:30")
        self.assertEqual(recs[0]["published"], "2026-11-10T09:20:00+05:30")
        self.assertEqual(news.to_records(items, at(9, 31), known={recs[0]["id"]}), [])

    def test_poll_twice_and_across_days_never_duplicates(self):
        db = Path(tempfile.mkdtemp())
        r1 = collect("news", db, at(23, 0, DAY - dt.timedelta(days=1)), None)
        self.assertEqual(r1["news"]["new"], 2)
        r2 = collect("news", db, at(7, 0), None)                     # next day, same feed content
        self.assertEqual(r2["news"]["new"], 0)
        self.assertFalse((db / "days" / DAY.isoformat() / "news.jsonl").exists())
        lines = layout.read_jsonl(db / "days" / (DAY - dt.timedelta(days=1)).isoformat() / "news.jsonl")
        self.assertEqual(len(lines), 2)
        self.assertEqual(len({r["id"] for r in lines}), 2)

    def test_upstox_news_for_heavyweights(self):
        db = Path(tempfile.mkdtemp())
        seed_meta(db)
        up = FakeUpstox()
        rec = collect("news", db, at(9, 55), up)
        self.assertEqual(rec["news"]["upstox_items"], 1)
        self.assertIn(("news", tuple(EQ.values())), up.calls)
        srcs = {r["source"] for r in layout.read_jsonl(db / "days" / DAY.isoformat() / "news.jsonl")}
        self.assertEqual(srcs, {"fake_feed", "upstox_news"})


class TestSnapshotToRecorder(unittest.TestCase):
    def test_chain_snapshot_roundtrip_into_recorder_format(self):
        db = Path(tempfile.mkdtemp())
        seed_meta(db)
        up = FakeUpstox()
        r1 = collect("snapshot", db, at(9, 31), up)
        self.assertEqual(r1["market"], "collected")
        self.assertEqual(r1["chains"], 2)
        up.oi_shift = 5000                                            # OI build-up between snapshots
        r2 = collect("snapshot", db, at(9, 47), up)
        self.assertEqual(r2["chains"], 2)
        d = db / "days" / DAY.isoformat()
        ch = pd.read_csv(d / f"chain_{NEAR.isoformat()}.csv.gz")
        self.assertEqual(list(ch.columns[:len(recorder.CHAIN_COLS)]), recorder.CHAIN_COLS)
        self.assertEqual(ch["ts"].nunique(), 2)
        self.assertEqual(len(ch), 2 * len(fake_chain(NEAR)))
        q = pd.read_csv(d / "quotes.csv.gz")
        self.assertEqual(len(q), 2 * (4 + len(EQ)))
        self.assertIn(FUT, set(q["key"]))
        self.assertEqual(float(q[q["key"] == FUT]["bid1_price"].iloc[0]), 100.0)
        m = layout.read_jsonl(d / "chain_metrics.jsonl")
        self.assertEqual(len(m), 2)
        self.assertIn("near_pcr_oi", m[0])
        self.assertIn("next_atm_iv", m[0])
        self.assertNotIn("near_doi_ce", m[0])                         # no earlier snapshot yet
        self.assertGreater(m[1]["near_doi_ce"], 0)                    # second one compares with the first
        man = json.loads((db / "MANIFEST.json").read_text())
        self.assertEqual(man["days"][DAY.isoformat()]["snapshots"], 2)
        self.assertEqual(man["last_runs"]["snapshot"]["market"], "collected")
        self.assertTrue((db / "README.md").exists())

        # ---- sync into a temp store; the recorder's own readers must understand it
        store = Path(tempfile.mkdtemp())
        res = sync.sync(db, store, now=at(18, 0))
        self.assertGreater(res["chain_rows"], 0)
        with mock.patch.object(recorder, "LIVE", store / "live"):
            df = recorder.load_day(DAY, f"chain_{NEAR.isoformat()}")
            self.assertEqual(list(df.columns), recorder.CHAIN_COLS)
            self.assertEqual(len(df), len(ch))
            self.assertEqual(str(df["ts"].dt.tz), "Asia/Kolkata")
            self.assertEqual(df["ts"].nunique(), 2)
            lm = recorder.latest_chain_metrics(DAY)
            self.assertAlmostEqual(lm["near_doi_ce"], m[1]["near_doi_ce"])
            early = recorder.latest_chain_metrics(DAY, at=at(9, 40))
            self.assertAlmostEqual(early["near_pcr_oi"], m[0]["near_pcr_oi"])
            dep = recorder.load_day(DAY, "depth")
            self.assertEqual(list(dep.columns), recorder.DEPTH_COLS)
            fut = dep[dep["key"] == FUT].iloc[0]
            self.assertEqual(fut["bidq_sum"], 150)
            self.assertAlmostEqual(fut["imbalance"], (150 - 180) / 330)
            self.assertAlmostEqual(fut["cp"], 100.5 - 62.0)

    def test_sync_merges_with_existing_recorder_files(self):
        """Mac recorded the morning (plain csv, recorder ts format); CI has later snapshots."""
        db = Path(tempfile.mkdtemp())
        seed_meta(db)
        collect("snapshot", db, at(11, 0), FakeUpstox())
        store = Path(tempfile.mkdtemp())
        live = store / "live" / DAY.isoformat()
        live.mkdir(parents=True)
        local = fake_chain(NEAR).assign(ts=at(9, 30, ).replace(microsecond=123456).isoformat(), expiry=NEAR.isoformat())
        local[recorder.CHAIN_COLS].to_csv(live / f"chain_{NEAR.isoformat()}.csv", index=False)
        (live / "chain_metrics.jsonl").write_text(json.dumps({"ts": at(9, 30).replace(microsecond=5).isoformat(),
                                                             "near_pcr_oi": 0.5}) + "\n")
        # before 15:45 today nothing is touched (the recorder may be writing)
        res = sync.sync(db, store, now=at(12, 0))
        self.assertTrue(res["deferred"])
        self.assertTrue((live / f"chain_{NEAR.isoformat()}.csv").exists())
        res = sync.sync(db, store, now=at(16, 0))
        self.assertFalse((live / f"chain_{NEAR.isoformat()}.csv").exists())          # folded into .csv.gz
        with mock.patch.object(recorder, "LIVE", store / "live"):
            df = recorder.load_day(DAY, f"chain_{NEAR.isoformat()}")
            self.assertEqual(df["ts"].nunique(), 2)
            self.assertTrue(df["ts"].is_monotonic_increasing)
            mm = recorder.load_day(DAY, "chain_metrics")
            self.assertEqual(len(mm), 2)
            self.assertEqual(recorder.latest_chain_metrics(DAY, at=at(10, 0))["near_pcr_oi"], 0.5)


class TestEodAndSync(unittest.TestCase):
    def test_eod_candles_expired_options_and_full_sync(self):
        db = Path(tempfile.mkdtemp())
        seed_meta(db)
        up = FakeUpstox()
        rec = collect("eod", db, at(16, 10), up)
        self.assertEqual(rec["market"], "collected")
        self.assertIn("nifty_1m", rec["candles"])
        self.assertIn("niftyfut_5m", rec["candles"])
        self.assertIn("hdfcbank_1m", rec["candles"])
        self.assertNotIn("mm_5m", rec["candles"])                     # failed instrument skipped ...
        self.assertIn("mm_1m", rec["candles"])                        # ... the rest still collected
        self.assertTrue(any("mm 5m" in e["what"] for e in rec["errors"]))
        x = rec["expired_options"][NEAR.isoformat()]
        self.assertEqual(x["wanted"], 66)                             # 33 strikes x CE/PE around ATM
        self.assertEqual(x["atm"], 24500)                             # NIFTY 1m close of the expiry day
        self.assertTrue(x["complete"])
        xo = pd.read_csv(db / "days" / NEAR.isoformat() / "expired_options_1m.csv.gz")
        self.assertEqual(list(xo.columns), layout.EXPIRED_COLS)
        self.assertEqual(xo["strike"].min(), 23700)
        self.assertEqual(xo["strike"].max(), 25300)
        # second run: already complete -> no expired-candle calls
        up2 = FakeUpstox()
        rec2 = collect("eod", db, at(19, 10), up2)
        self.assertEqual(rec2["expired_options"][NEAR.isoformat()], "already complete")
        self.assertFalse(any(c[0] == "get" and "historical-candle" in c[1] for c in up2.calls))

        # ---- sync into a store that already has older bars and one of the headlines
        store = Path(tempfile.mkdtemp())
        (store / "intraday").mkdir(parents=True)
        old = bars(DAY - dt.timedelta(days=1), 5, count=3)
        old.to_csv(store / "intraday" / "nifty_5m.csv")
        hd = store / "news" / "headlines"
        hd.mkdir(parents=True)
        first = layout.read_jsonl(db / "days" / DAY.isoformat() / "news.jsonl")[0]
        (hd / f"{(DAY - dt.timedelta(days=1)).isoformat()}.jsonl").write_text(json.dumps(first) + "\n")
        res = sync.sync(db, store, now=at(20, 0))
        self.assertFalse(res["errors"])
        n5 = pd.read_csv(store / "intraday" / "nifty_5m.csv", index_col=0)
        self.assertEqual(n5.index.name, "ts")
        self.assertEqual(list(n5.columns), ["open", "high", "low", "close", "volume", "oi"])
        self.assertEqual(len(n5), 3 + 8)
        idx = pd.to_datetime(n5.index, utc=True)
        self.assertTrue(idx.is_monotonic_increasing and idx.is_unique)
        self.assertTrue((store / "intraday" / "mm_1m.csv").exists())
        fut = pd.read_csv(store / "intraday" / "niftyfut_1m.csv", index_col=0)
        self.assertTrue((fut["oi"] == 5000).all())
        opt = pd.read_csv(store / "options_hist_1m" / f"{NEAR.isoformat()}.csv")
        self.assertEqual(list(opt.columns), layout.EXPIRED_COLS)
        self.assertEqual(len(opt), 66 * 3)
        ids = [r["id"] for f in hd.glob("*.jsonl") for r in layout.read_jsonl(f)]
        self.assertEqual(len(ids), len(set(ids)))                     # the pre-existing headline was not re-added
        self.assertIn(first["id"], ids)
        before = {p: p.read_bytes() for p in store.rglob("*") if p.is_file() and p.name != "collect_sync.json"}
        res2 = sync.sync(db, store, now=at(20, 5))
        self.assertEqual(res2["files_read"], 0)                       # state: nothing changed
        res3 = sync.sync(db, store, now=at(20, 10), full=True)        # full re-read: still idempotent
        self.assertEqual(res3["bars"] + res3["news"] + res3["option_bars"] + res3["chain_rows"], 0)
        after = {p: p.read_bytes() for p in store.rglob("*") if p.is_file() and p.name != "collect_sync.json"}
        self.assertEqual(before, after)

    def test_expired_options_without_plus_is_skipped(self):
        db = Path(tempfile.mkdtemp())
        seed_meta(db)
        rec = collect("eod", db, at(16, 10), FakeUpstox(plus=False))
        self.assertIn("Upstox Plus", rec["expired_options"]["skipped"])
        self.assertEqual(rec["market"], "collected")

    def test_merge_is_idempotent(self):
        db = Path(tempfile.mkdtemp())
        seed_meta(db)
        stg = Path(tempfile.mkdtemp())
        collect("snapshot", db, at(10, 0), FakeUpstox(), staging=stg)
        snap = {p: p.read_bytes() for p in db.rglob("*") if p.is_file() and p.name != "MANIFEST.json"}
        res = merge.merge_tree(stg, db)                               # what the push-retry loop does
        self.assertEqual(res["rows"], 0)
        self.assertEqual(snap, {p: p.read_bytes() for p in db.rglob("*") if p.is_file() and p.name != "MANIFEST.json"})


class TestTokenHandling(unittest.TestCase):
    def test_cli_exits_zero_without_token_and_still_collects_news(self):
        out = Path(tempfile.mkdtemp())
        env = {k: v for k, v in os.environ.items() if k not in ("UPSTOX_ANALYTICS_TOKEN", "UPSTOX_ACCESS_TOKEN",
                                                                  "UPSTOX_TOTP_SECRET")}
        from nifty_engine.collect.__main__ import main
        from nifty_engine.data import news_sources
        for cmd in ("snapshot", "eod", "news"):
            buf = io.StringIO()
            with mock.patch.dict(os.environ, env, clear=True), \
                    mock.patch.object(news_sources, "load_sources", return_value=SOURCES), \
                    mock.patch.object(news_sources.requests, "Session", FakeSession), redirect_stdout(buf):
                code = main([cmd, "--out", str(out)])
            self.assertEqual(code, 0)
            rec = json.loads(buf.getvalue())
            self.assertFalse(rec["token"])
            self.assertIn("news", rec)
            if cmd != "news":
                self.assertIn("no UPSTOX_ANALYTICS_TOKEN", rec["market"])
        heads = list(out.glob("days/*/news.jsonl"))
        self.assertTrue(heads)
        self.assertTrue(list(out.glob("runs/*.jsonl")))

    def test_rejected_token_stops_upstox_calls_and_is_never_logged(self):
        db = Path(tempfile.mkdtemp())
        seed_meta(db)
        up = FakeUpstox(fail_quotes_401=True)
        with self.assertLogs("nifty_engine.collect.market", level="WARNING") as logs:
            rec = collect("snapshot", db, at(10, 0), up)
        self.assertTrue(rec["token_invalid"])
        self.assertFalse(any(c[0] in ("chain", "news") for c in up.calls))     # nothing after the 401
        self.assertGreater(rec["news"]["new"], 0)                               # free news still collected
        blob = json.dumps(rec) + "\n".join(logs.output)
        blob += "".join(p.read_text(encoding="utf-8") for p in db.rglob("*.jsonl"))
        self.assertNotIn(TOKEN, blob)


class TestFatal(unittest.TestCase):
    def test_bug_keeps_collected_data_and_is_reported(self):
        db = Path(tempfile.mkdtemp())

        def broken(ctx, force=False):
            news.poll(ctx)
            raise ValueError("boom")

        with mock.patch.dict(market.JOBS, {"news": broken}), self.assertLogs("nifty_engine.collect.market", "ERROR"):
            rec = collect("news", db, at(7, 0), None)
        self.assertIn("boom", rec["fatal"])
        self.assertEqual(len(layout.read_jsonl(db / "days" / DAY.isoformat() / "news.jsonl")), 2)   # still merged
        self.assertEqual(layout.read_jsonl(next((db / "runs").glob("*.jsonl")))[0]["fatal"], rec["fatal"])


class TestInstruments(unittest.TestCase):
    def test_resolve_from_master_once_per_day(self):
        from nifty_engine.calendar import now_ist
        from nifty_engine.data import instruments
        today = now_ist().date()
        e1, e2 = today + dt.timedelta(days=2), today + dt.timedelta(days=9)
        master = pd.DataFrame([
            {"segment": "NSE_FO", "instrument_type": "FUT", "underlying_symbol": "NIFTY", "expiry_date": e2,
             "instrument_key": FUT, "trading_symbol": "NIFTY FUT", "strike_price": 0},
            {"segment": "NSE_FO", "instrument_type": "CE", "underlying_symbol": "NIFTY", "expiry_date": e1,
             "instrument_key": "NSE_FO|1", "trading_symbol": "x", "strike_price": 24500},
            {"segment": "NSE_FO", "instrument_type": "PE", "underlying_symbol": "NIFTY", "expiry_date": e2,
             "instrument_key": "NSE_FO|2", "trading_symbol": "y", "strike_price": 24500},
            {"segment": "NSE_EQ", "instrument_type": "EQ", "underlying_symbol": "", "expiry_date": None,
             "instrument_key": EQ["M&M"], "trading_symbol": "M&M", "strike_price": 0}])
        out = Path(tempfile.mkdtemp())
        with mock.patch.object(instruments, "load_master", return_value=master) as lm:
            ctx = market.Context(out, up=None, clock=now_ist, pause=0)
            m = market.resolve_instruments(ctx)
            market.resolve_instruments(market.Context(out, up=None, clock=now_ist, pause=0))
        self.assertEqual(lm.call_count, 1)                            # cached in meta/instruments.json
        self.assertEqual(m["future"]["key"], FUT)
        self.assertEqual(m["equities"], {"M&M": EQ["M&M"]})
        self.assertEqual(m["expiries"], [e1.isoformat(), e2.isoformat()])
        self.assertEqual(market.candle_targets(m)["mm"], EQ["M&M"])


if __name__ == "__main__":
    unittest.main()
