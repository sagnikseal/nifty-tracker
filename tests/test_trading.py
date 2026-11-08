"""Trading tests: intraday features have no look-ahead, the ATM rules behave exactly as specified,
the real-price replay fills conservatively, news parsing/polling, the news-impact study, agent
guardrails, and the live runtime + API end to end (paper only, no network)."""

import datetime as dt
import json
import math
import os
import sys
import tempfile
import threading
import time
import unittest
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

TMP = tempfile.mkdtemp(prefix="ne_v6_")
os.environ.setdefault("NE_DATA_DIR", TMP)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nifty_engine.calendar import IST  # noqa: E402


def synth_5m(days=30, seed=1, start="2024-01-01"):
    rng = np.random.default_rng(seed)
    out = []
    px = 22000.0
    for d in pd.bdate_range(start, periods=days):
        idx = pd.date_range(d + pd.Timedelta(hours=9, minutes=15), periods=75, freq="5min", tz=IST)
        r = rng.normal(0, 0.0012, len(idx))
        c = px * np.exp(np.cumsum(r))
        o = np.r_[px * (1 + rng.normal(0, 0.002)), c[:-1]]
        h, lo = np.maximum(o, c) * 1.0004, np.minimum(o, c) * 0.9996
        out.append(pd.DataFrame({"open": o, "high": h, "low": lo, "close": c}, index=idx))
        px = c[-1]
    return pd.concat(out)


class TestIntradayFeatures(unittest.TestCase):
    def test_no_lookahead_and_labels(self):
        from nifty_engine.intraday_model import build_frame, feature_columns
        n5 = synth_5m(12)
        X = build_frame(n5)
        self.assertGreater(len(X), 0)
        day = X["date"].iloc[-1]
        T = X[X["date"] == day].index[20]
        # perturb every bar that closes after T: features at T must not change, labels must
        future = n5.index + pd.Timedelta(minutes=5) > T
        n5b = n5.copy()
        n5b.loc[future, ["open", "high", "low", "close"]] *= 1.03
        Xb = build_frame(n5b)
        cols = feature_columns(X)
        a, b = X.loc[T, cols].astype(float), Xb.loc[T, cols].astype(float)
        diff = [c for c in cols if not (np.isclose(a[c], b[c], equal_nan=True))]
        self.assertEqual(diff, [], f"features changed by future data: {diff}")
        self.assertNotAlmostEqual(float(X.loc[T, "y30_ret"]), float(Xb.loc[T, "y30_ret"]))
        # label = close of the bar ending at T+30 vs close at T
        g = n5[n5.index.date == T.date()]
        k = list(g.index).index(T - pd.Timedelta(minutes=5))
        self.assertAlmostEqual(float(X.loc[T, "y30_ret"]), math.log(g["close"].iloc[k + 6] / g["close"].iloc[k]), places=10)

    def test_last_only_matches_full(self):
        from nifty_engine.intraday_model import build_frame, feature_columns
        n5 = synth_5m(8)
        full = build_frame(n5)
        last_day = n5.index[-1].date()
        cut = n5[(n5.index.date < last_day) | (n5.index.time <= dt.time(11, 0))]
        live = build_frame(cut, last_only=True)
        T = live.index[-1]
        self.assertEqual(T.time(), dt.time(11, 5))
        cols = [c for c in feature_columns(full) if c in live.columns]
        np.testing.assert_allclose(live.loc[T, cols].astype(float).values, full.loc[T, cols].astype(float).values, rtol=1e-9)

    def test_walk_forward_runs_and_is_out_of_sample(self):
        from nifty_engine.intraday_model import build_frame, evaluate, walk_forward
        X = build_frame(synth_5m(60, seed=3))
        oos = walk_forward(X, "y30", members=("xgb",), warmup_days=30, refit_days=10)
        self.assertGreater(len(oos), 0)
        self.assertTrue((oos["date"] >= sorted(X["date"].unique())[30]).all())
        r = evaluate(oos)
        self.assertIn("p_value", r)
        self.assertTrue(0.3 < r["accuracy"] < 0.7)      # random walk: no edge


class TestATMRules(unittest.TestCase):
    def setUp(self):
        from nifty_engine.strategies.atm_rules import ATMRules, PositionState
        self.R = ATMRules(lot_size=65)
        self.t0 = dt.datetime(2026, 11, 9, 10, 0, tzinfo=IST)
        self.st = PositionState(1, 100.0, 65, self.t0, 23800.0)

    def test_stop_at_2000(self):
        from nifty_engine.strategies.atm_rules import on_price
        self.assertIsNone(on_price(self.R, self.st, 100 - 1999 / 65, self.t0, 0.5))
        self.assertEqual(on_price(self.R, self.st, 100 - 2000 / 65, self.t0, 0.5), "stop")

    def test_target_exits_unless_80pct(self):
        from nifty_engine.strategies.atm_rules import PositionState, on_price
        self.assertEqual(on_price(self.R, self.st, 100 + 5000 / 65, self.t0, 0.79), "target")
        st = PositionState(1, 100.0, 65, self.t0, 23800.0)
        self.assertIsNone(on_price(self.R, st, 100 + 5000 / 65, self.t0, 0.81))           # extend 30 min
        self.assertEqual(st.extend_until, self.t0 + dt.timedelta(minutes=30))
        t1 = self.t0 + dt.timedelta(minutes=30)
        self.assertIsNone(on_price(self.R, st, 180, t1, 0.85))                             # extend again
        self.assertEqual(st.extensions, 2)
        self.assertEqual(on_price(self.R, st, 180, t1 + dt.timedelta(minutes=30), 0.6), "extension_end")

    def test_put_uses_down_probability(self):
        from nifty_engine.strategies.atm_rules import PositionState, on_price
        st = PositionState(-1, 100.0, 65, self.t0, 23800.0)
        self.assertIsNone(on_price(self.R, st, 100 + 5000 / 65, self.t0, 0.15))           # P(down)=0.85 -> hold
        st2 = PositionState(-1, 100.0, 65, self.t0, 23800.0)
        self.assertEqual(on_price(self.R, st2, 100 + 5000 / 65, self.t0, 0.85), "target")

    def test_two_closes_against_only_in_profit(self):
        from nifty_engine.strategies.atm_rules import on_bar_close
        self.assertIsNone(on_bar_close(self.R, self.st, 23790, 95.0, self.t0))   # against 1, in loss
        self.assertIsNone(on_bar_close(self.R, self.st, 23780, 95.0, self.t0))   # against 2 but in loss
        self.assertEqual(on_bar_close(self.R, self.st, 23770, 101.0, self.t0), "against")
        self.st.against_count = 0
        self.assertIsNone(on_bar_close(self.R, self.st, 23760, 101.0, self.t0))
        self.assertIsNone(on_bar_close(self.R, self.st, 23765, 101.0, self.t0))  # with the trade: reset
        self.assertEqual(self.st.against_count, 0)

    def test_time_exit_and_entry_window(self):
        from nifty_engine.strategies.atm_rules import on_price
        self.assertEqual(on_price(self.R, self.st, 100, self.t0.replace(hour=15, minute=15), 0.5), "time")
        self.assertEqual(self.R.entry_direction(0.61, self.t0, 0), 1)
        self.assertEqual(self.R.entry_direction(0.39, self.t0, 0), -1)
        self.assertEqual(self.R.entry_direction(0.55, self.t0, 0), 0)
        self.assertEqual(self.R.entry_direction(0.7, self.t0, 1), 0)                        # 1 trade per day
        self.assertEqual(self.R.entry_direction(0.7, self.t0.replace(hour=13, minute=45), 0), 0)

    def test_news_reaction_needs_validation(self):
        from nifty_engine.strategies.atm_rules import ATMRules, on_headline
        R = ATMRules(news_exit_bps=8.0)
        self.assertIsNone(on_headline(R, self.st, -12.0, validated=False))
        self.assertEqual(on_headline(R, self.st, -12.0, validated=True), "news")
        self.assertIsNone(on_headline(R, self.st, 12.0, validated=True))


class TestATMReplay(unittest.TestCase):
    def test_stop_fill_is_conservative(self):
        from nifty_engine.config import EngineConfig
        from nifty_engine.paper.atm_replay import replay_day
        from nifty_engine.paper.sim import rates_from
        from nifty_engine.strategies.atm_rules import ATMRules
        cfg = EngineConfig()
        d = dt.date(2026, 11, 9)
        T = pd.Timestamp("2026-11-09 09:20", tz=IST)
        preds = pd.DataFrame({"p_close": [0.7], "p30": [0.5]}, index=[T])
        spot5 = pd.Series([23800.0], index=[T - pd.Timedelta(minutes=5)])
        mins = pd.date_range(T, periods=30, freq="1min")
        px = np.r_[np.full(10, 100.0), np.full(20, 40.0)]          # crash through the stop
        opt = pd.DataFrame({"ts": mins, "open": px, "high": px + 0.5, "low": px - 0.5, "close": px,
                            "strike": 23800.0, "type": "CE"})
        res = replay_day(ATMRules(), cfg, d, preds, spot5, opt, dt.date(2026, 11, 10), rates_from(cfg))
        self.assertEqual(res["exit_reason"], "stop")
        self.assertLess(res["net_pnl"], -2000)                     # gap through the stop costs more
        self.assertEqual(res["type"], "CE")


class TestNewsSources(unittest.TestCase):
    RSS = b"""<?xml version="1.0"?><rss version="2.0"><channel><title>x</title>
      <item><title>RBI cuts repo rate</title><link>https://a/1</link><pubDate>Fri, 06 Nov 2026 19:10:00</pubDate></item>
      <item><title>SEBI order</title><link>https://a/2</link><pubDate>06 Nov, 2026 +0530</pubDate></item>
    </channel></rss>"""
    ATOM = b"""<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"><entry><title>Fed holds rates</title>
      <link href="https://f/1"/><updated>2026-11-05T16:00:00Z</updated></entry></feed>"""

    def test_parse_rss_atom_and_odd_dates(self):
        from nifty_engine.data.news_sources import parse_feed
        a = parse_feed(self.RSS, "rbi")
        self.assertEqual([i["title"] for i in a], ["RBI cuts repo rate", "SEBI order"])
        self.assertEqual(a[0]["ts"].utcoffset(), dt.timedelta(hours=5, minutes=30))     # RBI: no zone -> IST
        self.assertEqual(a[1]["ts"].date(), dt.date(2026, 11, 6))                        # SEBI odd format
        b = parse_feed(self.ATOM, "fed")
        self.assertEqual(b[0]["url"], "https://f/1")
        self.assertEqual(b[0]["ts"].hour, 21)                                            # 16:00Z = 21:30 IST

    def test_conditional_get_and_backoff(self):
        from nifty_engine.data.news_sources import NewsPoller

        class R:
            def __init__(self, code, content=b"", headers=None):
                self.status_code, self.content, self.headers = code, content, headers or {}
                self.ok = code < 400

            def raise_for_status(self):
                if self.status_code >= 400:
                    raise RuntimeError(self.status_code)

        class S:
            headers = {}

            def __init__(self):
                self.calls = []

            def get(self, url, headers=None, timeout=None):
                self.calls.append(headers or {})
                if "bad" in url:
                    return R(403)
                if len([c for c in self.calls if c is not None]) == 1:
                    return R(200, TestNewsSources.RSS, {"ETag": "abc"})
                return R(304)

        s = S()
        p = NewsPoller([{"name": "good", "url": "http://good", "category": "x", "poll_s": 60},
                        {"name": "bad", "url": "http://bad", "category": "x", "poll_s": 60}], session=s)
        items = p.poll(force=True, max_workers=1)
        self.assertEqual(len(items), 2)
        self.assertFalse(p.states["bad"].ok)
        self.assertGreater(p.states["bad"].next_due - time.time(), 100)            # backed off (2 x 60 s)
        items2 = p.poll(force=True, max_workers=1)
        self.assertEqual(items2, [])
        self.assertTrue(any(c.get("If-None-Match") == "abc" for c in s.calls))
        self.assertTrue(p.states["good"].ok)


class TestNewsImpact(unittest.TestCase):
    def test_study_finds_injected_effect_and_features_use_past_only(self):
        from nifty_engine.config import EngineConfig
        from nifty_engine.semantic import news_impact as ni
        rng = np.random.default_rng(5)
        days = pd.bdate_range("2026-05-01", periods=60)
        bars, heads = [], []
        for d in days:
            idx = pd.date_range(d + pd.Timedelta(hours=9, minutes=15), periods=375, freq="1min", tz=IST)
            r = rng.normal(0, 0.0003, len(idx))
            for k in rng.choice(np.arange(30, 300), 6, replace=False):
                sgn = rng.choice([-1, 1])
                r[k + 1:k + 31] += sgn * 0.0002                                    # 30-min drift after news
                heads.append({"id": f"{d.date()}-{k}", "title": f"h{k}", "source": "t",
                              "first_seen": idx[k] + pd.Timedelta(seconds=20), "india_relevance": 0.9,
                              "nifty_direction": float(sgn), "magnitude": 0.6, "surprise": 0.6, "confidence": 0.7,
                              "channel": "rates", "horizon": "intraday", "scorer": "test"})
            c = 23000 * np.exp(np.cumsum(r))
            bars.append(pd.DataFrame({"open": c, "high": c, "low": c, "close": c}, index=idx))
        B, S = pd.concat(bars), pd.DataFrame(heads)
        ev = ni.build_events(EngineConfig(), S, B)
        self.assertEqual(len(ev), len(S))
        self.assertTrue((ev["fwd30_bps"] * np.sign(ev["nifty_direction"]) > 0).mean() > 0.7)
        out = ni.study(EngineConfig(), ev)
        self.assertGreater(out["oos"]["ic"], 0.3)
        self.assertEqual(out["status"], "PASS")
        m = ni.load_model()
        self.assertTrue(m.validated)
        self.assertGreater(m.predict_bps({"nifty_direction": 1, "magnitude": 0.6, "india_relevance": 0.9,
                                          "surprise": 0.6, "confidence": 0.7, "channel": "rates"}), 0)
        fn = ni.NewsFeatureFn(S, m)
        t = S["first_seen"].iloc[10]
        before = fn((t - pd.Timedelta(seconds=1)).to_pydatetime())
        after = fn((t + pd.Timedelta(seconds=1)).to_pydatetime())
        self.assertNotEqual(before.get("nw_n_30m"), after.get("nw_n_30m"))   # a headline counts only once seen


class TestAgentGuardrails(unittest.TestCase):
    def test_guardrails(self):
        from nifty_engine.agent.llm_agent import TradingAgent
        from nifty_engine.config import EngineConfig
        from nifty_engine.strategies.atm_rules import ATMRules
        ag = TradingAgent.__new__(TradingAgent)
        ag.rules = ATMRules()
        base = {"position": None, "entry_window_open": True, "trades_today": 0}
        d = ag.apply_guardrails({"action": "ENTER_LONG", "confidence": 0.5, "p_up_30m": 0.6, "guardrail": None}, base)
        self.assertEqual((d["action"], d["guardrail"]), ("STAND_ASIDE", "confidence below 0.60"))
        d = ag.apply_guardrails({"action": "ENTER_LONG", "confidence": 0.7, "p_up_30m": 0.4, "guardrail": None}, base)
        self.assertEqual(d["action"], "STAND_ASIDE")
        d = ag.apply_guardrails({"action": "ENTER_SHORT", "confidence": 0.7, "p_up_30m": 0.3, "guardrail": None},
                                dict(base, entry_window_open=False))
        self.assertEqual(d["guardrail"], "outside the entry window")
        d = ag.apply_guardrails({"action": "ENTER_SHORT", "confidence": 0.7, "p_up_30m": 0.3, "guardrail": None}, base)
        self.assertEqual(d["action"], "ENTER_SHORT")
        d = ag.apply_guardrails({"action": "EXIT", "confidence": 0.7, "p_up_30m": 0.3, "guardrail": None}, base)
        self.assertEqual(d["action"], "STAND_ASIDE")                                 # nothing to exit
        self.assertIn("Rs 2,000", __import__("nifty_engine.agent.prompts", fromlist=["x"]).agent_system(ATMRules()))
        self.assertTrue(EngineConfig().atm_stop_inr == 2000)


class TestRuntimeAndAPI(unittest.TestCase):
    def test_manual_paper_trade_and_api(self):
        from nifty_engine import live
        from nifty_engine.api import create_app
        from nifty_engine.config import EngineConfig
        from nifty_engine.live import runtime as rtm
        from nifty_engine.paper.ledger import Ledger
        cfg = EngineConfig()
        cfg.agent_enabled = False
        led = Ledger(Path(TMP) / "trading_ledger.sqlite")
        rt = rtm.Runtime(cfg, led, start_threads=False)
        rt.upstox.token = ""                                     # never touch the network
        orig = rtm.market_status
        rtm.market_status = lambda now: "OPEN"
        try:
            rt.quote = {"spot": 23810.0, "vix": 14.0, "source": "test", "fetched": time.time()}
            r = rt.manual_order("buy_ce")
            self.assertTrue(r["ok"], r)
            snap = rt.snapshot()
            pos = snap["positions"][0]
            self.assertEqual((pos["side"], pos["strike"], pos["qty"]), ("CE", 23800.0, 65))
            self.assertEqual(pos["price_source"], "model_vix")
            self.assertFalse(rt.manual_order("buy_pe")["ok"])    # one position per book
            # stop: crash the spot so the modelled option loses > Rs 2,000
            rt.quote["spot"] = 23400.0
            rt.manage_positions(dt.datetime.now(IST).replace(hour=11, minute=0))
            tr = led.trades()
            self.assertEqual(tr.iloc[-1]["exit_reason"], "stop")
            self.assertLessEqual(tr.iloc[-1]["net_pnl"], -2000)
            # API end to end on a real local server
            import uvicorn
            app = create_app(rt)
            server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=18917, log_level="error"))
            th = threading.Thread(target=server.run, daemon=True)
            th.start()
            for _ in range(50):
                if server.started:
                    break
                time.sleep(0.1)
            get = lambda p: json.loads(urllib.request.urlopen(f"http://127.0.0.1:18917{p}", timeout=5).read())  # noqa: E731
            st = get("/api/state")
            self.assertIn("quote", st)
            self.assertEqual(st["rules"]["target_inr"], 5000)
            self.assertEqual(get("/api/trades")[0]["exit_reason"], "stop")
            self.assertIn("manual", get("/api/equity"))
            req = urllib.request.Request("http://127.0.0.1:18917/api/auto", method="POST",
                                        data=json.dumps({"book": "agent", "enabled": False}).encode(),
                                        headers={"Content-Type": "application/json"})
            self.assertTrue(json.loads(urllib.request.urlopen(req, timeout=5).read())["ok"])
            html = urllib.request.urlopen("http://127.0.0.1:18917/", timeout=5).read().decode()
            self.assertIn("/static/app.js", html)
            md = urllib.request.urlopen("http://127.0.0.1:18917/api/report/intraday", timeout=5).read().decode()
            self.assertTrue(md.startswith("#"))
            server.should_exit = True
            th.join(timeout=5)
        finally:
            rtm.market_status = orig
        self.assertIsNotNone(live)


if __name__ == "__main__":
    unittest.main()
