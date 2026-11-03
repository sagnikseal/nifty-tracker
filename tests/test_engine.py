"""Core tests (stdlib unittest; run: python -m unittest discover -s tests -v).

The most important ones guard against the failure mode that broke the legacy v3 model:
information from the future leaking into features or training."""

from __future__ import annotations

import datetime as dt
import os
import tempfile
import unittest

import numpy as np
import pandas as pd

os.environ.setdefault("NE_DATA_DIR", tempfile.mkdtemp(prefix="ne_test_"))

from nifty_engine.calendar import next_weekly_expiry  # noqa: E402
from nifty_engine.config import EngineConfig  # noqa: E402
from nifty_engine.features import build_daily_features, feature_columns  # noqa: E402
from nifty_engine.model import ModelSpec, evaluate, walk_forward  # noqa: E402
from nifty_engine.options import bs_price, build_structure  # noqa: E402
from nifty_engine.paper.charges import ChargeRates, order_charges  # noqa: E402
from nifty_engine.paper.replay import replay  # noqa: E402


def make_market(n=1600, seed=0, independent=True):
    """Random-walk market where nothing is predictable (open->close independent of the past)."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2014-01-01", periods=n)
    gaps, intraday = rng.normal(0, 0.004, n), rng.normal(0, 0.008, n)
    c = np.empty(n)
    o = np.empty(n)
    prev = 20000.0
    for i in range(n):
        o[i] = prev * np.exp(gaps[i])
        c[i] = o[i] * np.exp(intraday[i])
        prev = c[i]
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.003, n)))
    lo = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.003, n)))
    nifty = pd.DataFrame(dict(open=o, high=h, low=lo, close=c, volume=0.0), index=idx)
    vix = pd.DataFrame(dict(open=15.0, high=16.0, low=14.0, close=15 + rng.normal(0, 1, n)), index=idx)

    def asset(scale):
        cc = scale * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
        oo = np.r_[cc[0], cc[:-1]] * np.exp(rng.normal(0, 0.003, n))
        return pd.DataFrame(dict(open=oo, high=np.maximum(oo, cc), low=np.minimum(oo, cc), close=cc), index=idx)

    return {"nifty": nifty, "indiavix": vix, "spx": asset(4000), "n225": asset(30000), "brent": asset(80)}


class TestLeakage(unittest.TestCase):
    def test_features_do_not_see_the_future(self):
        raw = make_market()
        F1 = build_daily_features(raw, "oc")
        cut = raw["nifty"].index[1000]
        raw2 = {k: v.copy() for k, v in raw.items()}
        for k, v in raw2.items():
            m = v.index > cut
            for col in ("open", "high", "low", "close"):
                v.loc[m, col] = v.loc[m, col] * 1.37  # wreck everything after `cut`
        # also wreck NIFTY's own high/low/close on `cut` itself (unknown at 09:20 that day)
        for col in ("high", "low", "close"):
            raw2["nifty"].loc[cut, col] *= 0.71
        raw2["indiavix"].loc[cut, "close"] *= 0.5
        raw2["spx"].loc[cut, "close"] *= 0.5  # US session of `cut` closes after NIFTY's decision
        F2 = build_daily_features(raw2, "oc")
        cols = feature_columns(F1)
        a = F1.loc[:cut, cols]
        b = F2.loc[:cut, cols]
        pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-9)

    def test_no_skill_on_unpredictable_market(self):
        raw = make_market(n=1500, seed=3)
        F = build_daily_features(raw, "oc")
        cfg = EngineConfig()
        cfg.min_train_days = 500
        oos = walk_forward(F, cfg, ModelSpec("logit_only", use_gbm=False))
        m = evaluate(oos)
        self.assertLess(m["accuracy"], 0.56, m)          # nothing to find -> ~50%
        self.assertGreater(m["p_value_vs_coinflip"], 0.001)


class TestOptionsAndCosts(unittest.TestCase):
    def test_put_call_parity(self):
        S, K, T, iv = 23000, 23000, 7 / 365, 0.14
        c, p = bs_price(S, K, T, iv, "CE"), bs_price(S, K, T, iv, "PE")
        self.assertAlmostEqual(c - p, S - K * np.exp(-0.065 * T), places=4)

    def test_spread_value_bounded_by_width(self):
        st = build_structure(1, 23010, "debit_spread", 150)
        v = st.value(25000, 1 / 365, 14)
        self.assertLessEqual(v, 150.0 + 1e-6)

    def test_option_charges(self):
        # Sell 1 lot (65) at Rs 100 premium = Rs 6,500 turnover
        ch = order_charges("option", "SELL", 6500, ChargeRates())
        self.assertAlmostEqual(ch["stt"], 9.75, places=2)          # 0.15%
        self.assertAlmostEqual(ch["brokerage"], 16.25, places=2)  # min(20, 0.25%)

    def test_tuesday_expiry_after_sept_2025(self):
        self.assertEqual(next_weekly_expiry(dt.date(2026, 11, 3), 1).weekday(), 1)
        self.assertEqual(next_weekly_expiry(dt.date(2024, 6, 3)).weekday(), 3)


class TestReplay(unittest.TestCase):
    def test_replay_runs_and_accounts_costs(self):
        raw = make_market(n=1300, seed=5)
        F = build_daily_features(raw, "oc")
        cfg = EngineConfig()
        cfg.min_train_days = 500
        cfg.prob_threshold = 0.5
        oos = walk_forward(F, cfg, ModelSpec("logit_only", use_gbm=False))
        res = replay(F, oos, cfg, "options")
        s = res["summary"]
        self.assertGreater(s["trades"], 50)
        self.assertGreater(s["total_charges"], 0)
        # With no edge, costs must make the strategy lose money on average
        self.assertLess(res["trades"]["net_pnl"].mean(), res["trades"]["gross_pnl"].mean())


if __name__ == "__main__":
    unittest.main()


class TestLivePaperCycle(unittest.TestCase):
    def test_signal_enter_mark_exit(self):
        from nifty_engine.data import store
        from nifty_engine.model import train_and_register
        from nifty_engine.paper.ledger import Ledger
        from nifty_engine.paper.live import LivePaperTrader

        raw = make_market(n=1100, seed=9)
        for k, v in raw.items():
            store.save(k, v)
        cfg = EngineConfig()
        cfg.min_train_days = 500
        cfg.prob_threshold = 0.5   # force a trade
        F = build_daily_features(raw, "oc")
        train_and_register(F, cfg, [ModelSpec("logit_only", use_gbm=False)])
        led = Ledger(os.path.join(os.environ["NE_DATA_DIR"], "test_ledger.sqlite"))
        tr = LivePaperTrader(cfg, led)
        tr.upstox.token = ""
        os.environ.pop("UPSTOX_TOTP_SECRET", None)
        prices = {"spot": 23000.0, "vix": 14.0, "source": "test"}
        tr.spot_and_vix = lambda: prices
        tr.today_open = lambda: 23000.0
        sig = tr.morning_signal()
        self.assertIn("p_up", sig)
        tid = tr.enter(sig, allow_no_edge=True)
        self.assertIsNotNone(tid)
        self.assertEqual(len(led.open_trades()), 1)
        prices["spot"] = 23000.0 + 120 * sig["direction"]   # move in our favour
        tr.mark_and_manage(force_exit=True)
        t = led.trades().iloc[-1]
        self.assertEqual(t["status"], "CLOSED")
        self.assertGreater(t["charges"], 0)
        self.assertGreater(t["gross_pnl"], 0)
        self.assertEqual(t["price_source"], "model_vix")


class TestScorecard(unittest.TestCase):
    def test_scorecard_counts_all_days(self):
        from nifty_engine.paper.ledger import Ledger
        from nifty_engine.paper.live import LivePaperTrader
        led = Ledger(os.path.join(os.environ["NE_DATA_DIR"], "score.sqlite"))
        led.kv_set("outcome_2026-11-02", {"up": 1, "p_up": 0.6, "direction": 1})
        led.kv_set("outcome_2026-11-03", {"up": 0, "p_up": 0.52, "direction": 0})
        sc = LivePaperTrader(EngineConfig(), led).prediction_scorecard()
        self.assertEqual(sc["days"], 2)
        self.assertAlmostEqual(sc["accuracy_all_days"], 0.5)
        self.assertEqual(sc["accuracy_confident"], 1.0)
