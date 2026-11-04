"""v5 tests: causality of every new feature, option math, replay, news timing, strategy books,
the recorder and the research harness. Offline (no network, no Ollama needed)."""

from __future__ import annotations

import datetime as dt
import json
import os
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

os.environ.setdefault("NE_DATA_DIR", tempfile.mkdtemp(prefix="ne_test_"))

from nifty_engine.calendar import IST  # noqa: E402
from nifty_engine.config import DATA_DIR, EngineConfig  # noqa: E402


def rand_ohlc(n=600, seed=1, freq="B", start="2022-01-03", vol=0.006, index=None):
    rng = np.random.default_rng(seed)
    idx = index if index is not None else pd.date_range(start, periods=n, freq=freq)
    n = len(idx)
    c = 20000 * np.exp(np.cumsum(rng.normal(0, vol, n)))
    o = np.r_[c[0], c[:-1]] * np.exp(rng.normal(0, vol / 3, n))
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, vol / 2, n)))
    lo = np.minimum(o, c) * (1 - np.abs(rng.normal(0, vol / 2, n)))
    return pd.DataFrame({"open": o, "high": h, "low": lo, "close": c, "volume": rng.integers(1, 1000, n).astype(float)}, index=idx)


def intraday_bars(days=6, seed=3):
    idx = []
    for d in pd.bdate_range("2026-10-05", periods=days):
        idx += list(pd.date_range(pd.Timestamp(dt.datetime.combine(d.date(), dt.time(9, 15)), tz=IST), periods=75, freq="5min"))
    return rand_ohlc(seed=seed, index=pd.DatetimeIndex(idx), vol=0.0012)


def wreck_after(df, cut_pos, factor=1.31):
    d2 = df.copy()
    d2.iloc[cut_pos + 1:, :4] = d2.iloc[cut_pos + 1:, :4].to_numpy() * factor
    return d2


def assert_same_until(test, A, B, pos):
    a, b = A.iloc[:pos + 1], B.iloc[:pos + 1]
    pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-9, atol=1e-9)


class TestSMC(unittest.TestCase):
    def test_smc_features_are_causal(self):
        from nifty_engine.analytics.smc import smc_features
        df = rand_ohlc(500, seed=2)
        cut = 350
        assert_same_until(self, smc_features(df), smc_features(wreck_after(df, cut)), cut)

    def test_swing_confirmed_only_k_bars_later(self):
        from nifty_engine.analytics.smc import confirmed_swings
        h = np.array([1, 2, 3, 9, 3, 2, 1, 1, 1], float)
        sh, sh_i, _, _ = confirmed_swings(h, h - 0.5, k=3)
        self.assertTrue(np.isnan(sh[:6]).all())          # not known before bar 3 + 3
        self.assertEqual(sh[6], 9.0)
        self.assertEqual(sh_i[6], 3)

    def test_fvg_and_bos(self):
        from nifty_engine.analytics.smc import smc_features
        # a swing high at bar 2, a pullback, then a bullish displacement that breaks it and leaves a gap
        c = [100, 101, 102, 101, 100, 101, 100, 99, 100, 102, 106, 111, 112, 113]
        o = [100] + c[:-1]
        h = [max(a, b) + 0.5 for a, b in zip(o, c)]
        lo = [min(a, b) - 0.5 for a, b in zip(o, c)]
        df = pd.DataFrame({"open": o, "high": h, "low": lo, "close": c}, index=pd.bdate_range("2024-01-01", periods=len(o)))
        S = smc_features(df, k=1)
        self.assertGreaterEqual(S["fvg_new"].max(), 1)
        self.assertGreaterEqual(S["bos"].max() + S["choch"].max(), 1)
        self.assertGreater(S["n_bull_fvg"].iloc[-1], 0)

    def test_setups_are_causal(self):
        from nifty_engine.analytics.smc import smc_setups
        df = intraday_bars(8)
        sess = df.index.to_series().dt.date
        cut = 400
        a = smc_setups(df, session=sess, k=2)
        b = smc_setups(wreck_after(df, cut), session=sess, k=2)
        lim = df.index[cut]
        pd.testing.assert_frame_equal(a[a["ts_entry"] <= lim].reset_index(drop=True),
                                      b[b["ts_entry"] <= lim].reset_index(drop=True))


class TestTechnicalAndRegime(unittest.TestCase):
    def test_intraday_frame_is_causal(self):
        from nifty_engine.analytics.technical import intraday_technical_frame
        df = intraday_bars(5)
        cut = 222
        assert_same_until(self, intraday_technical_frame(df), intraday_technical_frame(wreck_after(df, cut)), cut)

    def test_opening_range_unknown_until_complete(self):
        from nifty_engine.analytics.technical import opening_range
        df = intraday_bars(2)
        o = opening_range(df, minutes=15)
        first = o[df.index.date == df.index[0].date()]
        self.assertTrue(first["orb_hi"].iloc[:3].isna().all())   # 09:15, 09:20, 09:25 bars
        self.assertFalse(np.isnan(first["orb_hi"].iloc[3]))

    def test_regime_is_causal(self):
        from nifty_engine.analytics.regime import regime_features
        df = rand_ohlc(900, seed=4)
        vix = pd.Series(15 + np.random.default_rng(1).normal(0, 1, len(df)), index=df.index)
        cut = 700
        A = regime_features(df, vix)
        d2 = wreck_after(df, cut - 1)              # lag=1: row t may use bar t-1, never bar t
        v2 = vix.copy()
        v2.iloc[cut:] *= 2
        B = regime_features(d2, v2)
        assert_same_until(self, A, B, cut)


class TestDailyGroupsLeakage(unittest.TestCase):
    def test_all_long_history_groups_are_causal(self):
        from nifty_engine.features import build_daily_features, feature_columns
        from test_engine import make_market
        raw = make_market(n=1300, seed=11)
        idx = raw["nifty"].index
        for name in ("kospi", "twii", "shanghai", "hdb", "hdfcbank", "inda", "usdinr"):
            raw[name] = rand_ohlc(index=idx, seed=hash(name) % 100)
        groups = ["base", "asia_extra", "adr", "smc", "structure", "regime", "events"]
        F1 = build_daily_features(raw, "oc", groups=groups)
        cut = idx[1000]
        raw2 = {k: v.copy() for k, v in raw.items()}
        for k, v in raw2.items():
            m = v.index > cut
            v.loc[m, ["open", "high", "low", "close"]] *= 1.37
        for col in ("high", "low", "close"):
            raw2["nifty"].loc[cut, col] *= 0.71               # unknown at 09:20 on `cut`
        for k in ("indiavix", "spx", "hdb", "hdfcbank", "inda", "usdinr", "kospi", "twii", "shanghai"):
            raw2[k].loc[cut, "close"] *= 0.5                   # closes after the decision
        F2 = build_daily_features(raw2, "oc", groups=groups)
        cols = feature_columns(F1)
        self.assertGreater(len(cols), 80)
        pd.testing.assert_frame_equal(F1.loc[:cut, cols], F2.loc[:cut, cols], check_exact=False, rtol=1e-9)


class TestOvernightFutures(unittest.TestCase):
    def test_move_from_nifty_close_to_0900(self):
        from nifty_engine.data import global_markets as gm
        idx = pd.date_range(pd.Timestamp("2026-10-05 00:30", tz=IST), periods=24 * 6, freq="60min").as_unit("us")
        px = pd.Series(np.arange(len(idx), dtype=float) + 5000.0, index=idx)
        gm.RAW_1H.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"open": px, "high": px, "low": px, "close": px, "volume": 1.0}, index=idx).to_csv(gm.RAW_1H / "es.csv")
        days = pd.DatetimeIndex(["2026-10-06", "2026-10-07"])
        X = gm.overnight_features(days, names=["es"])
        # bars start on the half hour: the last one finished by 15:30 on 6 Oct starts 14:30; the last one
        # finished by 09:00 on 7 Oct starts 07:30 (the 08:30 bar is still open at 09:00)
        p0 = px[pd.Timestamp("2026-10-06 14:30", tz=IST)]
        p1 = px[pd.Timestamp("2026-10-07 07:30", tz=IST)]
        self.assertAlmostEqual(X.loc["2026-10-07", "on_es"], np.log(p1 / p0))
        (gm.RAW_1H / "es.csv").unlink()


class TestOptionsMicro(unittest.TestCase):
    def test_implied_vol_round_trip(self):
        from nifty_engine.options import bs_price, implied_vol
        for typ in ("CE", "PE"):
            p = bs_price(24000, 24100, 5 / 365, 0.13, typ)
            self.assertAlmostEqual(implied_vol(p, 24000, 24100, 5 / 365, typ), 0.13, places=3)

    def _chain(self, spot=24000):
        from nifty_engine.options import bs_price
        rows = []
        for k in range(23000, 25050, 50):
            for typ in ("CE", "PE"):
                p = bs_price(spot, k, 5 / 365, 0.14, typ)
                oi = 1000 + (5000 if (typ == "CE" and k == 24500) else 0) + (7000 if (typ == "PE" and k == 23500) else 0)
                rows.append({"strike": k, "type": typ, "ltp": p, "bid": p * 0.99, "ask": p * 1.01, "oi": oi, "volume": 100})
        return pd.DataFrame(rows)

    def test_chain_metrics(self):
        from nifty_engine.analytics.options_micro import chain_metrics
        m = chain_metrics(self._chain(), 24000, 5 / 365)
        self.assertEqual(m["call_wall"], 24500)
        self.assertEqual(m["put_wall"], 23500)
        self.assertAlmostEqual(m["atm_iv"], 0.14, places=2)
        self.assertGreater(m["pcr_oi"], 1.0)
        self.assertTrue(23500 <= m["max_pain"] <= 24500)
        self.assertGreater(m["implied_move_pct"], 0)

    def test_gex_sign_convention(self):
        from nifty_engine.analytics.options_micro import chain_metrics
        ch = self._chain()
        ch.loc[ch["type"] == "PE", "oi"] = 0
        self.assertGreater(chain_metrics(ch, 24000, 5 / 365)["gex_cr"], 0)   # calls only -> dealers long gamma

    def test_buildup(self):
        from nifty_engine.analytics.options_micro import buildup
        self.assertEqual(buildup(5, 100), "long_buildup")
        self.assertEqual(buildup(-5, 100), "short_buildup")
        self.assertEqual(buildup(5, -100), "short_covering")
        self.assertEqual(buildup(-5, -100), "long_unwinding")


class TestReplay(unittest.TestCase):
    def _n5(self, day, drift):
        idx = pd.date_range(pd.Timestamp(dt.datetime.combine(day, dt.time(9, 15)), tz=IST), periods=75, freq="5min")
        c = 24000 + drift * np.arange(75)
        return pd.DataFrame({"open": c - drift / 2, "high": c + 5, "low": c - 5, "close": c}, index=idx)

    def test_modeled_long_call_profits_on_rally(self):
        from nifty_engine.options import Leg
        from nifty_engine.research.option_replay import simulate_modeled
        from nifty_engine.strategies.base import TradePlan
        day = dt.date(2026, 10, 6)
        plan = TradePlan("t", day, dt.time(9, 20), 1, "itm_buy", [Leg("CE", 23950, +1)], dt.date(2026, 10, 13))
        r = simulate_modeled(plan, EngineConfig(), self._n5(day, 4.0), 14.0)
        self.assertEqual(r["exit_reason"], "time")
        self.assertGreater(r["net_pnl"], 0)
        self.assertGreater(r["charges"], 0)

    def test_real_prices_and_credit_stop(self):
        from nifty_engine.data.expired_options import OUT
        from nifty_engine.options import Leg, bs_price
        from nifty_engine.research.option_replay import _CACHE, simulate_real
        from nifty_engine.strategies.base import TradePlan
        day, exp = dt.date(2026, 10, 7), dt.date(2026, 10, 13)
        n5 = self._n5(day, 12.0)              # strong rally: short call spread must stop out
        rows = []
        for k in (24100, 24250):
            for ts, r in n5.iterrows():
                T = (dt.datetime.combine(exp, dt.time(15, 30), tzinfo=IST) - ts.to_pydatetime()).total_seconds() / (365 * 86400)
                o, c = bs_price(r["open"], k, T, 0.13, "CE"), bs_price(r["close"], k, T, 0.13, "CE")
                rows.append({"ts": ts.isoformat(), "open": o, "high": max(o, c), "low": min(o, c), "close": c,
                             "volume": 10, "oi": 100, "strike": float(k), "type": "CE", "key": f"k{k}"})
        OUT.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(OUT / f"{exp.isoformat()}.csv", index=False)
        _CACHE.clear()
        plan = TradePlan("credit_spread", day, dt.time(9, 20), -1, "credit_spread",
                         [Leg("CE", 24100, -1), Leg("CE", 24250, +1)], exp, credit_stop_mult=1.5)
        r = simulate_real(plan, EngineConfig(), n5)
        self.assertEqual(r["price_source"], "real")
        self.assertLess(r["entry_value"], 0)
        self.assertEqual(r["exit_reason"], "stop")
        self.assertLess(r["net_pnl"], 0)
        self.assertGreater(r["max_loss"], 0)


class TestStats(unittest.TestCase):
    def test_deflated_sharpe_falls_with_trials(self):
        from nifty_engine.research.stats import deflated_sharpe
        x = np.random.default_rng(0).normal(0.1, 1, 300)
        self.assertGreater(deflated_sharpe(x, 1)["dsr"], deflated_sharpe(x, 50)["dsr"])

    def test_pbo_and_dm(self):
        from nifty_engine.research.stats import dm_logloss, pbo_cscv
        M = pd.DataFrame(np.random.default_rng(1).normal(0, 1, (40, 6)))
        p = pbo_cscv(M)["pbo"]
        self.assertTrue(0 <= p <= 1)
        y = (np.random.default_rng(2).random(400) > 0.5).astype(int)
        good = np.where(y == 1, 0.6, 0.4)
        self.assertGreater(dm_logloss(np.full(400, 0.5), good, y)["t"], 3)


class TestNews(unittest.TestCase):
    def test_timing_and_contamination(self):
        from nifty_engine.semantic import news_llm as nl
        cfg = EngineConfig()
        cfg.llm_knowledge_cutoff = "2026-10-01"
        t_in = dt.datetime(2026, 10, 6, 8, 30, tzinfo=IST)
        t_late = dt.datetime(2026, 10, 6, 9, 30, tzinfo=IST)
        t_old = dt.datetime(2026, 9, 29, 8, 30, tzinfo=IST)
        nl.store_items([{"title": "Markets crash as war fears grip Nifty", "first_seen": t_in},
                        {"title": "Nifty rally record high as FII inflows surge", "first_seen": t_late},
                        {"title": "RBI rate cut boosts Sensex", "first_seen": t_old}])
        heads = nl.load_headlines()
        old_id = heads[heads["title"].str.startswith("RBI")]["id"].iloc[0]
        nl._append_jsonl(nl.SCORES, [{"id": old_id, "first_seen": t_old.isoformat(), "scorer": "fake-llm",
                                      "lex": nl.lexicon_score("RBI rate cut boosts Sensex"),
                                      "llm": {**nl.lexicon_score("x"), "nifty_direction": -1.0, "india_relevance": 1.0}}])
        S = nl.scored_frame(cfg)
        self.assertEqual(S[S["id"] == old_id]["scorer"].iloc[0], "lexicon")     # pre-cutoff LLM score ignored
        f = nl.window_features(S, *nl.preopen_window(dt.date(2026, 10, 6), dt.date(2026, 10, 5)))
        self.assertEqual(f["nw_n"], 1.0)        # the 09:30 headline is not visible at 09:10
        self.assertLess(f["nw_dir"], 0)

    def test_llm_client_parses_and_fails_safe(self):
        from nifty_engine.semantic.llm import LocalLLM
        self.assertFalse(LocalLLM(url="http://127.0.0.1:9").available())
        resp = mock.Mock()
        resp.raise_for_status = lambda: None
        resp.json = lambda: {"message": {"content": json.dumps({"items": [{"id": "a", "nifty_direction": 0.4}]})}}
        with mock.patch("nifty_engine.semantic.llm.requests.post", return_value=resp):
            out = LocalLLM().json_chat("s", "u", {"type": "object"})
        self.assertEqual(out["items"][0]["id"], "a")


class TestBooks(unittest.TestCase):
    def _ctx(self, **kw):
        from nifty_engine.strategies.base import Context
        base = dict(cfg=EngineConfig(), day=dt.date(2026, 10, 6), now=dt.time(9, 20), spot=24010.0, vix=14.0,
                    expiries=[dt.date(2026, 10, 6), dt.date(2026, 10, 13), dt.date(2026, 10, 20)],
                    p_up=0.6, regime={"vrp_har": 0.002, "vix_pct_1y": 0.8})
        base.update(kw)
        return Context(**base)

    def test_condor_strikes_and_filters(self):
        from nifty_engine.strategies.books import BOOKS
        p = BOOKS["iron_condor"].propose(self._ctx())
        self.assertEqual(len(p.legs), 4)
        shorts = sorted(l.strike for l in p.legs if l.qty_sign < 0)
        self.assertLess(shorts[0], 24010)
        self.assertGreater(shorts[1], 24010)
        self.assertEqual(p.expiry, dt.date(2026, 10, 13))       # never 0DTE
        self.assertIsNone(BOOKS["iron_condor"].propose(self._ctx(regime={"vrp_har": -0.001})))

    def test_0dte_only_on_expiry_day(self):
        from nifty_engine.strategies.books import BOOKS
        self.assertIsNotNone(BOOKS["iron_fly_0dte"].propose(self._ctx(now=dt.time(9, 45))))
        self.assertIsNone(BOOKS["iron_fly_0dte"].propose(self._ctx(day=dt.date(2026, 10, 7))))

    def test_event_and_turn_of_month(self):
        from nifty_engine.strategies.books import BOOKS
        nxt = [d.date() for d in pd.bdate_range("2026-10-07", periods=30)]
        p = BOOKS["event_straddle"].propose(self._ctx(now=dt.time(15, 10), events_next=["rbi_mpc"], next_sessions=nxt))
        self.assertEqual(p.exit_day, nxt[0])
        self.assertEqual(len(p.legs), 2)
        ctx = self._ctx(day=dt.date(2026, 10, 29), sessions_left_in_month=2, now=dt.time(15, 10),
                        expiries=[dt.date(2026, 11, 3), dt.date(2026, 11, 10)],
                        next_sessions=[d.date() for d in pd.bdate_range("2026-10-30", periods=10)])
        t = BOOKS["turn_of_month"].propose(ctx)
        self.assertEqual(t.exit_day, dt.date(2026, 11, 4))      # 3rd session of November
        self.assertGreater(t.expiry, t.exit_day)

    def test_book_runner_cycle(self):
        from nifty_engine.options import Leg
        from nifty_engine.paper.books_live import BookRunner
        from nifty_engine.paper.ledger import Ledger
        from nifty_engine.paper.live import LivePaperTrader
        from nifty_engine.strategies.base import TradePlan
        led = Ledger(os.path.join(os.environ["NE_DATA_DIR"], "books.sqlite"))
        cfg = EngineConfig()
        tr = LivePaperTrader(cfg, led)
        tr.upstox.token = ""
        os.environ.pop("UPSTOX_TOTP_SECRET", None)
        px = {"spot": 24000.0, "vix": 14.0, "source": "test"}
        tr.spot_and_vix = lambda: px
        run = BookRunner(cfg, tr)
        ctx = self._ctx(spot=24000.0, day=dt.date.today())
        plan = TradePlan("credit_spread", dt.date.today(), dt.time(9, 20), 1, "credit_spread",
                         [Leg("PE", 23800, -1), Leg("PE", 23650, +1)], dt.date.today() + dt.timedelta(days=6),
                         credit_stop_mult=1.5, credit_target_pct=60.0)
        tid = run.open(plan, ctx)
        self.assertIsNotNone(tid)
        tr.mark_and_manage(force_exit=True)                 # v4 manager must leave v5 trades alone
        self.assertEqual(len(led.open_trades()), 1)
        run.manage(force_exit=True)
        t = led.trades().iloc[-1]
        self.assertEqual(t["status"], "CLOSED")
        self.assertEqual(t["book"], "credit_spread")
        self.assertGreater(t["charges"], 0)


class TestRecorderAndEvents(unittest.TestCase):
    def test_flatten_feed(self):
        from nifty_engine.data.recorder import flatten_feed
        f = {"fullFeed": {"marketFF": {"ltpc": {"ltp": 101.5, "cp": 99.0},
                                       "marketLevel": {"bidAskQuote": [{"bidQ": 650, "bidP": 101.4, "askQ": 130, "askP": 101.6},
                                                                       {"bidQ": 325, "bidP": 101.3, "askQ": 65, "askP": 101.7}]},
                                       "optionGreeks": {"delta": 0.52, "gamma": 0.001}, "oi": 12345, "iv": 0.14, "vtt": 999}}}
        r = flatten_feed("NSE_FO|1", f)
        self.assertEqual(r["ltp"], 101.5)
        self.assertEqual(r["bidq_sum"], 975)
        self.assertAlmostEqual(r["imbalance"], (975 - 195) / (975 + 195))
        self.assertEqual(r["delta"], 0.52)

    def test_chain_poller_writes_metrics(self):
        from nifty_engine.data import recorder
        from nifty_engine.options import bs_price
        rows = []
        for k in range(23500, 24550, 50):
            for typ in ("CE", "PE"):
                p = bs_price(24000, k, 5 / 365, 0.14, typ)
                rows.append({"strike": float(k), "type": typ, "ltp": p, "bid": p - 0.5, "ask": p + 0.5, "oi": 1000,
                             "volume": 10, "iv": 14.0, "delta": None, "gamma": None, "spot": 24000.0})
        fake = mock.Mock()
        fake.option_expiries = lambda: [dt.date.today() + dt.timedelta(days=5)]
        fake.option_chain = lambda e: pd.DataFrame(rows)
        m = recorder.ChainPoller(EngineConfig(), fake).poll_once()
        self.assertIn("near_pcr_oi", m)
        got = recorder.latest_chain_metrics()
        self.assertAlmostEqual(float(got["near_atm_iv"]), 0.14, places=2)

    def test_fomc_maps_to_next_session(self):
        from nifty_engine.data.events import impact_sessions
        s = pd.bdate_range("2026-10-26", periods=6)
        imp = impact_sessions(s, [{"type": "fomc", "date": dt.date(2026, 10, 28), "after_close": True},
                                  {"type": "rbi_mpc", "date": dt.date(2026, 10, 27), "after_close": False}])
        self.assertIn("fomc", imp[pd.Timestamp("2026-10-29")])
        self.assertIn("rbi_mpc", imp[pd.Timestamp("2026-10-27")])


class TestResearchHarness(unittest.TestCase):
    def test_runs_end_to_end_on_synthetic_store(self):
        from nifty_engine.data import store
        from nifty_engine.research.experiments import run_all
        from test_engine import make_market
        raw = make_market(n=1300, seed=31)
        idx = raw["nifty"].index
        raw["kospi"] = rand_ohlc(index=idx, seed=5)
        for k, v in raw.items():
            store.save(k, v)
        out = run_all(EngineConfig(), only=["P0", "P1", "S7b", "S1", "M1", "F1"], quick=True)
        res = out["results"]
        self.assertEqual(res["P0"]["status"], "REPORTED")
        self.assertIn(res["P1"]["status"], ("PASS", "FAIL", "INSUFFICIENT_DATA"))
        self.assertEqual(res["S1"]["status"], "INSUFFICIENT_DATA")      # no 5-minute data in the test store
        self.assertTrue((DATA_DIR / "reports" / "RESEARCH_REPORT.md").exists())
        self.assertNotIn("ERROR", {r["status"] for r in res.values()}, res)


if __name__ == "__main__":
    unittest.main()
