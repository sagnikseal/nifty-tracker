"""Intraday zoo tests: panel alignment never fills across sessions, sequences and TSFM contexts are
strictly causal (bars after T cannot change anything; the bar starting at T is excluded), every
sequence architecture trains/predicts, early stopping holds out the LAST days, TSFM features work
with an injected predictor (+ disk cache), and missing optional packages fail with install hints."""

import importlib.util
import os
import pickle
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

TMP = tempfile.mkdtemp(prefix="ne_zoo_")
os.environ.setdefault("NE_DATA_DIR", TMP)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nifty_engine.calendar import IST  # noqa: E402
from nifty_engine import intraday_zoo as zoo  # noqa: E402

HAS_TORCH = importlib.util.find_spec("torch") is not None


def synth(days=12, seed=0, start="2024-01-01", px=22000.0, vol=0.0012, volume=False):
    rng = np.random.default_rng(seed)
    out = []
    for d in pd.bdate_range(start, periods=days):
        idx = pd.date_range(d + pd.Timedelta(hours=9, minutes=15), periods=75, freq="5min", tz=IST)
        c = px * np.exp(np.cumsum(rng.normal(0, vol, len(idx))))
        o = np.r_[px * (1 + rng.normal(0, 0.002)), c[:-1]]
        df = pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.0004, "low": np.minimum(o, c) * 0.9996,
                           "close": c}, index=idx)
        if volume:
            df["volume"] = rng.integers(1000, 5000, len(idx)).astype(float)
        out.append(df)
        px = c[-1]
    return pd.concat(out)


def decision_times(bars, lo="09:20", hi="14:45"):
    days = sorted(set(bars.index.date))
    ts = []
    for d in days:
        ts += list(pd.date_range(pd.Timestamp(f"{d} {lo}", tz=IST), pd.Timestamp(f"{d} {hi}", tz=IST), freq="5min"))
    return pd.DatetimeIndex(ts)


class TestPanel(unittest.TestCase):
    def test_align_and_fill_within_session_only(self):
        n = synth(3, seed=1)
        v = synth(3, seed=2, px=14.0, vol=0.003)
        f = synth(3, seed=3, px=22100.0, volume=True)
        d2 = sorted(set(n.index.date))[1]
        t_mid = pd.Timestamp(f"{d2} 11:00", tz=IST)
        t_first = pd.Timestamp(f"{d2} 09:15", tz=IST)
        v = v.drop([t_mid, t_first])                        # VIX missing mid-session and at the open of day 2
        f = f.drop([t_mid])
        # string index with offsets (the CSV format) must work too
        v.index = [str(t) for t in v.index]
        P = zoo.build_panel({"nifty": n, "vix": v, "niftyfut": f})
        self.assertEqual(list(P.columns[:4]), ["nifty_open", "nifty_high", "nifty_low", "nifty_close"])
        self.assertEqual(len(P), len(n))
        prev = t_mid - zoo.BAR
        self.assertAlmostEqual(P.loc[t_mid, "vix_close"], P.loc[prev, "vix_close"])     # ffilled in-session
        self.assertEqual(P.loc[t_mid, "vix_ok"], 0.0)
        self.assertAlmostEqual(P.loc[t_mid, "vix_high"], P.loc[t_mid, "vix_close"])    # flat filled bar
        self.assertTrue(np.isnan(P.loc[t_first, "vix_close"]))                         # never across days
        self.assertEqual(P.loc[t_mid, "niftyfut_volume"], 0.0)                           # volume not ffilled
        self.assertGreater(P.loc[prev, "niftyfut_volume"], 0)

    def test_evening_bars_dropped(self):
        n = synth(2)
        extra = n.iloc[[-1]].copy()
        extra.index = [n.index[-1].normalize() + pd.Timedelta(hours=18, minutes=15)]
        P = zoo.build_panel({"nifty": pd.concat([n, extra])})
        self.assertEqual(len(P), len(n))


class TestSequences(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.n = synth(6, seed=4)
        cls.v = synth(6, seed=5, px=14.0, vol=0.003)
        cls.P = zoo.build_panel({"nifty": cls.n, "vix": cls.v})

    def test_shapes_and_channels(self):
        T = decision_times(self.n)
        X = zoo.build_sequences(self.P, T, lookback=48)
        ch = zoo.sequence_channels(self.P)
        self.assertEqual(X.shape, (len(T), 48, len(ch)))
        self.assertEqual(X.dtype, np.float32)
        self.assertTrue(np.isfinite(X).all())
        for c in ("nifty_r", "nifty_hl", "nifty_co", "vix_r", "tod_sin", "tod_cos", "session_start", "is_today", "mask"):
            self.assertIn(c, ch)
        self.assertNotIn("nifty_vol", ch)                     # index has no volume
        sub = zoo.build_sequences(self.P, T[:5], lookback=16, channels=["nifty_r", "mask"])
        self.assertEqual(sub.shape, (5, 16, 2))
        with self.assertRaises(KeyError):
            zoo.build_sequences(self.P, T[:5], channels=["nope"])

    def test_no_lookahead_and_bar_at_T_excluded(self):
        d3 = sorted(set(self.n.index.date))[3]
        T = pd.Timestamp(f"{d3} 11:00", tz=IST)
        for scale in ("sample", "ewm", "none"):
            base = zoo.build_sequences(self.P, pd.DatetimeIndex([T]), lookback=48, scale=scale)
            # perturb every bar that has NOT closed by T (start >= T), in every series
            P2 = self.P.copy()
            late = P2.index >= T
            for c in P2.columns:
                if c.endswith(("_open", "_high", "_low", "_close")):
                    P2.loc[late, c] = P2.loc[late, c] * np.linspace(1.5, 3.0, int(late.sum()))
            pert = zoo.build_sequences(P2, pd.DatetimeIndex([T]), lookback=48, scale=scale)
            np.testing.assert_array_equal(base, pert, err_msg=f"look-ahead with scale={scale}")
            # the bar that closes exactly at T (start T-5) IS used ...
            P3 = self.P.copy()
            P3.loc[T - zoo.BAR, "nifty_close"] *= 1.01
            self.assertFalse(np.array_equal(base, zoo.build_sequences(P3, pd.DatetimeIndex([T]), 48, scale=scale)))
        # ... and is the LAST element of the window; the first bar after T is not in the window
        ch = zoo.sequence_channels(self.P)
        X = zoo.build_sequences(self.P, pd.DatetimeIndex([T]), lookback=48, scale="none")
        r = ch.index("nifty_r")
        c = self.P["nifty_close"]
        exp_last = np.log(c.loc[T - zoo.BAR] / c.loc[T - 2 * zoo.BAR]) * 1000
        exp_next = np.log(c.loc[T] / c.loc[T - zoo.BAR]) * 1000
        self.assertAlmostEqual(float(X[0, -1, r]), exp_last, places=3)
        self.assertNotAlmostEqual(float(X[0, -1, r]), exp_next, places=3)
        # off-grid decision time: 11:02 sees exactly the same bars as 11:00
        X2 = zoo.build_sequences(self.P, pd.DatetimeIndex([T + pd.Timedelta(minutes=2)]), 48, scale="none")
        np.testing.assert_array_equal(X, X2)

    def test_padding_mask_and_session_marks(self):
        first = self.n.index[0]
        T = pd.DatetimeIndex([first + pd.Timedelta(minutes=20)])     # 4 bars closed
        X = zoo.build_sequences(self.P, T, lookback=48)
        ch = zoo.sequence_channels(self.P)
        m = X[0, :, ch.index("mask")]
        self.assertEqual(m.sum(), 4)
        self.assertTrue((m[-4:] == 1).all() and (m[:-4] == 0).all())
        self.assertTrue((X[0, :-4, :] == 0).all())
        # a 09:30 decision on day 2: window spans day 1 (is_today 0) and today's 3 bars (is_today 1)
        d1 = sorted(set(self.n.index.date))[1]
        X = zoo.build_sequences(self.P, pd.DatetimeIndex([pd.Timestamp(f"{d1} 09:30", tz=IST)]), lookback=48)
        it = X[0, :, ch.index("is_today")]
        ss = X[0, :, ch.index("session_start")]
        self.assertEqual(it.sum(), 3)
        self.assertEqual(ss[-3], 1.0)                            # 09:15 bar starts the session
        # 09:15 decision: nothing closed today yet
        X = zoo.build_sequences(self.P, pd.DatetimeIndex([pd.Timestamp(f"{d1} 09:15", tz=IST)]), lookback=48)
        self.assertEqual(X[0, :, ch.index("is_today")].sum(), 0)


def _planted_dataset(days=40, seed=0):
    """Sequences where y depends on the last three returns (a learnable momentum signal)."""
    n = synth(days, seed=seed)
    P = zoo.build_panel({"nifty": n})
    T = decision_times(n)
    X = zoo.build_sequences(P, T, lookback=24, scale="none")
    ch = zoo.sequence_channels(P)
    r = X[:, -3:, ch.index("nifty_r")].sum(1)
    y = (r + np.random.default_rng(seed).normal(0, 0.5, len(r)) > 0).astype(float)
    days_arr = np.asarray(T.tz_localize(None).normalize())
    return X, y, days_arr


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestSeqMember(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X, cls.y, cls.days = _planted_dataset()

    def test_all_archs_fit_predict(self):
        n = len(self.X)
        cut = int(n * 0.75)
        for arch in zoo.ARCHS:
            t0 = time.perf_counter()
            m = zoo.SeqMember(arch, lookback=24, quick=True, device="cpu").fit(self.X[:cut], self.y[:cut],
                                                                               self.days[:cut])
            p = m.predict_proba(self.X[cut:])
            self.assertEqual(p.shape, (n - cut,))
            self.assertTrue(np.all((p > 0) & (p < 1)), arch)
            self.assertGreater(m.n_params_, 0)
            self.assertLess(time.perf_counter() - t0, 20, arch)

    def test_learns_planted_signal_and_val_is_last_days(self):
        cut = int(len(self.X) * 0.75)
        m = zoo.SeqMember("tcn", lookback=24, hidden=16, epochs=8, seeds=1, patience=3, device="cpu")
        m.fit(self.X[:cut], self.y[:cut], self.days[:cut])
        ud = np.unique(self.days[:cut])
        np.testing.assert_array_equal(m.val_days_, ud[-len(m.val_days_):])
        self.assertEqual(len(m.val_days_), int(round(len(ud) * 0.15)))
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(self.y[cut:], m.predict_proba(self.X[cut:]))
        self.assertGreater(auc, 0.7)
        self.assertIsNotNone(m.val_logloss_)
        self.assertLess(m.val_logloss_, m.val_logloss_base_)

    def test_tabular_determinism_and_pickle(self):
        cut = int(len(self.X) * 0.75)
        rng = np.random.default_rng(3)
        tab = rng.normal(5, 3, (len(self.X), 4))
        tab[::7, 1] = np.nan
        kw = dict(lookback=24, quick=True, device="cpu")
        a = zoo.SeqMember("patchtst", **kw).fit(self.X[:cut], self.y[:cut], self.days[:cut], tab[:cut])
        b = zoo.SeqMember("patchtst", **kw).fit(self.X[:cut], self.y[:cut], self.days[:cut], tab[:cut])
        pa, pb = a.predict_proba(self.X[cut:], tab[cut:]), b.predict_proba(self.X[cut:], tab[cut:])
        np.testing.assert_allclose(pa, pb, atol=1e-6)
        # standardisation stats come from the training part only (validation days excluded)
        fit_rows = ~np.isin(self.days[:cut], a.val_days_)
        np.testing.assert_allclose(a._tab_mu, np.nanmean(tab[:cut][fit_rows], 0))
        with self.assertRaises(ValueError):
            a.predict_proba(self.X[cut:])
        c = pickle.loads(pickle.dumps(a))
        np.testing.assert_allclose(c.predict_proba(self.X[cut:], tab[cut:]), pa, atol=1e-6)

    def test_walk_forward_member(self):
        oos = zoo.walk_forward_member(self.X, self.y, self.days,
                                      lambda: zoo.SeqMember("cnn", lookback=24, quick=True, device="cpu"),
                                      warmup_days=25, refit_days=10, max_folds=2)
        self.assertEqual(oos["fold"].nunique(), 2)
        ud = np.unique(self.days)
        self.assertEqual(oos["day"].min(), ud[25])
        self.assertTrue(oos["p"].between(0, 1).all())

    def test_bad_inputs(self):
        with self.assertRaises(ValueError):
            zoo.SeqMember("transformer-xl")
        with self.assertRaises(ValueError):
            zoo.SeqMember(quick=True).fit(self.X[:10], self.y[:9], self.days[:10])


class FakeTSFM:
    """Records contexts; forecasts a fixed drift with symmetric quantiles."""

    def __init__(self, drift=0.0002):          # log drift per 5-minute step
        self.drift, self.calls = drift, []

    def __call__(self, contexts, horizon, ends=None, **_):
        self.calls.append(([np.array(c) for c in contexts], horizon, np.array(ends)))
        last = np.array([c[-1] for c in contexts])[:, None, None]
        steps = np.arange(1, horizon + 1)[None, :, None]
        z = (np.asarray(zoo.QUANTILES) - 0.5)[None, None, :] * 0.004
        return last * np.exp(self.drift * steps + z * np.sqrt(steps))


class Boom:
    def __call__(self, *a, **k):
        raise AssertionError("cache should have been used")


class TestTSFM(unittest.TestCase):
    def setUp(self):
        self.n = synth(4, seed=6)
        self.close = self.n["close"]
        self.T = decision_times(self.n)
        self.cache = tempfile.mkdtemp(prefix="tsfm_", dir=TMP)

    def test_causal_context_and_columns(self):
        fake = FakeTSFM()
        out = zoo.tsfm_features(self.close, self.T, model="chronos-bolt-small", horizon_bars=6, context=100,
                                batch=64, cache_dir=self.cache, predictor=fake, min_context=10)
        name = "tsfm_chronos_bolt_small"
        for sfx in ("30m", "close"):
            for k in ("ret_med", "q10", "q90", "width", "p_up"):
                self.assertIn(f"{name}_{k}_{sfx}", out.columns)
        self.assertEqual(len(out), len(self.T))
        # every context ends with the close of the bar that closed at T and nothing later
        ctxs = [c for call in fake.calls for c in call[0]]
        todo = self.T.unique().sort_values()
        ends = zoo._closed_by(self.close.index, todo)
        rows = np.flatnonzero(ends >= 10)
        self.assertEqual(len(ctxs), len(rows))
        for c, ri in zip(ctxs, rows):
            t = todo[ri]
            known = self.close[self.close.index + zoo.BAR <= t]
            self.assertLessEqual(len(c), 100)
            np.testing.assert_allclose(c, known.values[-len(c):])
            self.assertEqual(c[-1], self.close.loc[t - zoo.BAR])
        # horizon covers T -> 15:15 for the earliest decision of each batch
        self.assertTrue(all(call[1] >= 6 for call in fake.calls))
        valid = out.dropna()
        self.assertTrue((valid[f"{name}_q10_30m"] <= valid[f"{name}_ret_med_30m"]).all())
        self.assertTrue((valid[f"{name}_ret_med_30m"] <= valid[f"{name}_q90_30m"]).all())
        self.assertTrue((valid[f"{name}_p_up_30m"] > 0.5).all())              # positive drift
        np.testing.assert_allclose(valid[f"{name}_ret_med_30m"], 6 * 0.0002, atol=1e-9)
        # "to 15:15" uses the per-row step: 09:20 -> 71 bars, 14:45 -> 6 bars
        steps = (15 * 60 + 15 - (valid.index.hour * 60 + valid.index.minute)) // 5
        np.testing.assert_allclose(valid[f"{name}_ret_med_close"], 0.0002 * np.asarray(steps), atol=1e-9)
        # the first decisions of the sample have too little history -> NaN
        self.assertTrue(out.iloc[0].isna().all())

    def test_perturbing_future_bars_changes_nothing(self):
        T = pd.DatetimeIndex([self.T[200]])
        a = zoo.tsfm_features(self.close, T, predictor=FakeTSFM(), use_cache=False, min_context=10)
        c2 = self.close.copy()
        c2[c2.index >= T[0]] *= 1.2
        b = zoo.tsfm_features(c2, T, predictor=FakeTSFM(), use_cache=False, min_context=10)
        pd.testing.assert_frame_equal(a, b)

    def test_cache_roundtrip_and_incremental(self):
        half = self.T[: len(self.T) // 2]
        a = zoo.tsfm_features(self.close, half, cache_dir=self.cache, predictor=FakeTSFM(), min_context=10)
        b = zoo.tsfm_features(self.close, half, cache_dir=self.cache, predictor=Boom(), min_context=10)
        pd.testing.assert_frame_equal(a, b, check_freq=False)
        fake = FakeTSFM()
        full = zoo.tsfm_features(self.close, self.T, cache_dir=self.cache, predictor=fake, min_context=10)
        n_new = sum(len(call[0]) for call in fake.calls)
        self.assertEqual(n_new, len(self.T) - len(half))                     # only new rows forecast
        pd.testing.assert_frame_equal(full.loc[half], a, check_freq=False)
        files = list(Path(self.cache).glob("*.csv.gz"))
        self.assertEqual(len(files), 1)                                     # consolidated
        self.assertIn(self.T.max().strftime("%Y%m%dT%H%M"), files[0].name)

    def test_p_up(self):
        q = np.array([[-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3], [1, 2, 3, 4, 5, 6, 7, 8, 9],
                      [-9, -8, -7, -6, -5, -4, -3, -2, -1]], float)
        p = zoo._p_up(q)
        self.assertAlmostEqual(p[0], 0.5)
        self.assertGreater(p[1], 0.9)
        self.assertLess(p[2], 0.1)

    @unittest.skipIf(importlib.util.find_spec("chronos") is not None, "chronos installed")
    def test_missing_package_message(self):
        with self.assertRaises(ImportError) as cm:
            zoo.tsfm_features(self.close, self.T[-5:], model="chronos-bolt-small", use_cache=False, device="cpu")
        self.assertIn("pip install", str(cm.exception))

    @unittest.skipIf(zoo._kronos_dir() is not None, "Kronos installed")
    def test_missing_kronos_message(self):
        with self.assertRaises(ImportError) as cm:
            zoo.tsfm_features(self.close, self.T[-5:], model="kronos-small", use_cache=False, device="cpu")
        self.assertIn("git clone https://github.com/shiyu-coder/Kronos", str(cm.exception))

    @unittest.skipUnless(importlib.util.find_spec("chronos") is not None, "chronos-forecasting not installed")
    def test_real_chronos_small(self):  # pragma: no cover - only where the optional package exists
        out = zoo.tsfm_features(self.close, self.T[-3:], model="chronos-bolt-small", use_cache=False, device="cpu")
        self.assertEqual(out.notna().all(axis=1).sum(), 3)

    def test_unknown_model(self):
        with self.assertRaises(ValueError):
            zoo.tsfm_features(self.close, self.T[-3:], model="gpt-ts", use_cache=False)


class TestAvailable(unittest.TestCase):
    def test_keys(self):
        a = zoo.available_members()
        for k in ("torch", "mps", "cuda", *zoo.ARCHS, *zoo.TSFM_MODELS):
            self.assertIn(k, a)
            self.assertIsInstance(a[k], bool)
        self.assertEqual(a["tcn"], HAS_TORCH)


if __name__ == "__main__":
    unittest.main()
