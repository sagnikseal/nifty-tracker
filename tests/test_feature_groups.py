"""Feature groups: none may use data that arrives after the decision time T."""

import datetime as dt
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("NE_DATA_DIR", tempfile.mkdtemp(prefix="ne_v7_"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nifty_engine.calendar import IST  # noqa: E402


def bars(days=15, seed=0, base=22000.0, vol=True):
    rng = np.random.default_rng(seed)
    out, px = [], base
    for d in pd.bdate_range("2025-03-03", periods=days):
        idx = pd.date_range(d + pd.Timedelta(hours=9, minutes=15), periods=75, freq="5min", tz=IST)
        c = px * np.exp(np.cumsum(rng.normal(0, 0.0012, 75)))
        o = np.r_[px, c[:-1]]
        df = pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.0005, "low": np.minimum(o, c) * 0.9995, "close": c},
                          index=idx)
        if vol:
            df["volume"] = rng.integers(1000, 5000, 75).astype(float)
            df["oi"] = 1e6 + np.cumsum(rng.normal(0, 1000, 75))
        out.append(df)
        px = c[-1]
    return pd.concat(out)


def perturb_after(df, T, k=1.04):
    d = df.copy()
    m = d.index + pd.Timedelta(minutes=5) > T
    d.loc[m, ["open", "high", "low", "close"]] *= k
    if "volume" in d:
        d.loc[m, "volume"] *= 3
        d.loc[m, "oi"] *= 1.2
    return d


class TestV7NoLookahead(unittest.TestCase):
    def setUp(self):
        self.n5 = bars(15, 1)
        self.T = self.n5.index[75 * 12 + 30] + pd.Timedelta(minutes=5)
        self.dec = pd.DatetimeIndex([t + pd.Timedelta(minutes=5) for t in self.n5.index if t.date() >= self.T.date()
                                     and t.time() <= dt.time(14, 40)])

    def _same(self, A, B):
        a, b = A.loc[:self.T], B.loc[:self.T]
        bad = [c for c in A.columns if not np.allclose(a[c].astype(float), b[c].astype(float), equal_nan=True)]
        self.assertEqual(bad, [], f"look-ahead in {bad}")

    def test_technicals(self):
        from nifty_engine.analytics.patterns import tech_features
        A = tech_features(self.n5[["open", "high", "low", "close"]])
        B = tech_features(perturb_after(self.n5, self.T)[["open", "high", "low", "close"]])
        A.index, B.index = A.index + pd.Timedelta(minutes=5), B.index + pd.Timedelta(minutes=5)   # stamp = availability
        self.assertTrue(any(c.startswith("m60_") for c in A.columns))
        self._same(A, B)

    def test_cross_asset(self):
        from nifty_engine.analytics import cross_asset as ca
        bn, fut = bars(15, 2, 48000), bars(15, 3, 22050)
        stocks = {"hdfcbank": bars(15, 4, 1600), "reliance": bars(15, 5, 2500), "icicibank": bars(15, 6, 1200)}
        A = pd.concat([ca.banknifty_features(self.dec, self.n5, bn), ca.futures_features(self.dec, self.n5, fut),
                       ca.breadth_features(self.dec, self.n5, stocks)], axis=1)
        p = lambda d: perturb_after(d, self.T)  # noqa: E731
        B = pd.concat([ca.banknifty_features(self.dec, p(self.n5), p(bn)), ca.futures_features(self.dec, p(self.n5), p(fut)),
                       ca.breadth_features(self.dec, p(self.n5), {k: p(v) for k, v in stocks.items()})], axis=1)
        self.assertGreater(A["fut_vwap_dist"].notna().sum(), 0)
        self._same(A, B)

    def test_global_hourly(self):
        from nifty_engine.analytics import cross_asset as ca
        idx = pd.date_range("2025-03-01", "2025-03-25", freq="1h", tz="UTC")
        es = pd.DataFrame({"close": 5000 * np.exp(np.cumsum(np.random.default_rng(7).normal(0, 0.002, len(idx))))}, index=idx)
        A = ca.global_intraday_features(self.dec, {"es": es})
        es2 = es.copy()
        es2.loc[es2.index + pd.Timedelta(hours=1) > self.T, "close"] *= 1.1
        B = ca.global_intraday_features(self.dec, {"es": es2})
        self._same(A, B)

    def test_chain_day_features(self):
        from nifty_engine.analytics.chain_intraday import day_features
        d = self.T.date()
        g = self.n5[self.n5.index.date == d]
        spot5 = g["close"]
        mins = pd.date_range(g.index[0], periods=375, freq="1min")
        rows = []
        S0 = float(g["open"].iloc[0])
        K0 = round(S0 / 50) * 50
        rng = np.random.default_rng(3)
        for k in range(-3, 4):
            for typ in ("CE", "PE"):
                px = 120 + rng.normal(0, 2, 375).cumsum() * 0.1
                rows.append(pd.DataFrame({"ts": mins, "close": np.abs(px) + 5, "volume": rng.integers(100, 900, 375),
                                          "oi": 1e5 + rng.integers(0, 500, 375).cumsum(), "strike": K0 + 50 * k, "type": typ}))
        opt = pd.concat(rows)
        decs = [t + pd.Timedelta(minutes=5) for t in g.index if t.time() <= dt.time(14, 40)]
        A = pd.DataFrame(day_features(opt, spot5, decs, d + dt.timedelta(days=3))).set_index("ts")
        opt2 = opt.copy()
        late = opt2["ts"] + pd.Timedelta(minutes=1) > self.T
        opt2.loc[late, "close"] *= 2
        opt2.loc[late, "oi"] *= 3
        B = pd.DataFrame(day_features(opt2, spot5, decs, d + dt.timedelta(days=3))).set_index("ts")
        self.assertIn("oc_pcr_oi", A)
        self._same(A, B)


if __name__ == "__main__":
    unittest.main()
