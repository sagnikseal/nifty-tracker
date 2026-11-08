"""Intraday model zoo: sequence members and time-series-foundation-model (TSFM) features for the
5-minute NIFTY direction problem in `intraday_model.py`.

What lives here
  * `build_panel`      - align several start-stamped 5-minute OHLC(V) series into one panel.
  * `build_sequences`  - [N, lookback, C] float32 windows of *closed* bars for each decision time T.
  * `SeqMember`        - small, heavily regularised neural members (TCN, GRU, LSTM, PatchTST-style
                         Transformer, 1-D CNN) with day-ordered early stopping and seed averaging.
  * `tsfm_features`    - zero-shot forecasts from Chronos-Bolt / Chronos-2 / TimesFM 2.5 / Kronos,
                         turned into causal features (quantiles of the forecast return, P(up)).
  * `walk_forward_member` - expanding-window, whole-day walk-forward for one sequence member (raw OOS
                         probabilities; calibration/selection stays with the main engine).

Timing rule (identical to intraday_model): a bar stamped s covers [s, s+5min) and is usable at T only
if s + 5min <= T. Everything here is built on that one comparison (`_closed_by`).

Honesty note: out-of-sample, the existing logistic/LightGBM members are ~50-51% accurate with no
log-loss skill. These members exist to give the walk-forward genuinely *different* hypotheses (raw
path shape, cross-asset lead/lag, pretrained priors), not because deep nets are expected to win.
"""

from __future__ import annotations

import contextlib
import functools
import glob
import hashlib
import importlib.util
import logging
import math
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from .calendar import IST

log = logging.getLogger(__name__)

BAR = pd.Timedelta(minutes=5)
_BAR_NS = 5 * 60 * 1_000_000_000
_DAY_NS = 86_400 * 1_000_000_000
SESSION_MINUTES = 375                   # 09:15 - 15:30
FIRST_BAR_MIN = 9 * 60 + 15             # first bar start (minutes after midnight)
LAST_BAR_MIN = 15 * 60 + 25             # last regular bar start (closes 15:30)
EXIT_MIN = 15 * 60 + 15                 # the 15:15 exit used by `yclose`
ARCHS = ("tcn", "gru", "lstm", "patchtst", "cnn")
QUANTILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
_SCALED_SUFFIXES = ("_r", "_hl", "_co")  # return/range channels: rescaled per sample (or by causal EWM)


# ============================================================================ time helpers
def _ist_index(idx) -> pd.DatetimeIndex:
    """Any index of timestamps (strings with offsets, naive = IST wall time, tz-aware) -> IST index."""
    if not isinstance(idx, pd.DatetimeIndex):
        try:
            idx = pd.DatetimeIndex(pd.to_datetime(pd.Index(idx)))
        except (ValueError, TypeError):
            idx = pd.DatetimeIndex(pd.to_datetime(pd.Index(idx), utc=True))
    if idx.tz is None:
        idx = idx.tz_localize(IST)
    return idx.tz_convert(IST)


def _ns(idx: pd.DatetimeIndex) -> np.ndarray:
    """int64 nanoseconds since epoch (pandas 3 may parse at 'us' resolution, so normalise the unit)."""
    return np.asarray(idx.as_unit("ns").asi8, dtype=np.int64)


def _day_number(idx: pd.DatetimeIndex) -> np.ndarray:
    """IST calendar day as an integer (days since epoch, wall-clock)."""
    return np.asarray(idx.tz_localize(None).as_unit("ns").asi8 // _DAY_NS, dtype=np.int64)


def _closed_by(bar_start: pd.DatetimeIndex, decision_ts: pd.DatetimeIndex) -> np.ndarray:
    """For each decision time T: the number of bars (in sorted `bar_start`) with start + 5min <= T.
    Bars [0, ends[i]) are known at T; bar ends[i] (the one starting at T) is NOT."""
    return np.searchsorted(_ns(bar_start) + _BAR_NS, _ns(decision_ts), side="right")


def _session_filter(df: pd.DataFrame) -> pd.DataFrame:
    m = df.index.hour * 60 + df.index.minute
    return df[(m >= FIRST_BAR_MIN) & (m <= LAST_BAR_MIN)]


# ============================================================================ panel
def _norm_bars(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.index = _ist_index(df.index)
    df.columns = [str(c).lower() for c in df.columns]
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return _session_filter(df)


def build_panel(series: Dict[str, pd.DataFrame], primary: Optional[str] = None) -> pd.DataFrame:
    """Merge start-stamped 5-minute OHLC(V) frames into one aligned panel.

    series: {"nifty": df, "vix": df, "banknifty": df, "niftyfut": df, ...}; each df has open/high/low/close
    (+ optional volume, oi) and a timestamp index (tz-aware, strings with offsets, or naive IST).
    Output: IST index = union of all bar starts within 09:15-15:25 (muhurat/evening bars dropped);
    columns `<name>_<field>` plus `<name>_ok` (1 = a real bar, 0 = filled). Missing bars are filled
    only from earlier bars of the SAME session: close/oi forward-filled, open/high/low set to that
    close (a flat, zero-range bar); volume of a missing bar is 0 and is never forward-filled. A series
    that has not printed yet in a session stays NaN (never carried across days). `primary` (default:
    "nifty" if present, else the first key) is placed first."""
    if not series:
        return pd.DataFrame()
    names = list(series)
    primary = primary or ("nifty" if "nifty" in names else names[0])
    names = [primary] + [n for n in names if n != primary]
    frames = {n: _norm_bars(series[n]) for n in names}
    idx = frames[names[0]].index
    for n in names[1:]:
        idx = idx.union(frames[n].index)
    idx = idx.sort_values()
    day = _day_number(idx)
    cols = {}
    for n in names:
        f = frames[n].reindex(idx)
        ok = f["close"].notna()
        close = f["close"].groupby(day).ffill()
        cols[f"{n}_open"] = f["open"].where(ok, close) if "open" in f else close
        cols[f"{n}_high"] = f["high"].where(ok, close) if "high" in f else close
        cols[f"{n}_low"] = f["low"].where(ok, close) if "low" in f else close
        cols[f"{n}_close"] = close
        if "volume" in f:
            cols[f"{n}_volume"] = f["volume"].where(ok, 0.0).fillna(0.0)
        if "oi" in f:
            cols[f"{n}_oi"] = f["oi"].groupby(day).ffill()
        cols[f"{n}_ok"] = ok.astype(float)
    out = pd.DataFrame(cols, index=idx)
    out.index.name = "ts"
    return out


def _prefixes(panel: pd.DataFrame) -> List[str]:
    pre = [c[:-6] for c in panel.columns if c.endswith("_close")]
    prim = "nifty" if "nifty" in pre else (pre[0] if pre else None)
    return ([prim] + [p for p in pre if p != prim]) if prim else []


# ============================================================================ sequences
def _bar_features(panel: pd.DataFrame, min_volume_share: float = 0.5) -> pd.DataFrame:
    """Per-bar stationary channels (each uses only that bar and earlier bars)."""
    idx = _ist_index(panel.index)
    pre = _prefixes(panel)
    out: Dict[str, pd.Series] = {}
    for k, p in enumerate(pre):
        c = panel[f"{p}_close"].astype(float).set_axis(idx)
        lc = np.log(c.where(c > 0))
        out[f"{p}_r"] = (lc - lc.shift(1)).clip(-0.1, 0.1)         # first bar of a session carries the gap
        if f"{p}_high" in panel and f"{p}_low" in panel:
            h = panel[f"{p}_high"].astype(float).set_axis(idx)
            lo = panel[f"{p}_low"].astype(float).set_axis(idx)
            out[f"{p}_hl"] = np.log(h.where(h > 0) / lo.where(lo > 0)).clip(0, 0.1)
        if k == 0 and f"{p}_open" in panel:
            o = panel[f"{p}_open"].astype(float).set_axis(idx)
            out[f"{p}_co"] = (lc - np.log(o.where(o > 0))).clip(-0.1, 0.1)
        vcol = f"{p}_volume"
        if vcol in panel and float((panel[vcol] > 0).mean()) >= min_volume_share:
            lv = np.log1p(panel[vcol].astype(float).clip(lower=0).set_axis(idx))
            out[f"{p}_vol"] = (lv - lv.rolling(SESSION_MINUTES, min_periods=20).mean()).clip(-5, 5)
    minutes = np.asarray(idx.hour * 60 + idx.minute, dtype=float) - FIRST_BAR_MIN
    ang = 2 * math.pi * minutes / SESSION_MINUTES
    out["tod_sin"] = pd.Series(np.sin(ang), index=idx)
    out["tod_cos"] = pd.Series(np.cos(ang), index=idx)
    day = _day_number(idx)
    out["session_start"] = pd.Series(np.r_[1.0, (day[1:] != day[:-1]).astype(float)] if len(day) else [],
                                     index=idx)
    return pd.DataFrame(out, index=idx)


def sequence_channels(panel: pd.DataFrame) -> List[str]:
    """All channel names `build_sequences` can produce for this panel (default order)."""
    return list(_bar_features(panel).columns) + ["is_today", "mask"]


def build_sequences(panel: pd.DataFrame, decision_ts: pd.DatetimeIndex, lookback: int = 48,
                    channels: Optional[List[str]] = None, scale: str = "sample") -> np.ndarray:
    """float32 [N, lookback, C]: for each decision time T the last `lookback` bars with start + 5min <= T
    (spanning previous sessions when needed), oldest first; position -1 is the bar that closed at T.

    Channels (see `sequence_channels`): `<s>_r` log close-to-close return (the first bar of a session
    carries the overnight gap), `<s>_hl` log(high/low), `<primary>_co` log(close/open), `<s>_vol` log-volume
    minus its causal 375-bar mean (only for series that actually report volume), `tod_sin/cos`,
    `session_start` (1 on the first bar of every session), `is_today` (bar belongs to T's session),
    `mask` (1 = real bar, 0 = zero padding when history is shorter than `lookback`).
    scale="sample": return/range channels divided by their RMS over the window (RevIN-style, keeps the
    sign of drift); "ewm": divided by a causal EWM RMS (half-life one session) so volatility regime
    stays visible; "none": raw x 1000. All values are finite (NaN -> 0) and clipped to +-8."""
    if scale not in ("sample", "ewm", "none"):
        raise ValueError("scale must be 'sample', 'ewm' or 'none'")
    T = _ist_index(decision_ts)
    panel = panel.copy()
    panel.index = _ist_index(panel.index)
    panel = panel[~panel.index.duplicated(keep="last")].sort_index()
    feats = _bar_features(panel)
    names_bar = list(feats.columns)
    all_names = names_bar + ["is_today", "mask"]
    channels = list(channels) if channels is not None else all_names
    bad = [c for c in channels if c not in all_names]
    if bad:
        raise KeyError(f"unknown channels {bad}; available: {all_names}")
    prim = _prefixes(panel)[0]
    valid = panel[f"{prim}_close"].notna().to_numpy()

    if scale == "ewm":
        for c in names_bar:
            if c.endswith(_SCALED_SUFFIXES):
                rms = np.sqrt((feats[c] ** 2).ewm(halflife=SESSION_MINUTES, min_periods=20).mean())
                feats[c] = feats[c] / rms.where(rms > 0)
    elif scale == "none":
        for c in names_bar:
            if c.endswith(_SCALED_SUFFIXES):
                feats[c] = feats[c] * 1000.0
    F = np.nan_to_num(feats.to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    L, n, Cb = int(lookback), len(F), F.shape[1]
    Fp = np.concatenate([np.zeros((L, Cb), np.float32), F], 0)
    mp = np.concatenate([np.zeros(L, np.float32), valid.astype(np.float32)])
    dp = np.concatenate([np.full(L, -1, np.int64), _day_number(panel.index)])
    ends = _closed_by(panel.index, T)                     # window = bars [ends-L, ends) = padded [ends, ends+L)
    W = np.lib.stride_tricks.sliding_window_view(Fp, L, axis=0)        # [n+1, Cb, L]
    seq = np.ascontiguousarray(W[ends].transpose(0, 2, 1))              # [N, L, Cb]
    mask = np.lib.stride_tricks.sliding_window_view(mp, L)[ends]        # [N, L]
    wday = np.lib.stride_tricks.sliding_window_view(dp, L)[ends]
    seq *= mask[:, :, None]
    if scale == "sample":
        cnt = np.maximum(mask.sum(1), 1.0)[:, None]
        for j, c in enumerate(names_bar):
            if c.endswith(_SCALED_SUFFIXES):
                rms = np.sqrt((seq[:, :, j] ** 2).sum(1, keepdims=True) / cnt)
                seq[:, :, j] = np.where(rms > 1e-12, seq[:, :, j] / np.where(rms > 1e-12, rms, 1.0), 0.0)
    is_today = ((wday == _day_number(T)[:, None]) & (mask > 0)).astype(np.float32)
    full = np.concatenate([seq, is_today[:, :, None], mask[:, :, None].astype(np.float32)], axis=2)
    np.clip(full, -8, 8, out=full)
    pick = [all_names.index(c) for c in channels]
    return np.ascontiguousarray(full[:, :, pick], dtype=np.float32)


# ============================================================================ torch helpers
def _torch():
    try:
        import torch
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise ImportError("PyTorch is required for sequence members: pip install torch") from e
    return torch


def resolve_device(device: str = "auto") -> str:
    torch = _torch()
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


@contextlib.contextmanager
def _cpu_threads(device: str):
    """Tiny nets are slower on many threads; use a sensible count on CPU and restore afterwards."""
    torch = _torch()
    if device != "cpu":
        yield
        return
    old = torch.get_num_threads()
    torch.set_num_threads(max(1, min(8, (os.cpu_count() or 2) // 2)))
    try:
        yield
    finally:
        torch.set_num_threads(old)


@functools.lru_cache(maxsize=1)
def _net_class():
    """Network definitions, built lazily so importing this module does not import torch."""
    torch = _torch()
    from torch import nn
    import torch.nn.functional as Fn

    class CausalConv(nn.Module):
        def __init__(self, cin, cout, k, d):
            super().__init__()
            self.pad = (k - 1) * d
            self.conv = nn.Conv1d(cin, cout, k, dilation=d)

        def forward(self, x):
            return self.conv(Fn.pad(x, (self.pad, 0)))

    class TCNBlock(nn.Module):
        def __init__(self, h, k, d, p):
            super().__init__()
            self.c1, self.c2 = CausalConv(h, h, k, d), CausalConv(h, h, k, d)
            self.drop, self.act = nn.Dropout(p), nn.GELU()

        def forward(self, x):
            y = self.drop(self.act(self.c1(x)))
            y = self.drop(self.c2(y))
            return self.act(x + y)

    class TCN(nn.Module):
        """Dilated causal convolutions (k=3, dilations 1,2,4,.. until the receptive field covers L)."""

        def __init__(self, C, L, h, p, k=3):
            super().__init__()
            self.inp = nn.Conv1d(C, h, 1)
            blocks, d, rf = [], 1, 1
            while rf < L:
                blocks.append(TCNBlock(h, k, d, p))
                rf += 2 * (k - 1) * d
                d *= 2
            self.blocks = nn.Sequential(*blocks)
            self.norm = nn.LayerNorm(h)
            self.out_dim = h

        def forward(self, x):                     # x [B, L, C]
            z = self.blocks(self.inp(x.transpose(1, 2)))
            return self.norm(z[:, :, -1])          # state at the last closed bar

    class RNN(nn.Module):
        def __init__(self, kind, C, h, p, layers=1):
            super().__init__()
            cls = nn.GRU if kind == "gru" else nn.LSTM
            self.rnn = cls(C, h, num_layers=layers, batch_first=True, dropout=p if layers > 1 else 0.0)
            self.norm = nn.LayerNorm(h)
            self.out_dim = h

        def forward(self, x):
            o, _ = self.rnn(x)
            return self.norm(o[:, -1])

    class CNN(nn.Module):
        def __init__(self, C, h, p):
            super().__init__()
            self.c1 = nn.Conv1d(C, h, 5, padding=2)
            self.c2 = nn.Conv1d(h, h, 5, padding=4, dilation=2)
            self.drop, self.act = nn.Dropout(p), nn.GELU()
            self.out_dim = 2 * h

        def forward(self, x):
            z = self.drop(self.act(self.c1(x.transpose(1, 2))))
            z = self.act(self.c2(z))
            return torch.cat([z.mean(2), z[:, :, -1]], 1)

    class PatchTST(nn.Module):
        """PatchTST-style encoder (channel-mixing variant: each patch embeds all channels)."""

        def __init__(self, C, L, d, p, patch=8, stride=4, layers=2):
            super().__init__()
            self.patch = min(patch, L)
            self.stride = max(1, min(stride, self.patch))
            n_p = (L - self.patch) // self.stride + 1
            heads = 4 if d % 4 == 0 and d >= 32 else 2
            self.embed = nn.Linear(self.patch * C, d)
            self.pos = nn.Parameter(torch.zeros(1, n_p, d))
            nn.init.normal_(self.pos, std=0.02)
            layer = nn.TransformerEncoderLayer(d, heads, dim_feedforward=2 * d, dropout=p, batch_first=True,
                                               activation="gelu", norm_first=True)
            self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
            self.norm, self.drop = nn.LayerNorm(d), nn.Dropout(p)
            self.out_dim = d

        def forward(self, x):                     # [B, L, C]
            B = x.shape[0]
            pt = x.unfold(1, self.patch, self.stride)          # [B, n_p, C, patch]
            z = self.embed(pt.reshape(B, pt.shape[1], -1)) + self.pos
            return self.norm(self.enc(self.drop(z))).mean(1)

    class Net(nn.Module):
        def __init__(self, arch, C, L, hidden, dropout, tab_dim=0):
            super().__init__()
            if arch == "tcn":
                self.enc = TCN(C, L, hidden, dropout)
            elif arch in ("gru", "lstm"):
                self.enc = RNN(arch, C, hidden, dropout)
            elif arch == "cnn":
                self.enc = CNN(C, hidden, dropout)
            elif arch == "patchtst":
                self.enc = PatchTST(C, L, hidden + (hidden % 2), dropout)
            else:
                raise ValueError(f"unknown arch {arch}; choose from {ARCHS}")
            self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.enc.out_dim + tab_dim, hidden), nn.GELU(),
                                      nn.Dropout(dropout), nn.Linear(hidden, 1))

        def forward(self, x, t=None):
            z = self.enc(x)
            if t is not None:
                z = torch.cat([z, t], 1)
            return self.head(z).squeeze(-1)

    return Net


# ============================================================================ sequence member
class SeqMember:
    """A small neural sequence classifier for P(up), averaged over `seeds` independently seeded nets.

    Regularisation: dropout, AdamW weight decay, gradient clipping, early stopping on the LAST
    `val_frac` of training DAYS (time-ordered; rows of a day never straddle the split), best-epoch
    weights restored. Optional tabular features (e.g. the intraday feature frame) are standardised with
    statistics from the training part only and concatenated after the sequence encoder.
    Pickle-friendly: only CPU state dicts are stored (works with joblib like the other members)."""

    def __init__(self, arch: str = "tcn", lookback: int = 48, hidden: int = 32, dropout: float = 0.3,
                 epochs: int = 30, lr: float = 1e-3, weight_decay: float = 1e-3, batch: int = 512, seeds: int = 3,
                 patience: int = 4, device: str = "auto", quick: bool = False, val_frac: float = 0.15,
                 seed: int = 7):
        if arch not in ARCHS:
            raise ValueError(f"unknown arch {arch}; choose from {ARCHS}")
        if quick:
            seeds, epochs, hidden, patience = 1, 3, min(hidden, 8), 1
        self.arch, self.lookback, self.hidden, self.dropout = arch, lookback, hidden, dropout
        self.epochs, self.lr, self.weight_decay, self.batch = epochs, lr, weight_decay, batch
        self.seeds, self.patience, self.device, self.quick = seeds, patience, device, quick
        self.val_frac, self.seed = val_frac, seed
        self.states: List[dict] = []
        self.history_: List[dict] = []
        self.n_params_ = 0
        self.in_shape_ = None
        self.tab_dim_ = 0
        self._tab_mu = self._tab_sd = None
        self._nets = None

    # ---- pickling: keep state dicts only
    def __getstate__(self):
        d = self.__dict__.copy()
        d["_nets"] = None
        return d

    def _make(self):
        Net = _net_class()
        C, L = self.in_shape_[1], self.in_shape_[0]
        return Net(self.arch, C, L, self.hidden, self.dropout, self.tab_dim_)

    def _prep_tab(self, tab):
        if tab is None or not self.tab_dim_:
            return None
        t = (np.asarray(tab, np.float64) - self._tab_mu) / self._tab_sd
        return np.clip(np.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0), -5, 5).astype(np.float32)

    def fit(self, seq: np.ndarray, y: np.ndarray, days: np.ndarray, tab: Optional[np.ndarray] = None) -> "SeqMember":
        torch = _torch()
        X = np.nan_to_num(np.asarray(seq, np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        y = np.asarray(y, np.float32).ravel()
        days = np.asarray(days)
        if not (len(X) == len(y) == len(days)) or X.ndim != 3:
            raise ValueError("seq [N, L, C], y [N] and days [N] must align")
        if np.isnan(y).any():
            raise ValueError("y contains NaN; drop unlabeled rows first")
        self.in_shape_ = (X.shape[1], X.shape[2])
        ud = np.unique(days)                                   # sorted: time order
        n_val = int(round(len(ud) * self.val_frac)) if len(ud) >= 5 else 0
        n_val = max(1, n_val) if len(ud) >= 5 else 0
        self.val_days_ = ud[len(ud) - n_val:] if n_val else ud[:0]
        is_val = np.isin(days, self.val_days_)
        if tab is not None:
            tab = np.asarray(tab, np.float64)
            if tab.ndim == 1:
                tab = tab[:, None]
            fit_part = tab[~is_val] if (~is_val).any() else tab
            with warnings.catch_warnings():                # all-NaN columns (e.g. no news) -> mu 0, sd 1
                warnings.simplefilter("ignore", RuntimeWarning)
                mu = np.nanmean(fit_part, 0)
                sd = np.nanstd(fit_part, 0)
            self._tab_mu = np.nan_to_num(mu, nan=0.0)
            self._tab_sd = np.where(np.isfinite(sd) & (sd > 1e-12), sd, 1.0)
            self.tab_dim_ = tab.shape[1]
        else:
            self.tab_dim_ = 0
        Tt = self._prep_tab(tab)
        dev = resolve_device(self.device)
        self.device_used_ = dev
        self.states, self.history_ = [], []
        with _cpu_threads(dev):
            Xd = torch.from_numpy(X).to(dev)
            yd = torch.from_numpy(y).to(dev)
            Td = torch.from_numpy(Tt).to(dev) if Tt is not None else None
            tr_idx = torch.from_numpy(np.flatnonzero(~is_val)).to(dev)
            va_idx = torch.from_numpy(np.flatnonzero(is_val)).to(dev)
            base = float(y[~is_val].mean()) if (~is_val).any() else 0.5
            for s in range(self.seeds):
                state, hist = self._train_one(self.seed + 1000 * s, Xd, yd, Td, tr_idx, va_idx, base, dev)
                self.states.append(state)
                self.history_.append(hist)
        self.n_params_ = int(sum(p.numel() for p in self._make().parameters()))
        self._nets = None
        vl = [h["best_val_logloss"] for h in self.history_ if h["best_val_logloss"] is not None]
        self.val_logloss_ = float(np.mean(vl)) if vl else None
        self.val_logloss_base_ = self.history_[0]["val_logloss_base"] if self.history_ else None
        return self

    def _train_one(self, seed, Xd, yd, Td, tr_idx, va_idx, base, dev):
        torch = _torch()
        torch.manual_seed(seed)
        np.random.seed(seed % (2 ** 32))
        net = self._make().to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        lossf = torch.nn.BCEWithLogitsLoss()
        gen = torch.Generator().manual_seed(seed)
        n_tr, has_val = len(tr_idx), len(va_idx) > 0
        best, best_state, best_ep, bad = math.inf, None, 0, 0
        val_base = None
        if has_val:
            yv = yd[va_idx]
            pb = min(max(base, 1e-4), 1 - 1e-4)
            val_base = float(-(yv * math.log(pb) + (1 - yv) * math.log(1 - pb)).mean())
        t0 = time.perf_counter()
        ep_done = 0
        for ep in range(self.epochs):
            net.train()
            perm = tr_idx[torch.randperm(n_tr, generator=gen).to(dev)]
            for i in range(0, n_tr, self.batch):
                b = perm[i:i + self.batch]
                logit = net(Xd[b], Td[b] if Td is not None else None)
                loss = lossf(logit, yd[b])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
            ep_done = ep + 1
            if has_val:
                vl = float(lossf(self._logits(net, Xd, Td, va_idx), yd[va_idx]))
                if vl < best - 1e-5:
                    best, best_ep, bad = vl, ep + 1, 0
                    best_state = {k: v.detach().to("cpu").clone() for k, v in net.state_dict().items()}
                else:
                    bad += 1
                    if bad >= self.patience:
                        break
        if best_state is None:
            best_state = {k: v.detach().to("cpu").clone() for k, v in net.state_dict().items()}
            best_ep = ep_done
        return best_state, {"seed": seed, "epochs_run": ep_done, "best_epoch": best_ep,
                            "best_val_logloss": best if has_val and best < math.inf else None,
                            "val_logloss_base": val_base, "seconds": time.perf_counter() - t0}

    @staticmethod
    def _logits(net, Xd, Td, idx, chunk: int = 8192):
        torch = _torch()
        net.eval()
        out = []
        with torch.inference_mode():
            for i in range(0, len(idx), chunk):
                b = idx[i:i + chunk]
                out.append(net(Xd[b], Td[b] if Td is not None else None))
        return torch.cat(out) if out else torch.zeros(0, device=Xd.device)

    def predict_proba(self, seq: np.ndarray, tab: Optional[np.ndarray] = None) -> np.ndarray:
        if not self.states:
            raise RuntimeError("SeqMember is not fitted")
        torch = _torch()
        X = np.nan_to_num(np.asarray(seq, np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        if X.shape[1:] != tuple(self.in_shape_):
            raise ValueError(f"expected sequences of shape [N, {self.in_shape_[0]}, {self.in_shape_[1]}], got {X.shape}")
        if self.tab_dim_ and tab is None:
            raise ValueError("this member was fitted with tabular features; pass tab=")
        if tab is not None and np.asarray(tab).ndim == 1:
            tab = np.asarray(tab)[:, None]
        Tt = self._prep_tab(tab)
        dev = resolve_device(self.device)
        with _cpu_threads(dev):
            if self._nets is None or self._nets[0] != dev:
                nets = []
                for st in self.states:
                    net = self._make()
                    net.load_state_dict(st)
                    nets.append(net.to(dev).eval())
                self._nets = (dev, nets)
            Xd = torch.from_numpy(X).to(dev)
            Td = torch.from_numpy(Tt).to(dev) if Tt is not None else None
            idx = torch.arange(len(X), device=dev)
            ps = [torch.sigmoid(self._logits(n, Xd, Td, idx)).float().cpu().numpy() for n in self._nets[1]]
        return np.clip(np.mean(ps, axis=0).astype(np.float64), 1e-4, 1 - 1e-4)


def walk_forward_member(seq: np.ndarray, y: np.ndarray, days: np.ndarray, make_member: Callable[[], SeqMember],
                        tab: Optional[np.ndarray] = None, warmup_days: int = 250, refit_days: int = 21,
                        max_train_days: Optional[int] = None, max_folds: Optional[int] = None) -> pd.DataFrame:
    """Expanding-window walk-forward by whole trading days for one sequence member. Rows with NaN y are
    excluded from training but still predicted. Returns one row per OOS prediction: row (position in the
    inputs), day, p (raw, uncalibrated), y, fold, fit_s, val_ll, val_ll_base. Calibration and member
    selection are left to the main engine (`intraday_model.walk_forward` conventions)."""
    y = np.asarray(y, float)
    days = np.asarray(days)
    ud = np.unique(days)
    out, i, fold = [], warmup_days, 0
    while i < len(ud) and (max_folds is None or fold < max_folds):
        te_days = ud[i:i + refit_days]
        tr_days = ud[max(0, i - max_train_days):i] if max_train_days else ud[:i]
        tr = np.isin(days, tr_days) & ~np.isnan(y)
        te = np.isin(days, te_days)
        t0 = time.perf_counter()
        m = make_member().fit(seq[tr], y[tr], days[tr], tab[tr] if tab is not None else None)
        fit_s = time.perf_counter() - t0
        p = m.predict_proba(seq[te], tab[te] if tab is not None else None)
        out.append(pd.DataFrame({"row": np.flatnonzero(te), "day": days[te], "p": p, "y": y[te], "fold": fold,
                                 "fit_s": fit_s, "val_ll": m.val_logloss_, "val_ll_base": m.val_logloss_base_}))
        log.info("zoo %s fold %d: train %d days, test %s..%s, fit %.1fs", m.arch, fold, len(tr_days),
                 str(te_days[0])[:10], str(te_days[-1])[:10], fit_s)
        i += refit_days
        fold += 1
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


# ============================================================================ TSFM features
TSFM_MODELS = {
    "chronos-bolt-small": ("chronos", "amazon/chronos-bolt-small"),
    "chronos-bolt-base": ("chronos", "amazon/chronos-bolt-base"),
    "chronos-2": ("chronos2", "amazon/chronos-2"),
    "timesfm-2.5": ("timesfm", "google/timesfm-2.5-200m-pytorch"),
    "kronos-small": ("kronos", ("NeoQuasar/Kronos-small", "NeoQuasar/Kronos-Tokenizer-base", 512)),
    "kronos-base": ("kronos", ("NeoQuasar/Kronos-base", "NeoQuasar/Kronos-Tokenizer-base", 512)),
}
TSFM_INSTALL = {
    "chronos": 'pip install "chronos-forecasting>=1.5"',
    "chronos2": 'pip install "chronos-forecasting>=2.0"',
    "timesfm": 'pip install "timesfm[torch] @ git+https://github.com/google-research/timesfm.git"',
    "kronos": ("git clone https://github.com/shiyu-coder/Kronos ~/Kronos && pip install -r ~/Kronos/requirements.txt "
               "&& export KRONOS_PATH=~/Kronos   (Kronos is not on PyPI; KRONOS_PATH must point at the repo root)"),
}


def _kronos_dir() -> Optional[Path]:
    for cand in (os.environ.get("KRONOS_PATH"), str(Path.home() / "Kronos")):
        if cand and (Path(cand).expanduser() / "model" / "kronos.py").exists():
            return Path(cand).expanduser()
    return None


def _chronos_major() -> int:
    try:
        from importlib.metadata import version
        return int(version("chronos-forecasting").split(".")[0])
    except Exception:  # noqa: BLE001
        return 0


def _stack_quantiles(q) -> np.ndarray:
    """Normalise adapter outputs to numpy [N, H, Q]."""
    if isinstance(q, (list, tuple)):
        q = [x.detach().float().cpu().numpy() if hasattr(x, "detach") else np.asarray(x) for x in q]
        q = np.stack([x[0] if x.ndim == 3 else x for x in q])          # chronos-2: [variates, H, Q] per item
    elif hasattr(q, "detach"):
        q = q.detach().float().cpu().numpy()
    q = np.asarray(q, float)
    if q.ndim == 4:                                                     # [N, variates, H, Q]
        q = q[:, 0]
    return q


class _ChronosAdapter:
    """chronos-forecasting: BaseChronosPipeline.from_pretrained(...).predict_quantiles(context, prediction_length,
    quantile_levels) -> (quantiles [N, H, Q], mean). Chronos-2 (package >= 2.0) dispatches through the same
    entry point and returns per-series [variates, H, Q]; `_stack_quantiles` handles both."""

    def __init__(self, repo: str, device: str):
        from chronos import BaseChronosPipeline
        self.pipe = BaseChronosPipeline.from_pretrained(repo, device_map=device)

    def __call__(self, contexts, horizon, **_):
        import torch
        ctx = [torch.tensor(np.asarray(c, np.float32)) for c in contexts]
        q, _mean = self.pipe.predict_quantiles(ctx, prediction_length=horizon, quantile_levels=list(QUANTILES))
        return _stack_quantiles(q)


class _TimesFMAdapter:
    """timesfm 2.5: TimesFM_2p5_200M_torch.from_pretrained(...); compile(ForecastConfig(...));
    forecast(horizon, inputs) -> (point [N, H], quantiles [N, H, 10] = mean then q10..q90)."""

    def __init__(self, repo: str, context: int, max_horizon: int = 128):
        import timesfm
        self.model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(repo)
        self.max_h = max(128, int(math.ceil(max_horizon / 128) * 128))
        self.model.compile(timesfm.ForecastConfig(
            max_context=max(32, int(math.ceil(context / 32) * 32)), max_horizon=self.max_h, normalize_inputs=True,
            use_continuous_quantile_head=True, force_flip_invariance=True, infer_is_positive=True,
            fix_quantile_crossing=True))

    def __call__(self, contexts, horizon, **_):
        _pt, qf = self.model.forecast(horizon=horizon, inputs=[np.asarray(c, np.float32) for c in contexts])
        return np.asarray(qf, float)[:, :horizon, 1:10]


class _KronosAdapter:
    """Kronos (github.com/shiyu-coder/Kronos, not on PyPI). README API:
        tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
        model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
        predictor = KronosPredictor(model, tokenizer, device=..., max_context=512)
        pred_df = predictor.predict(df=x_df[open,high,low,close(,volume,amount)], x_timestamp=Series,
                                    y_timestamp=Series, pred_len=H, T=1.0, top_p=0.9, sample_count=1)
    Kronos samples paths (it averages `sample_count` internally), so quantiles come from `samples`
    independent single-sample calls. Thin and isolated on purpose: only this class touches the repo."""

    def __init__(self, repo: str, tok_repo: str, max_context: int, device: str, ohlc: pd.DataFrame,
                 samples: int = 20, seed: int = 7):
        kdir = _kronos_dir()
        if kdir is None:
            raise ImportError("Kronos not found. Install: " + TSFM_INSTALL["kronos"])
        if str(kdir) not in sys.path:
            sys.path.insert(0, str(kdir))
        from model import Kronos, KronosPredictor, KronosTokenizer  # type: ignore  # noqa: E402
        tok = KronosTokenizer.from_pretrained(tok_repo)
        mdl = Kronos.from_pretrained(repo)
        self.pred = KronosPredictor(mdl, tok, device=device, max_context=max_context)
        self.ohlc = ohlc
        self.samples, self.seed = samples, seed

    def __call__(self, contexts, horizon, ends=None, **_):
        import torch
        cols = [c for c in ("open", "high", "low", "close", "volume") if c in self.ohlc]
        dfs, xts, yts = [], [], []
        for c, e in zip(contexts, ends):
            w = self.ohlc.iloc[e - len(c):e]
            stamps = w.index.tz_localize(None)
            dfs.append(w[cols].reset_index(drop=True).astype(float))
            xts.append(pd.Series(stamps))
            yts.append(pd.Series(stamps[-1] + BAR * np.arange(1, horizon + 1)))
        paths = np.zeros((len(dfs), self.samples, horizon))
        same_len = len({len(d) for d in dfs}) == 1
        for s in range(self.samples):
            torch.manual_seed(self.seed + s)
            if same_len and hasattr(self.pred, "predict_batch"):
                outs = self.pred.predict_batch(df_list=dfs, x_timestamp_list=xts, y_timestamp_list=yts,
                                               pred_len=horizon, T=1.0, top_p=0.9, sample_count=1, verbose=False)
            else:
                outs = [self.pred.predict(df=d, x_timestamp=x, y_timestamp=yy, pred_len=horizon, T=1.0, top_p=0.9,
                                          sample_count=1, verbose=False) for d, x, yy in zip(dfs, xts, yts)]
            for i, o in enumerate(outs):
                paths[i, s] = np.asarray(o["close"], float)[:horizon]
        return np.quantile(paths, QUANTILES, axis=1).transpose(1, 2, 0)    # [N, H, Q]


def _load_tsfm(model: str, device: str, context: int, ohlc: Optional[pd.DataFrame], kronos_samples: int):
    kind, spec = TSFM_MODELS[model]
    if kind in ("chronos", "chronos2"):
        if importlib.util.find_spec("chronos") is None or (kind == "chronos2" and _chronos_major() < 2):
            raise ImportError(f"{model} needs chronos-forecasting. Install: {TSFM_INSTALL[kind]}")
        return _ChronosAdapter(spec, device)
    if kind == "timesfm":
        if importlib.util.find_spec("timesfm") is None:
            raise ImportError(f"{model} needs timesfm 2.5 (GitHub). Install: {TSFM_INSTALL['timesfm']}")
        return _TimesFMAdapter(spec, context)
    if kind == "kronos":
        if _kronos_dir() is None:
            raise ImportError(f"{model} needs the Kronos repo. Install: {TSFM_INSTALL['kronos']}")
        repo, tok, maxc = spec
        return _KronosAdapter(repo, tok, min(maxc, context), device if device != "auto" else resolve_device(),
                              ohlc, samples=kronos_samples)
    raise ValueError(model)


def _p_up(qret: np.ndarray) -> np.ndarray:
    """P(return > 0) from quantiles [N, 9] at QUANTILES: piecewise-linear CDF inside q10..q90, a normal
    tail (mu = median, sigma from q10/q90) outside."""
    lv = np.asarray(QUANTILES)
    out = np.full(len(qret), np.nan)
    for i, q in enumerate(qret):
        if not np.all(np.isfinite(q)):
            continue
        q = np.sort(q)
        if q[0] <= 0 <= q[-1] and q[-1] > q[0]:
            cdf = float(np.interp(0.0, q, lv))
        else:
            sig = max((q[-1] - q[0]) / 2.5631, 1e-9)
            cdf = 0.5 * math.erfc(-(0.0 - q[4]) / sig / math.sqrt(2))
        out[i] = 1.0 - cdf
    return np.clip(out, 0.001, 0.999)


def _tsfm_name(model: str) -> str:
    return "tsfm_" + model.replace("-", "_").replace(".", "")


def tsfm_features(close_5m: pd.Series, decision_ts: pd.DatetimeIndex, model: str = "chronos-bolt-small",
                  horizon_bars: int = 6, context: int = 512, batch: int = 256, cache_dir=None, *,
                  to_close: bool = True, ohlc: Optional[pd.DataFrame] = None, device: str = "auto",
                  predictor: Optional[Callable] = None, use_cache: bool = True, kronos_samples: int = 20,
                  min_context: int = 32) -> pd.DataFrame:
    """Zero-shot TSFM forecasts as FEATURES for each decision time T (never the final model).

    Context for T = closes of the last `context` bars with start + 5min <= T (strictly causal; the
    bar starting at T is excluded). Forecasts are price levels; features are log-return quantiles
    versus the close at T:
      tsfm_<name>_ret_med_<h>, _q10_<h>, _q90_<h>, _width_<h> (q90 - q10), _p_up_<h>
    for h = `5*horizon_bars` minutes (e.g. "30m"), and with to_close=True the same with suffix "close"
    for the per-row horizon "T to 15:15". Rows with < `min_context` bars of history are NaN.

    Results are cached on disk (csv.gz per model/horizon/context, file stamped with the last decision
    time); because each row depends only on bars closed by its T, cached rows are reused across calls
    and only new decision times are forecast. `predictor` (callable(contexts: list[np.ndarray],
    horizon: int, ends=...) -> [N, horizon, 9] price quantiles at QUANTILES) can be injected for tests
    or custom models. `ohlc` (OHLC(V) frame on the same bars) is used by Kronos; otherwise flat bars.
    Raises ImportError with the install command if the model's package is missing."""
    if model not in TSFM_MODELS and predictor is None:
        raise ValueError(f"unknown TSFM {model}; choose from {list(TSFM_MODELS)}")
    s = close_5m.copy()
    s.index = _ist_index(s.index)
    s = _session_filter(s[~s.index.duplicated(keep="last")].sort_index().to_frame("close"))["close"].dropna()
    T = _ist_index(decision_ts)
    name = _tsfm_name(model)
    hz = f"{5 * horizon_bars}m"
    suffixes = [hz] + (["close"] if to_close else [])
    cols = [f"{name}_{k}_{sfx}" for sfx in suffixes for k in ("ret_med", "q10", "q90", "width", "p_up")]

    # cache
    cached = pd.DataFrame(columns=cols)
    cdir, files = None, []
    if use_cache:
        if cache_dir is None:
            from .config import DATA_DIR
            cache_dir = Path(DATA_DIR) / "cache" / "tsfm"
        cdir = Path(cache_dir)
        cdir.mkdir(parents=True, exist_ok=True)
        stem = f"{name}_h{horizon_bars}_c{context}{'_close' if to_close else ''}_"
        files = sorted(glob.glob(str(cdir / (stem + "*.csv.gz"))))
        parts = []
        for f in files:
            try:
                d = pd.read_csv(f, index_col=0)
                d.index = _ist_index(d.index)
                parts.append(d)
            except Exception as e:  # noqa: BLE001
                log.warning("ignoring unreadable TSFM cache %s: %s", f, e)
        if parts:
            cached = pd.concat(parts)
            cached = cached[~cached.index.duplicated(keep="last")]
    uniq = T.unique().sort_values()
    todo = uniq[~uniq.isin(cached.index)] if len(cached) else uniq
    new = pd.DataFrame(np.nan, index=todo, columns=cols)

    if len(todo):
        vals = s.to_numpy(float)
        ends = _closed_by(s.index, todo)
        minutes = np.asarray(todo.hour * 60 + todo.minute)
        h_close = (EXIT_MIN - minutes) // 5                  # bars from T to 15:15
        ok = ends >= min_context
        rows = np.flatnonzero(ok)
        if len(rows):
            if predictor is None:
                ohlc_k = None
                if TSFM_MODELS[model][0] == "kronos":
                    if ohlc is not None:
                        ohlc_k = ohlc.copy()
                        ohlc_k.index = _ist_index(ohlc_k.index)
                        ohlc_k = ohlc_k[~ohlc_k.index.duplicated(keep="last")].sort_index().reindex(s.index)
                        ohlc_k.columns = [str(c).lower() for c in ohlc_k.columns]
                    else:
                        ohlc_k = pd.DataFrame({"open": s, "high": s, "low": s, "close": s})
                predictor = _load_tsfm(model, resolve_device(device) if device == "auto" else device, context,
                                       ohlc_k, kronos_samples)
            t0 = time.perf_counter()
            for b0 in range(0, len(rows), batch):
                rb = rows[b0:b0 + batch]
                eb = ends[rb]
                ctxs = [vals[max(0, e - context):e] for e in eb]
                hb = int(horizon_bars)
                hc = h_close[rb] if to_close else np.zeros(len(rb), int)
                H = max(hb, int(hc.max()) if to_close and (hc > 0).any() else hb)
                q = _stack_quantiles(predictor(ctxs, H, ends=eb))
                if q.shape != (len(rb), H, len(QUANTILES)):
                    raise ValueError(f"predictor returned {q.shape}, expected {(len(rb), H, len(QUANTILES))}")
                last = vals[eb - 1][:, None]
                with np.errstate(all="ignore"):
                    qr = np.log(np.where(q > 0, q, np.nan) / last[:, :, None])   # [n, H, Q]
                blocks = [(hz, np.full(len(rb), hb - 1))]
                if to_close:
                    blocks.append(("close", np.where(hc > 0, hc - 1, -1)))
                for sfx, step in blocks:
                    sel = np.full((len(rb), len(QUANTILES)), np.nan)
                    good = step >= 0
                    sel[good] = qr[np.flatnonzero(good), step[good]]
                    sel = np.sort(sel, axis=1)
                    new.iloc[rb, new.columns.get_loc(f"{name}_ret_med_{sfx}")] = sel[:, 4]
                    new.iloc[rb, new.columns.get_loc(f"{name}_q10_{sfx}")] = sel[:, 0]
                    new.iloc[rb, new.columns.get_loc(f"{name}_q90_{sfx}")] = sel[:, -1]
                    new.iloc[rb, new.columns.get_loc(f"{name}_width_{sfx}")] = sel[:, -1] - sel[:, 0]
                    new.iloc[rb, new.columns.get_loc(f"{name}_p_up_{sfx}")] = np.where(good, _p_up(sel), np.nan)
            log.info("%s: %d decision times forecast in %.1fs", model, len(rows), time.perf_counter() - t0)

    if len(cached) and len(new):
        allrows = pd.concat([cached.reindex(columns=cols).astype(float), new])
    else:
        allrows = new if len(new) else cached.reindex(columns=cols)
    allrows = allrows[~allrows.index.duplicated(keep="last")].sort_index()
    if use_cache and len(new) and len(allrows):
        stamp = allrows.index.max().strftime("%Y%m%dT%H%M")
        h8 = hashlib.sha1(np.asarray(_ns(allrows.index)).tobytes()).hexdigest()[:8]
        path = cdir / f"{stem}{stamp}_{h8}.csv.gz"
        allrows.to_csv(path)
        for f in files:                                      # consolidated into the new file
            if Path(f) != path:
                with contextlib.suppress(OSError):
                    os.remove(f)
    out = allrows.reindex(T)
    out.index.name = "ts"
    return out.astype(float)


# ============================================================================ capabilities
def available_members() -> Dict[str, bool]:
    """What can run in this environment: torch + device, each sequence arch, each TSFM."""
    has_torch = importlib.util.find_spec("torch") is not None
    res: Dict[str, bool] = {"torch": has_torch, "cuda": False, "mps": False}
    if has_torch:
        try:
            import torch
            res["cuda"] = bool(torch.cuda.is_available())
            mps = getattr(torch.backends, "mps", None)
            res["mps"] = bool(mps is not None and mps.is_available())
        except Exception:  # noqa: BLE001
            res["torch"] = has_torch = False
    for a in ARCHS:
        res[a] = has_torch
    chronos = importlib.util.find_spec("chronos") is not None
    res["chronos-bolt-small"] = res["chronos-bolt-base"] = chronos and has_torch
    res["chronos-2"] = chronos and has_torch and _chronos_major() >= 2
    res["timesfm-2.5"] = importlib.util.find_spec("timesfm") is not None and has_torch
    res["kronos-small"] = res["kronos-base"] = _kronos_dir() is not None and has_torch
    return res


__all__ = ["build_panel", "build_sequences", "sequence_channels", "SeqMember", "walk_forward_member",
           "tsfm_features", "available_members", "resolve_device", "ARCHS", "TSFM_MODELS", "QUANTILES"]
