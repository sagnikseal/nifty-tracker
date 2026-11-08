# Intraday model zoo

`nifty_engine/intraday_zoo.py` adds new members for the 5-minute NIFTY direction problem (`y30`,
`yclose`) in `intraday_model.py`. It follows the same rules: a decision at T uses only bars whose
start + 5 min <= T, walk-forward is by whole trading days, and the main engine still does
calibration (Platt on earlier OOS predictions), member selection (trailing OOS log-loss) and
day-clustered tests.

**Expectations.** So far the logistic and LightGBM members are about 50-51% accurate out of sample,
with no log-loss skill. Nothing below is expected to change that. The zoo gives the walk-forward
*different* hypotheses: the raw path shape, cross-asset lead/lag (VIX, BankNifty, futures), and the
priors of pretrained models. Every member here is small and heavily regularised. If none beats the
base rate on trailing log-loss, the selector keeps choosing `base`, and that is the right outcome.

## Inputs

* `build_panel({"nifty": df, "vix": df, "banknifty": df, "niftyfut": df})` aligns 5-minute bars into
  one panel (`nifty_close`, `vix_close`, ...). Gaps are filled only from earlier bars of the same
  session, never across days. Volume is never forward-filled.
* `build_sequences(panel, decision_ts, lookback=48)` returns `[N, 48, C]` float32 windows of the
  bars closed by T. Channels:
  * log returns, high/low ranges and the body (close/open)
  * normalised volume, only for series that report it (the index CSVs have zero volume)
  * VIX changes
  * time-of-day sin/cos
  * session-start flag, `is_today`, and a padding `mask`

  Return and range channels are rescaled within each window (`scale="sample"`, RevIN-style).
  `scale="ewm"` instead divides by a causal one-session EWM, which keeps the volatility regime
  visible.
* Tests check causality directly. They perturb every bar after T, check that the bar starting at T
  is excluded, and check that the bar closing at T is the last element of the window.

## Members (`SeqMember(arch=...)`)

| arch | What it is | Why it is in the zoo | Params (C=12, hidden 32) | CPU cost here* |
|---|---|---|---|---|
| `tcn` | Dilated causal 1-D convs (k=3, dilations 1-8, residual), state at last bar | Multi-scale local patterns, receptive field 61 bars | 26.4k | 540 ms |
| `gru` | 1-layer GRU, last hidden state | Recurrent baseline with a different inductive bias | 5.6k | 140 ms |
| `lstm` | 1-layer LSTM | Same as GRU with gated memory; cheap diversity | 7.0k | 110 ms |
| `patchtst` | Patches of 8 bars (stride 4), linear embedding + learned positions, 2-layer pre-norm Transformer (d=32, 4 heads), mean-pool | Attention over ~40-minute patches; SOTA family for long-horizon forecasting | 21.7k | 180 ms |
| `cnn` | Two 1-D conv layers, mean + last pooling | Cheapest non-linear sequence baseline | 9.2k | 76 ms |

\* Milliseconds per 1,000 training rows per epoch, for one seed, measured on this Windows CPU.

All members are regularised the same way:
* dropout 0.3, AdamW (weight decay 1e-3), gradient clipping, plain BCE
* early stopping on the **last 15% of training days** (time-ordered; validation never uses random
  rows), with the best-epoch weights restored
* an average over `seeds=3` nets

You can pass tabular features (for example the intraday feature frame) with `fit(..., tab=X)`. They are
standardised with training-part statistics only and concatenated after the encoder. `quick=True`
gives 1 seed, 3 epochs and hidden 8, for tests. `fit` records `val_logloss_` vs
`val_logloss_base_`, so you can see in each fold whether a net beat the base rate on held-out days.

**Cost on a 16 GB M5 MacBook Air (estimate, not measured).** The full problem is about 62k decision
rows x 48 x ~12 channels, which is about 140 MB as float32 and fits easily in memory. One refit is
about 35k training rows on average x ~8 epochs (early stopping) x 3 seeds. On MPS the expected fit
times are:
* CNN / GRU / LSTM / PatchTST: roughly 10-30 s per refit
* TCN: roughly 30-90 s per refit (it is the heaviest)

The full walk-forward has about 45 refits per horizon, so allow 10-60 minutes per arch per horizon.
To speed things up, use `seeds=2` or `hidden=24` for TCN. The Air is fanless, so long runs throttle.
Run one arch at a time.

## TSFM features (`tsfm_features`)

Pretrained forecasters run zero-shot on the closes known at T (the last `context` bars). Their
forecast distribution becomes **features**, not the final model:
* columns `tsfm_<name>_{ret_med,q10,q90,width,p_up}_30m`
* the same columns with suffix `_close`, for the per-row horizon T -> 15:15

Results are cached as csv.gz under `store/cache/tsfm/`. Each row depends only on bars before its T,
so later runs forecast only the new decision times.

| model | Package | Why | Cost on M5 (estimate, full history ~62k decisions) |
|---|---|---|---|
| `chronos-bolt-small` / `-base` | `chronos-forecasting` | T5 encoder, direct multi-step quantiles, very fast | small: minutes; base: ~4x |
| `chronos-2` | `chronos-forecasting>=2` | Newer Chronos with a group-attention architecture | about Bolt-base |
| `timesfm-2.5` | timesfm (GitHub) | Google decoder-only 200M model with a continuous quantile head | tens of minutes |
| `kronos-small` / `-base` | Kronos repo (`KRONOS_PATH`) | Pretrained on OHLCV K-lines from 45+ exchanges; finance-specific | **hours**: it samples autoregressively (`kronos_samples=20` paths). Use `to_close=False`, fewer samples, or every third decision time |

If a package is missing, the call raises `ImportError` with the exact install command (see
the optional section of `requirements.txt`). `available_members()` reports what can run here.

## How to run

The `tab` features should exclude the label and price columns (`feature_columns` does this).

```python
from nifty_engine import intraday_model as im, intraday_zoo as zoo
n5, v5, b5 = im.load_bars("nifty_5m"), im.load_bars("indiavix_5m"), im.load_bars("banknifty_5m")
X = im.build_frame(n5, v5)                                   # rows: index = decision time T
P = zoo.build_panel({"nifty": n5, "vix": v5, "banknifty": b5})
seq = zoo.build_sequences(P, X.index, lookback=48)
oos = zoo.walk_forward_member(seq, X["y30"].values, X["date"].values,
                              lambda: zoo.SeqMember("patchtst"),       # device auto -> mps on the Mac
                              tab=X[im.feature_columns(X)].values)     # optional
# raw OOS P(up) per row -> calibrate / select / test with intraday_model's Platt + evaluate()

feats = zoo.tsfm_features(P["nifty_close"], X.index, model="chronos-bolt-small")   # cached on disk
X = X.join(feats)        # then the existing logit/LightGBM members can use the TSFM columns
```

Tests: `python -m unittest tests.test_intraday_zoo -v` runs in about 5 s on CPU. The TSFM logic is
tested with an injected fake predictor. Real TSFM tests are skipped unless the package is installed.

## References

* Bai, Kolter, Koltun (2018). *An Empirical Evaluation of Generic Convolutional and Recurrent
  Networks for Sequence Modeling* (TCN). arXiv:1803.01271.
* Cho et al. (2014), GRU; Hochreiter & Schmidhuber (1997), LSTM.
* Nie et al. (2023). *A Time Series is Worth 64 Words: Long-term Forecasting with Transformers*
  (PatchTST), ICLR. arXiv:2211.14730.
* Kim et al. (2022). *Reversible Instance Normalization for Accurate Time-Series Forecasting
  against Distribution Shift* (RevIN), ICLR.
* Ansari et al. (2024). *Chronos: Learning the Language of Time Series*. arXiv:2403.07815.
  Chronos-Bolt and Chronos-2: `amazon/chronos-bolt-*` and `amazon/chronos-2` model cards,
  https://github.com/amazon-science/chronos-forecasting
* Das et al. (2024). *A decoder-only foundation model for time-series forecasting* (TimesFM), ICML.
  arXiv:2310.10688. https://github.com/google-research/timesfm
* Shi et al. (2025). *Kronos: A Foundation Model for the Language of Financial Markets*.
  https://github.com/shiyu-coder/Kronos
