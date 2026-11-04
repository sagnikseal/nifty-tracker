# NIFTY 50 Tracker: v5 engine (analytics, local-LLM news, multi-strategy paper trading, research harness)

> **New in v5:** see [`docs/V5_ENGINE.md`](docs/V5_ENGINE.md). It covers SMC / technical / option-chain analytics, related global markets, news read by a free local LLM (Ollama), nine options strategy books paper-traded side by side, and a pre-registered research harness that tests every idea on real expired-option prices. Everything runs free on your Mac.
>
> **Background:** [`docs/UPGRADE_PLAN_V4.md`](docs/UPGRADE_PLAN_V4.md). The v3 model's 97% accuracy came from label leakage in `scripts/build_real_training_data.py`. v4 replaced it with a leakage-tested, walk-forward engine.

## v5 quick start (Mac)

```bash
pip install -r requirements-engine.txt
brew install ollama && ollama pull llama3.1:8b      # free local LLM for news (optional; lexicon fallback)
cp .env.example .env                                # set UPSTOX_ANALYTICS_TOKEN
python -m nifty_engine update-data && python -m nifty_engine fetch-intraday && python -m nifty_engine train
python -m nifty_engine research                     # every experiment -> store/reports/RESEARCH_REPORT.md
python -m nifty_engine run                          # recorder + paper books + dashboard at http://localhost:8010/paper
```

## v4 quick start

```bash
pip install -r requirements-engine.txt
python -m nifty_engine all          # download data (2007-), walk-forward train, backtest report
python -m nifty_engine run          # auto paper trader + dashboard at http://localhost:8010/paper
```

Or run it anywhere with Docker: `docker compose up -d`. On Windows without Docker, double-click `run_engine.bat`.

- **Broker independent.** Signals use only free daily data: NIFTY, India VIX, US/Asia indices, FX, crude, yields. Upstox is an optional adapter for live option bid/ask. Put `UPSTOX_ACCESS_TOKEN`, or `UPSTOX_API_KEY` + `UPSTOX_PIN` + `UPSTOX_TOTP_SECRET`, in `.env`.
- **Paper trading.** ₹1,00,000 by default, 1 lot ITM CE/PE, entry 09:20, exit by 15:15, full NSE charges (2026 STT). Edit `config/engine.json`.
- **Self-retraining.** Weekly champion/challenger walk-forward. Every model version is kept in `store/models/`.
- **Tests:** `python -m unittest discover -s tests`. These include tests proving features can't see the future.

### Full model ensemble on your Mac / GPU desktop

```bash
pip install -r requirements-engine.txt -r requirements-zoo.txt
# NVIDIA GPU (Windows/Linux): reinstall torch with CUDA, e.g.
#   pip install torch --index-url https://download.pytorch.org/whl/cu128
python -m nifty_engine update-data
python -m nifty_engine zoo --fresh --promote   # 12 models, walk-forward, deploys best-3 if it beats the champion
```

Apple Silicon (M-series) uses the GPU through MPS automatically, and NVIDIA uses CUDA. The Transformer member is only enabled when a GPU is present (on a laptop CPU it takes 40+ minutes). Trained models live in `store/models/`. Copy that folder to the machine that runs the paper trader.

---

## Legacy v3 terminal (kept for reference)

# NIFTY 50 Institutional Tracker & 4-Pillar AI Prediction Terminal (Overhaul v2.0)

An institutional-grade intraday prediction engine, options microstructure analyzer, cross-asset macro radar, geopolitical shock scanner, and Smart Money Concepts (SMC) End-of-Day tracker for **NSE NIFTY 50 (`^NSEI`)**.

---

## What Was Fixed & Overhauled in v2.0

### 1. Eliminated the `CONNECTING...` Freeze Permanently
- **Non-Blocking Async Background Worker (`scripts/realtime_engine.py`)**: External broker, macro, and RSS requests now run in a background worker thread (`asyncio.to_thread`) every 1.0s rather than blocking FastAPI's `/ws/stream` and `/api/signal` handlers. Responses now return in **<5ms**.
- **Proper URL-Encoding & Intraday Endpoints for Upstox (`UpstoxProvider`)**: Properly encodes `NSE_INDEX|Nifty 50` (`NSE_INDEX%7CNifty%2050`), targets the Upstox v2 intraday endpoint (`/v2/historical-candle/intraday/.../1minute`), and automatically calculates the next Thursday weekly expiry date for `/v2/option/chain`.
- **Complete `AngelOneProvider` & `GrowwProvider` Implementation**: Implements AngelOne SmartAPI live quotes (`99926000` NIFTY 50 & `99926017` India VIX) and historical intraday candles.
- **Auto-Sync Credentials & Explicit State Machine (`realtime_client.js`)**: Automatically pushes saved `localStorage` broker credentials to `POST /api/admin/configure` on page load, enforces a 2.2s connection watchdog, and seamlessly activates the built-in **In-Browser 4-Pillar Quant Engine** when viewing on static GitHub Pages without a local Python server.
- **Strict Live-Only Safety Lock (`block_synthetic_trades`)**: Optional toggle that locks the engine to `NO_TRADE_CASH (SYNTHETIC_FEED_LOCKED)` whenever your broker token expires or falls back to modeled option chains.

### 2. The 4-Pillar Calibrated AI Ensemble (`scripts/train_ml_model.py` & `models/nifty_ensemble_weights.json`)
Predictions combine **16 engineered features** across four pillars using an L2-regularized Logistic + Gradient Residual Stump ensemble calibrated via **Pool-Adjacent-Violators (PAVA) Isotonic Regression**:

| Pillar | Weight | Key Signals & Market Drivers |
| :--- | :--- | :--- |
| **1. Technical, SMC & Volume VWAP** | **28%** | Multi-timeframe (1m/5m/15m) Supertrend alignment, Futures/Proxy Volume-Weighted VWAP Z-Score, Central Pivot Range (TC/Pivot/BC width), PDH/PDL Liquidity Sweeps, 3-Candle Fair Value Gaps (FVG), Break of Structure (BOS), and Opening Range Breakout (ORB). |
| **2. Options Microstructure & Top-6 Heavyweights** | **34%** | True analytical Black-Scholes Gamma & Dealer Net GEX (in ₹ Cr), Zero-Gamma Flip strike, Call/Put OI Walls, 15-minute rolling ring buffer for 5m PCR Velocity, Trapped Call/Put Writer Short-Covering Panic detector, and **Top 6 NIFTY Heavyweights Basket** (`HDFCBANK`, `RELIANCE`, `ICICIBANK`, `INFY`, `LT`, `TCS` — ~42% of NIFTY weight) with automatic divergence veto. |
| **3. Global Macro & Economic Cross-Asset Shocks** | **18%** | Live cross-asset tracking of **Brent Crude Oil** (`$/bbl`), **USD/INR** spot, **US Dollar Index (DXY)**, **US 10Y Treasury Yield**, **S&P 500 E-Mini Futures**, **GIFT Nifty** spread, and **India VIX** level + velocity regime switch. |
| **4. Geopolitical Conflict & Live News NLP Wire** | **20%** | Multi-feed live RSS parser (*Economic Times*, *Moneycontrol*, *LiveMint*) + custom wire injection scored by a dual **Geopolitical War/Conflict Escalation Lexicon** (war, missile, airstrike, blockade, sanctions, Red Sea/Hormuz/border tensions) and **Central Bank / Macro Policy NLP** (RBI MPC, Fed FOMC, CPI, GDP, FII flows). |

---

## Repository Structure

```text
nifty-tracker/
├── .github/workflows/
│   └── update_nifty.yml            # Daily EOD sync, ML calibration & GitHub Pages deploy
├── data/
│   ├── nifty_data.json             # 248-session NIFTY 50 EOD OHLC, SMC & CPR dataset
│   └── trading_marks.json          # Persistent trade journal annotations
├── models/
│   └── nifty_ensemble_weights.json # Calibrated 4-Pillar ML weights & isotonic knots
├── scripts/
│   ├── fetch_nifty.py              # Daily EOD NIFTY 50 fetcher & SMC/CPR calculator
│   ├── train_ml_model.py           # Triple-Barrier ML trainer & PAVA isotonic calibrator
│   ├── realtime_engine.py          # FastAPI + WebSocket 4-Pillar realtime server
│   └── backtest_stress_test.py     # Multi-regime walk-forward & war/macro stress tester
├── index.html                      # Institutional 4-Pillar AI & EOD SMC Terminal UI
├── style.css                       # Responsive dark/light theme styling
├── app.js                          # EOD SMC/CPR cards, interactive OHLC canvas & journal
├── realtime_client.js              # WebSocket/HTTP client & In-Browser 4-Pillar engine
├── requirements.txt                # Python dependencies
└── .env.example                    # Template for Upstox / AngelOne / Groww tokens
```

---

## Quick Start: Replace Your Local Clone & Push to GitHub

1. **Extract the Overhaul Zip into Your Local Clone**:
   ```bash
   cd /path/to/your/nifty-tracker
   unzip -o ~/Downloads/nifty-tracker-overhaul.zip -d .
   ```

2. **Install Python Dependencies**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate   # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Verify the Calibrated ML Model & Walk-Forward Stress Tests**:
   ```bash
   python scripts/fetch_nifty.py
   python scripts/train_ml_model.py
   python scripts/backtest_stress_test.py
   ```

4. **Launch the Live 4-Pillar Realtime Backend Engine**:
   ```bash
   cp .env.example .env
   # Edit .env to add your UPSTOX_ACCESS_TOKEN or ANGELONE_ACCESS_TOKEN
   python scripts/realtime_engine.py
   ```
   The engine starts on `http://localhost:8000` (WebSocket stream at `ws://localhost:8000/ws/stream`, health check at `http://localhost:8000/api/health`).

5. **Open the Trading Terminal**:
   - Serve the directory locally (`python -m http.server 3000` and visit `http://localhost:3000`) or open `index.html` directly.
   - Click **⚙ Broker API & Feeds** in the top navigation bar to paste your daily broker token, test backend latency, or toggle the **Strict Live-Only Lock**.

6. **Commit & Push to Your GitHub Repository**:
   ```bash
   git add .
   git commit -m "feat: complete v2.0 overhaul with 4-Pillar AI engine, macro/geopolitical NLP, and broker streaming fix"
   git push origin main
   ```
