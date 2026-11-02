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
