# 📈 NIFTY 50 Daily High-Low Tracker & Trading Journal

An automated **GitHub Pages** web application that tracks **NIFTY 50 Daily High, Low, Range, CPR, and Candlestick metrics** after the National Stock Exchange (NSE) closes each day, and provides a built-in interface to **mark trading observations, key levels, trade setups, and journal notes**.

Powered by automated **GitHub Actions** running on a weekday schedule (Monday to Friday after 3:30 PM IST).

---

## 🚀 Live Features

- **Daily High & Low Spotlight**: Prominently highlights today's High, Low, Range (in points and %), settled Close, and compares with previous day levels.
- **20-Day Average Daily Range (ADR)**: Compare today's volatility and expansion against the 20-day historical average.
- **52-Week Range Gauge**: Visual slider indicating where today's price sits within the 52-week band.
- **Next Session CPR & Pivots**: Automatically computes Central Pivot Range (TC, BC, Pivot), CPR width (Narrow / Average / Wide), and classic support/resistance levels ($R_1$, $R_2$, $S_1$, $S_2$).
- **Interactive Candlestick Chart**: Embedded TradingView Lightweight Charts canvas with time-scale zooming (1M, 3M, 6M, 1Y, All) and hover crosshairs.
- **Trading Info Marking & Journaling**:
  - Mark daily bias: 🟢 Bullish, 🔴 Bearish, 🟡 Range / Chop, ⚡ High Volatility.
  - Tag price action patterns: *Trend Day, Inside Bar, Outside Bar, Gap & Go, CPR Bounce, Fake Breakout, Expiry Move*.
  - Log traded levels (PDH, PDL, CPR, round numbers) and trade execution outcomes (Profit, Loss, Breakeven, Observer).
  - Add freeform psychology notes and custom `#tags`.
  - Stored persistently in browser `localStorage`.
  - Export & Import journal backups anytime (`.json`).
  - Marked days appear with custom badges on the candlestick chart and in the table!
- **Data Table**: Filterable by "All Days", "My Marks Only", or "Inside Bars", with search and CSV download.
- **Theme Toggle**: Switch between Dark and Light trading themes.

---

## 🛠️ Project Structure

```
Trading/
├── .github/
│   └── workflows/
│       └── update_nifty.yml       # GitHub Actions: Mon-Fri scheduled run & deployment
├── data/
│   └── nifty_data.json            # Historical & latest OHLC + pivots data (pre-populated)
├── scripts/
│   └── fetch_nifty.py             # Python script fetching ^NSEI & calculating metrics
├── index.html                     # Responsive web interface
├── style.css                      # Trading terminal styling (dark/light theme)
├── app.js                         # Application logic, chart, table & journal state
├── README.md                      # Documentation & deployment guide
└── .gitignore                     # Git ignore rules
```

---

## ⚙️ GitHub Actions Automation

The workflow file [`.github/workflows/update_nifty.yml`](.github/workflows/update_nifty.yml) runs automatically:

- **Schedule**:
  - `30 10 * * 1-5`: Runs at **10:30 UTC** (**4:00 PM IST**), 30 minutes after NSE market closes at 3:30 PM IST.
  - `15 11 * * 1-5`: Runs at **11:15 UTC** (**4:45 PM IST**) to ensure final post-market settlement prices are recorded.
- **Manual Trigger**: Can be run anytime on demand via the **Run workflow** button in GitHub Actions.
- **Auto-Commit**: Automatically commits updated `data/nifty_data.json` to the repo with `[skip ci]`.
- **Pages Deploy**: Deploys the freshly updated site directly to GitHub Pages.

---

## 🚢 How to Deploy to Your GitHub Pages (Step-by-Step)

### Step 1: Initialize & Commit Locally (Already configured)

If you haven't committed yet:
```bash
git add .
git commit -m "Initial commit: NIFTY Daily Tracker & Trading Journal"
```

### Step 2: Create a New GitHub Repository

1. Open [GitHub](https://github.com/new).
2. Create a new repository (e.g. `nifty-daily-tracker` or `Trading`).
3. Set it to **Public** (or Private if you have GitHub Pro for Pages).
4. Do not initialize with README or .gitignore (we already have them).

### Step 3: Link & Push to GitHub

In your local terminal inside this folder, run:
```bash
# Replace <YOUR-USERNAME> and <REPO-NAME> with your GitHub details
git remote add origin https://github.com/<YOUR-USERNAME>/<REPO-NAME>.git
git branch -M main
git push -u origin main
```

### Step 4: Configure GitHub Pages Settings

1. On GitHub, go to your repository **Settings** tab.
2. In the left sidebar, click **Pages** (under "Code and automation").
3. Under **Build and deployment**:
   - Change **Source** to **GitHub Actions**.

### Step 5: Grant Workflow Write Permissions

Because the workflow commits the newly fetched daily data back into `data/nifty_data.json`, enable write permissions:
1. In your repository **Settings**, go to **Actions** -> **General** (in left sidebar).
2. Scroll down to **Workflow permissions**.
3. Select **Read and write permissions**.
4. Check **Allow GitHub Actions to create and approve pull requests** (optional, recommended).
5. Click **Save**.

### Step 6: Test Your First Workflow Run

1. Go to the **Actions** tab in your repository.
2. In the left sidebar, click **Update NIFTY Data & Deploy to Pages**.
3. Click the **Run workflow** dropdown on the right and click the green **Run workflow** button.
4. Once completed (approx. 40 seconds), your site will be live at:
   `https://<YOUR-USERNAME>.github.io/<REPO-NAME>/`

---

## 💻 Local Development

You can run and test the app locally anytime without any external dependencies:

```bash
# Run the Python data fetcher locally (fetches latest NIFTY data)
python3 scripts/fetch_nifty.py

# Start a simple local server to preview the web app
python3 -m http.server 8000
```
Then visit `http://localhost:8000` in your web browser.

---

## 📝 License

MIT License. Built for index traders and market participants.
