# Data collection with GitHub Actions

> **Off by default.** Both workflows run only after you set the repository variable
> `COLLECT_ENABLED` to `true` (Settings → Secrets and variables → Actions → Variables).
>
> `sagnikseal/nifty-tracker` is a **public** repository, and a `data` branch inside it would be public too.
> Before enabling, set `DATA_REPO` to a **private** repository and `DATA_REPO_TOKEN` to a fine-grained token
> with Contents read/write on that repository. Exchange and broker data may not be redistributed publicly.
>
> Actions minutes stay free, because they are billed to the public code repository.

Some market data can only be captured while it is happening: option-chain snapshots (OI, bid/ask,
IV through the day), quotes with order-book depth, and news headlines with the time they first
appeared. If the Mac is off, that data is gone. Two scheduled GitHub Actions workflows collect it
every market day and store it on a separate **`data` branch** of the repository (never on `main`).
`python -m nifty_engine.collect sync` pulls it into the local store in the formats the engine
already reads.

The local live recorder (`python -m nifty_engine run` / `record`) is still better when the Mac is
on: it polls the chain every 30 s and streams 30-level depth over WebSocket. The Actions collector
fills the gaps (every 15 min) and catches news overnight and at weekends.

## What is collected

| Job | When (IST) | What | Source |
|---|---|---|---|
| `snapshot` | every 15 min, 09:16-15:31, Mon-Fri | Full option chain of the nearest 2 NIFTY expiries (LTP, bid/ask and quantities, OI, previous OI, volume, IV, greeks, spot) plus derived chain metrics (PCR, walls, max pain, ATM IV, straddle, skew, GEX, OI build-up vs the previous snapshot); full quotes with 5-level depth for NIFTY, India VIX, Bank Nifty, the near-month NIFTY future and the 15 heavyweights; one news poll | Upstox (Analytics Token), RSS |
| `news` | every 15 min at all other times (overnight, pre-open, weekends) | Headlines from all free RSS/Atom sources in `data/news_sources.py` (plus `config/news_sources.json`), Upstox News for the heavyweights, stamped with `first_seen` | RSS, Upstox |
| `eod` | 16:10 and again at 19:10, Mon-Fri | Today's 1- and 5-minute candles (with volume and OI where available) for NIFTY, India VIX, Bank Nifty, the near-month NIFTY future and each heavyweight. On/after a weekly expiry: 1-minute candles of that expiry's options, strikes ATM ± 800 (step 50, CE and PE = 66 contracts) for the 8 days up to expiry. A final news poll. | Upstox (expired options need **Upstox Plus**) |

The collector skips snapshots outside 09:15-15:45 IST and on holidays (it checks whether any
instrument traded today). News is collected on every run. The second EOD run is a safety net: all
writes are idempotent, so nothing is stored twice.

### Schedules (cron is UTC; IST = UTC + 5:30)

| Workflow | Cron (UTC) | IST |
|---|---|---|
| `collect_market.yml` (snapshot) | `46 3 * * 1-5` | 09:16 Mon-Fri |
| | `1,16,31,46 4-9 * * 1-5` | 09:31, 09:46 ... 15:16 Mon-Fri |
| | `1 10 * * 1-5` | 15:31 Mon-Fri |
| `collect_market.yml` (news) | `7,22,37,52 0-3,10-23 * * 1-5` | weekdays (UTC) 05:37-09:22 and 15:37-05:22 |
| | `7,22,37,52 * * * 0,6` | Saturday and Sunday (UTC), around the clock |
| `collect_eod.yml` | `40 10 * * 1-5` | 16:10 Mon-Fri |
| | `40 13 * * 1-5` | 19:10 Mon-Fri (catch-up) |

The minutes are offset from :00/:15/:30/:45 because GitHub is busiest at the top of the hour.
GitHub still starts scheduled runs late, usually by 3-20 minutes and sometimes more. The collector
records the time the data was actually fetched.

## Storage on the `data` branch

```
MANIFEST.json                         days, sizes, last run of each job, expired-option status
README.md
meta/instruments.json                 instrument keys used today (future, stocks, expiries)
runs/YYYY-MM.jsonl                    one line per run: status, counts, errors (never the token)
days/YYYY-MM-DD/
    chain_<expiry>.csv.gz             option-chain snapshots: recorder CHAIN_COLS + instrument_key, close_price, pop, pcr
    chain_metrics.jsonl               one line per snapshot, same keys as the recorder (near_*, next_*, iv_term)
    quotes.csv.gz                     ts, key, OHLC, volume, OI, ..., bid1..5/ask1..5 price, qty, orders
    news.jsonl                        id, title, summary, source, url, published, first_seen
    candles/<series>_<1m|5m>.csv.gz   ts, open, high, low, close, volume, oi
    expired_options_1m.csv.gz         ts, open, high, low, close, volume, oi, strike, type, key (expiry days)
    expired_options_1m.json           ATM used, contracts wanted/got, complete flag
```

Series names: `nifty`, `indiavix`, `banknifty`, `niftyfut`, and the heavyweight symbols in lower
case without `&` (`hdfcbank`, `mm` for M&M), the same names as `store/intraday/`.
Timestamps are ISO 8601 in IST (`2026-11-10T09:31:04.123456+05:30`).

**How files are written.** Snapshot tables are appended as new gzip members that contain only
rows not already stored, so the earlier bytes never change and git stores each append as a small
delta. Candles and expired options are merged by key and rewritten only when the content changes.
JSONL files are appended by key. The workflows share one `concurrency` group, so only one run writes
to the branch at a time. If a push is still rejected (for example after a manual push), the run
resets to the new remote tip, merges its own data again and pushes; that is a conflict-free
version of fetch + rebase.

**Size.** These estimates come from synthetic data with noisy prices. Real data compresses
better, so treat them as an upper bound.

| Item | Per session | Per year (≈250 sessions, 52 expiries) |
|---|---|---|
| Option chains (2 expiries × 26 snapshots, 150-250 strikes each) | 1.0-1.6 MB | 250-400 MB |
| Quotes with depth (19 instruments × 26) | 0.1 MB | 25 MB |
| Candles (19 series, 1m + 5m) | 0.13 MB | 35 MB |
| News (~300-800 new headlines/day, every day) | 0.1-0.3 MB raw | 15-40 MB packed |
| Expired options 1m (66 contracts × ~6 sessions, weekly) | 3 MB per expiry | ~160 MB |
| **Total** | | **≈ 0.5-0.65 GB/year** |

The largest single file is about 3 MB, far below GitHub's 100 MB limit. GitHub recommends keeping
repositories under about 1-5 GB. After about two years, start a fresh branch or repository (for
example by setting `DATA_REPO` to a new repository) and keep the old one as an archive. CI never
downloads the whole branch: each run does a shallow, partial, sparse checkout of the last 6 days
(about 10-20 MB).

## Setup

1. **Add the Upstox token as a secret.** Generate an Analytics Token in the Upstox developer
   console. It is read-only, valid for one year, and needs no daily login. In GitHub, open the
   repository's *Settings → Secrets and variables → Actions → New repository secret* and add it as
   name `UPSTOX_ANALYTICS_TOKEN`. The workflows pass it only to the collect step as an environment
   variable, and the collector never prints it.
   - Without the secret, the workflows still succeed (exit 0) and collect free news only. Each run
     shows a "No Upstox token" warning.
   - If the token expires or is revoked (HTTP 401), the run stops calling Upstox, still collects
     news, exits 0, and shows an "Upstox token rejected" error annotation. Check the Actions page now
     and then, or the `last_runs` section of `MANIFEST.json` and `status`.
2. **Keep the data private (recommended).** This repository is **public**, so a `data` branch in
   it is public too. Upstox's API terms and NSE's data policies restrict redistributing exchange
   data, so publishing chains and quotes may breach them. To store the data in a private repository
   instead:
   - Create an empty private repository, for example `you/nifty-data`.
   - Create a fine-grained personal access token with access to that repository only, with
     *Contents: Read and write*.
   - In this repository, add the variable `DATA_REPO` = `you/nifty-data` (*Settings → Secrets and
     variables → Actions → Variables*) and the secret `DATA_REPO_TOKEN` = the token.
   - Actions minutes are still billed to this public repository, so they stay free. The `data`
     branch is created in the private repository on the first run.
3. **Enable and test.** Open *Actions → Collect market data → Run workflow*. Choose `news` (works at
   any time) or `snapshot` with *force* to test outside market hours. The first run creates the
   orphan `data` branch.

`permissions: contents: write` is set in both workflows so the default `GITHUB_TOKEN` can push to
the `data` branch. Commits are made as `github-actions[bot]` and include `[skip ci]`. Pushes made
with `GITHUB_TOKEN` do not trigger other workflows anyway.

## GitHub Actions minutes

GitHub bills each job rounded up to a whole minute. A news run takes under 1 minute, a snapshot
about 1-2 minutes (setup about 30 s, Upstox and RSS about 30-60 s), and an EOD run about 2-5
minutes (EOD runs that download expired options take longer).

| Schedule | Runs / month | Billed minutes / month (approx.) |
|---|---|---|
| Snapshot (26 per weekday) | ~565 | ~1,100 |
| News (72 per weekday, 96 per weekend day) | ~2,370 | ~2,400 |
| EOD (2 per weekday) | ~43 | ~170 |
| **Total (default)** | **~2,970** | **~3,700** |

- **Public repository** (as now): standard GitHub-hosted runners are **free and unlimited**, so the
  default schedule costs nothing.
- **Private repository:** GitHub Free includes 2,000 minutes/month and Pro 3,000, so the default
  schedule would exceed both. When the quota runs out, collection stops for the rest of the month
  unless you set a spending limit (Linux minutes cost a fraction of a cent each; see GitHub's Actions pricing page). For a private repository,
  use a lean schedule, about 1,300 minutes/month:
  ```yaml
  # collect_market.yml, snapshots every 30 min:
  - cron: '46 3 * * 1-5'
  - cron: '16,46 4-9 * * 1-5'
  - cron: '16 10 * * 1-5'
  # news: hourly overnight, every 15 min in the pre-open, every 2 h at weekends:
  - cron: '7 0-1,10-23 * * 1-5'
  - cron: '7,22,37,52 2-3 * * 1-5'
  - cron: '7 */2 * * 0,6'
  ```
  If you change the news crons, keep them starting with `7` so the job still recognises them as
  news runs; the "Choose mode" step matches on the minute field. Also drop the second EOD cron.
  Keeping the code repository public and only the data repository private (step 2 above) avoids
  the question entirely.

## Syncing to the Mac

```bash
python -m nifty_engine.collect sync      # fetch the data branch and merge it into store/
python -m nifty_engine.collect status    # what has been collected and synced
```

On the first run, `sync` makes a shallow clone of the data branch at `store/cache/data_branch`
(`git clone --depth 1 --single-branch --branch data <origin URL>`). After that it runs
`git fetch --depth 1` + `reset`. Your main checkout is never touched. Options:

- `--repo URL` or `NE_DATA_REPO=URL`: the repository that holds the data branch (default: this
  checkout's `origin`; use the private data repository's URL if you set `DATA_REPO`).
- `--data-branch-dir DIR`: read an existing checkout of the data branch as-is (no fetch).
- `--no-fetch`: use the cached clone without fetching.
- `--include-today`: also merge today's recorder files before 15:45 IST. By default they are left
  alone while the live recorder may still be writing them.
- `--full`: re-read every file (normally only files changed since the last sync are read).

Where the data goes (idempotent, so running sync twice changes nothing):

| Data branch | Local store | Rule |
|---|---|---|
| `days/D/chain_<e>.csv.gz` | `store/live/D/chain_<e>.csv.gz` | union with the recorder's rows (recorder wins on the same ts/strike/type), sorted by time, plain `.csv` folded into `.csv.gz` |
| `days/D/chain_metrics.jsonl` | `store/live/D/chain_metrics.jsonl` | union by timestamp, sorted |
| `days/D/quotes.csv.gz` | `store/live/D/quotes.csv.gz` and `depth.csv.gz` | quotes kept in full; top-5 depth also converted to the recorder's `DEPTH_COLS` |
| `days/D/news.jsonl` | `store/news/headlines/<first_seen date>.jsonl` | skipped if the same `headline_id` is already stored within ±4 days |
| `days/D/candles/<s>.csv.gz` | `store/intraday/<s>.csv` | appended, de-duplicated on `ts`, 09:15-15:29 bars only (same rules as `intraday_store`) |
| `days/E/expired_options_1m.csv.gz` | `store/options_hist_1m/E.csv` | appended, de-duplicated on (key, ts) |

After a sync, `recorder.load_day`, `latest_chain_metrics`, research experiment F2, the intraday
model and the ATM replay all see the collected data. When the same headline was seen by both the
Mac and the collector, the record that was stored first is kept.

To sync automatically, add a cron or launchd job, for example every day at 18:00:
`0 18 * * * cd ~/nifty-tracker && .venv/bin/python -m nifty_engine.collect sync >> store/sync.log 2>&1`.

## Running the collector by hand

```bash
python -m nifty_engine.collect news --out /tmp/collect             # writes /tmp/collect/days/<today>/news.jsonl
python -m nifty_engine.collect snapshot --out /tmp/collect --force # needs UPSTOX_ANALYTICS_TOKEN in .env
python -m nifty_engine.collect eod --out /tmp/collect --force --no-expired
python -m nifty_engine.collect merge --out /tmp/collect --data-branch-dir ~/nifty-data
```

## Limitations

- **Granularity.** Snapshots are 15 minutes apart, not the recorder's 30 s. OI build-up metrics
  (`near_doi_*`, `*_buildup_*`) compare against the previous stored snapshot, so their scale
  depends on the gap. Depth is the REST top-5 book; 30-level depth needs the WebSocket recorder.
- **Late or dropped cron runs.** GitHub does not guarantee scheduled times. The first snapshot of
  the day often lands after 09:20 IST. Research experiment F2 uses the last snapshot at or before
  09:20, and if there is none it falls back to the first snapshot of the day, which is then later
  than a 09:20 decision. Treat F2 results on collector-only days with care.
- **first_seen precision.** News is polled every 15 minutes plus the start delay, so `first_seen`
  can be up to ~20-30 minutes after publication. It is conservative (never early), so there is
  no look-ahead. Feeds that block cloud IP ranges (often NSE's archives and some media sites)
  may fail from GitHub's runners; `runs/*.jsonl` lists `sources_failed` for each run. GDELT is not
  polled in CI because shared runner IPs hit its rate limit.
- **Holidays.** No holiday calendar is used. A snapshot is skipped when no instrument traded today.
  If Upstox does not return a last-trade time, the snapshot is kept rather than lost.
- **Expired options.** The strike range is centred on NIFTY's close on the expiry day (± 800 points),
  so on weeks with a big move, strikes that were ATM early in the week may be missing. This needs
  Upstox Plus; without it the step logs "Upstox Plus not available" and continues. An expiry
  counts as complete when all but at most two contracts were downloaded; incomplete expiries are
  retried for up to 4 days.
- **Scheduled workflows in public repositories** are disabled by GitHub after 60 days without
  repository activity. The collector pushes to the `data` branch several times a day, which
  should count as activity. If the workflows are ever disabled, re-enable them in the Actions tab.
- **Repository size** grows by about 0.5-0.65 GB a year (see above).
- **Not tested against the live API from CI yet.** The Upstox endpoints (quotes, option chain, v3
  intraday candles, News, expired instruments) are called through the existing `UpstoxProvider` /
  `expired_options` code, and the unit tests use fakes with the documented response shapes. Run
  `snapshot --force` and `eod --force` once by hand (or through *Run workflow*) and check
  `status` before relying on it.
