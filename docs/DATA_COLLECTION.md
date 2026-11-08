# Data collection with GitHub Actions

> **Off by default.** The workflow runs only after you set the repository variable
> `COLLECT_ENABLED` to `true` (Settings → Secrets and variables → Actions → Variables).
>
> If this repository is **public**, point `DATA_REPO` at a **private** repository first, and set
> `DATA_REPO_TOKEN` to a token with Contents read/write there. Exchange and broker data should not
> be redistributed publicly. Actions minutes stay free either way, since they're billed to this
> (public) code repository, not to `DATA_REPO`.

## What it does

One workflow, one job, one file: [`.github/workflows/collect.yml`](../.github/workflows/collect.yml).
Every 15 minutes on market days it:

1. checks out the `data` branch of `DATA_REPO` as the previous run left it (creating the branch on
   first use);
2. runs four of the engine's own commands, each already incremental on its own:
   - `chain-poll` — one option-chain snapshot (skips if the market is closed or no token is set)
   - `refresh-intraday` — recent 1-/5-minute NIFTY/VIX candles
   - `fetch-history` — Bank Nifty, the heavyweight stocks, NIFTY futures
   - `news-capture` — headlines (free sources; needs no token)
3. commits and pushes whatever changed.

Because every run starts by checking out whatever the last run left, and every command only
appends new rows, there's no separate merge step: each run naturally continues "after the last
data" rather than re-fetching or overwriting it.

The local live recorder (`python -m nifty_engine run` / `record`) is still better when the Mac is
on: it polls the option chain every 30 s and streams order-book depth over WebSocket, neither of
which fits a 15-minute cron tick. This workflow fills the gaps for whenever the Mac is off.

## Setup

1. Create (or choose) a repository for the data. If this code repository is public, make it a
   **separate, private** one.
2. In this repository's Settings → Secrets and variables → Actions:
   - **Secrets:** `UPSTOX_ANALYTICS_TOKEN` (your Upstox token); `DATA_REPO_TOKEN` (a token with
     Contents read/write on the data repository — skip this if `DATA_REPO` is this same repository).
   - **Variables:** `DATA_REPO` (`owner/repo` of the data repository); `COLLECT_ENABLED` = `true`.
3. Run the workflow once by hand from the Actions tab ("Collect Upstox data + news" → Run workflow)
   to confirm it works, then let the schedule take over.

## Getting the data onto your Mac

Not automated yet. For now:

```bash
git fetch https://github.com/<owner>/<data-repo>.git data
git archive FETCH_HEAD store | tar -x -C /path/to/nifty-tracker
```

Only copy in files you don't already have locally more completely (the Mac's own live recorder is
usually ahead of the 15-minute snapshots).

## Limits

- **Snapshot granularity.** Every 15 minutes, not every 30 seconds — enough to fill gaps, not a
  substitute for running the engine locally during the day.
- **No order-book depth.** Depth needs a persistent WebSocket connection, which doesn't fit a short
  scheduled job. It's only ever recorded by the local engine.
- **Without `UPSTOX_ANALYTICS_TOKEN`**, only the free news capture runs; the Upstox-dependent steps
  skip themselves cleanly.
- **Scheduled workflows in public repositories** are disabled by GitHub after 60 days with no
  repository activity. A commit to `main` resets that clock.
