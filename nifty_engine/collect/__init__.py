"""GitHub Actions data collector.

Market data that cannot be downloaded later (option-chain snapshots, quotes with depth, live news
with first-seen times, 1-minute candles of just-expired options) is collected by scheduled
workflows even when the Mac is off, stored on the orphan `data` branch, and merged into the local
store with `python -m nifty_engine.collect sync`. See docs/DATA_COLLECTION.md.

Modules: layout (branch layout, gzip/JSONL append helpers), market (Upstox snapshot/EOD jobs),
news (RSS + Upstox news poll), merge (staging -> data branch), sync (data branch -> local store).
"""

from .layout import LAYOUT_VERSION  # noqa: F401
