"""SQLite trade ledger shared by the live paper trader and the API (survives restarts)."""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from ..config import DATA_DIR

DB = DATA_DIR / "paper" / "ledger.sqlite"
_LOCK = threading.Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS trades (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  book TEXT, mode TEXT, date TEXT, entry_ts TEXT, exit_ts TEXT,
  direction INTEGER, style TEXT, legs TEXT, lots INTEGER,
  entry_value REAL, exit_value REAL, gross_pnl REAL, charges REAL, net_pnl REAL,
  exit_reason TEXT, p_up REAL, model_version TEXT, status TEXT, price_source TEXT
);
CREATE TABLE IF NOT EXISTS equity (date TEXT, book TEXT, equity REAL, PRIMARY KEY(date, book));
CREATE TABLE IF NOT EXISTS signals (ts TEXT PRIMARY KEY, payload TEXT);
CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT);
"""


class Ledger:
    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path or DB)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)
            cols = {r[1] for r in c.execute("PRAGMA table_info(trades)")}
            if "rules" not in cols:        # v5: per-trade exit rules for the strategy books
                c.execute("ALTER TABLE trades ADD COLUMN rules TEXT")

    @contextmanager
    def _conn(self):
        c = sqlite3.connect(self.path, timeout=10)
        try:
            with c:            # commit / rollback
                yield c
        finally:
            c.close()          # sqlite3's own context manager does not close the connection

    def insert_trade(self, row: Dict[str, Any]) -> int:
        row = dict(row)
        for k in ("legs", "rules"):
            if isinstance(row.get(k), (list, dict)):
                row[k] = json.dumps(row[k], default=str)
        cols = ",".join(row)
        q = ",".join("?" * len(row))
        with _LOCK, self._conn() as c:
            cur = c.execute(f"INSERT INTO trades ({cols}) VALUES ({q})", list(row.values()))
            return int(cur.lastrowid)

    def update_trade(self, tid: int, **kw) -> None:
        if "legs" in kw and not isinstance(kw["legs"], str):
            kw["legs"] = json.dumps(kw["legs"])
        sets = ",".join(f"{k}=?" for k in kw)
        with _LOCK, self._conn() as c:
            c.execute(f"UPDATE trades SET {sets} WHERE id=?", [*kw.values(), tid])

    def open_trades(self, book: Optional[str] = None) -> pd.DataFrame:
        q = "SELECT * FROM trades WHERE status='OPEN'" + (" AND book=?" if book else "")
        with self._conn() as c:
            return pd.read_sql_query(q, c, params=[book] if book else [])

    def trades(self, mode: Optional[str] = None) -> pd.DataFrame:
        q = "SELECT * FROM trades" + (" WHERE mode=?" if mode else "") + " ORDER BY id"
        with self._conn() as c:
            return pd.read_sql_query(q, c, params=[mode] if mode else [])

    def set_equity(self, date: str, book: str, equity: float) -> None:
        with _LOCK, self._conn() as c:
            c.execute("INSERT OR REPLACE INTO equity VALUES (?,?,?)", (date, book, equity))

    def equity_curve(self) -> pd.DataFrame:
        with self._conn() as c:
            return pd.read_sql_query("SELECT * FROM equity ORDER BY date", c)

    def log_signal(self, ts: str, payload: Dict) -> None:
        with _LOCK, self._conn() as c:
            c.execute("INSERT OR REPLACE INTO signals VALUES (?,?)", (ts, json.dumps(payload, default=str)))

    def last_signal(self) -> Optional[Dict]:
        with self._conn() as c:
            r = c.execute("SELECT payload FROM signals ORDER BY ts DESC LIMIT 1").fetchone()
        return json.loads(r[0]) if r else None

    def kv_get(self, k: str, default=None):
        with self._conn() as c:
            r = c.execute("SELECT v FROM kv WHERE k=?", (k,)).fetchone()
        return json.loads(r[0]) if r else default

    def kv_set(self, k: str, v) -> None:
        with _LOCK, self._conn() as c:
            c.execute("INSERT OR REPLACE INTO kv VALUES (?,?)", (k, json.dumps(v, default=str)))

    def realized_pnl(self, book: str, mode: str = "live") -> float:
        with self._conn() as c:
            r = c.execute("SELECT COALESCE(SUM(net_pnl),0) FROM trades WHERE book=? AND mode=? AND status='CLOSED'",
                          (book, mode)).fetchone()
        return float(r[0])
