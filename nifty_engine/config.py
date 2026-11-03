"""Central configuration. Every value can be overridden with an env var
(prefix NE_, e.g. NE_CAPITAL=100000) or a JSON file at config/engine.json."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("NE_DATA_DIR", ROOT / "store"))
CONFIG_FILE = ROOT / "config" / "engine.json"


def _load_dotenv() -> None:
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()


@dataclass
class EngineConfig:
    # ---- account ---------------------------------------------------------
    capital: float = 50_000.0            # paper capital in INR
    books: str = "options"               # "options", "futures" or "options,futures"
    option_style: str = "debit_spread"   # "debit_spread" or "itm_buy"
    spread_width_pts: int = 150          # short leg distance for debit spreads
    lot_size: int = 65                   # NIFTY lot (65 from Jan-2026 series)
    max_lots: int = 2

    # ---- risk --------------------------------------------------------------
    risk_per_trade_pct: float = 5.0      # max loss budget per trade (% of equity)
    max_premium_pct: float = 30.0        # max premium outlay per trade (% of equity)
    option_stop_pct: float = 45.0        # exit if option/spread value falls this % from entry
    option_target_pct: float = 70.0      # take profit if value rises this %
    daily_loss_limit_pct: float = 6.0
    max_drawdown_kill_pct: float = 25.0  # stop trading (paper) after this drawdown
    max_trades_per_day: int = 1

    # ---- signal ------------------------------------------------------------
    target: str = "oc"                   # "oc" = today open->close, "cc" = close->next close
    prob_threshold: float = 0.56         # trade only if p>=thr (long) or p<=1-thr (short)
    deadzone_bps: float = 5.0            # labels: |ret| below this counted as no-edge (dropped)
    retrain_every_days: int = 21
    min_train_days: int = 750
    history_start: str = "2008-01-01"

    # ---- execution / costs ---------------------------------------------------
    brokerage_per_order: float = 20.0
    option_slippage_pts: float = 1.0     # per leg per side, on top of modeled half-spread
    futures_slippage_pts: float = 1.0
    entry_time: str = "09:20"
    exit_time: str = "15:15"

    # ---- data ----------------------------------------------------------------
    broker: str = "auto"                 # auto | upstox | yahoo  (signals never depend on a broker)
    poll_seconds: int = 30

    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls) -> "EngineConfig":
        cfg = cls()
        if CONFIG_FILE.exists():
            for k, v in json.loads(CONFIG_FILE.read_text(encoding="utf-8")).items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
        for f in fields(cls):
            env_val = os.environ.get(f"NE_{f.name.upper()}")
            if env_val is None or f.name == "extra":
                continue
            typ = type(getattr(cfg, f.name))
            setattr(cfg, f.name, typ(env_val) if typ is not bool else env_val.lower() in ("1", "true", "yes"))
        return cfg

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def book_list(self):
        return [b.strip() for b in self.books.split(",") if b.strip()]
