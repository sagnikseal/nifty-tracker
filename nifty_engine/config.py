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
    capital: float = 100_000.0           # paper capital in INR (1 ITM lot = Rs10-16k premium)
    books: str = "options"               # "options", "futures" or "options,futures"
    option_style: str = "itm_buy"        # "itm_buy" (default, best in replay) or "debit_spread"
    spread_width_pts: int = 150          # short leg distance for debit spreads
    lot_size: int = 65                   # NIFTY lot (65 from Jan-2026 series)
    max_lots: int = 1

    # ---- risk --------------------------------------------------------------
    risk_per_trade_pct: float = 10.0     # loss budget at the disaster stop (% of equity)
    max_premium_pct: float = 35.0        # max premium outlay per trade (% of equity)
    option_stop_pct: float = 60.0        # disaster stop only: tight stops lost money in replay (VARIANTS.md)
    option_target_pct: float = 70.0      # take profit if value rises this %
    daily_loss_limit_pct: float = 6.0
    max_drawdown_kill_pct: float = 30.0  # stop trading (paper) after this drawdown
    paper_trade_without_edge: bool = True  # paper mode may trade an unproven model (flagged)
    max_trades_per_day: int = 1

    # ---- iron condor book ("condor" in books) --------------------------------
    condor_k: float = 1.25               # short strikes at k x VIX-implied daily sigma from spot
    condor_wing: int = 200               # long wings this many points beyond the shorts
    condor_stop_mult: float = 1.5        # exit if loss reaches this multiple of the credit
    condor_p_threshold: float = 0.80     # trade only when range model P(stays inside) >= this
    condor_max_loss_pct: float = 15.0    # max defined loss per trade (% of equity)

    # ---- signal ------------------------------------------------------------
    target: str = "oc"                   # "oc" = today open->close, "cc" = close->next close
    prob_threshold: float = 0.56         # trade only if p>=thr (long) or p<=1-thr (short)
    deadzone_bps: float = 5.0            # labels: |ret| below this counted as no-edge (dropped)
    retrain_every_days: int = 21
    use_zoo: bool = True                 # weekly retrain also runs the model zoo and may deploy best-3
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

    # ---- feature groups, research strategy books, local LLM ----------------------------
    feature_groups: str = "base"         # daily-model feature groups (features.FEATURE_GROUPS); add a group
                                         # only after `python -m nifty_engine research` shows it helps
    strategy_books: str = ("debit_spread,credit_spread,iron_condor,iron_fly_0dte,event_straddle,"
                           "turn_of_month,smc_intraday,llm_analyst")   # research paper books (strategies/books.py), off by default
    book_capital: float = 100_000.0      # paper equity per book
    llm_model: str = "qwen3.5:9b"        # local Ollama model (free, ~6.6 GB, fits a 16 GB Mac next to the engine).
                                         # Compare candidates with `python -m nifty_engine llm-bench`.
    llm_url: str = "http://localhost:11434"
    llm_knowledge_cutoff: str = "2026-03-10"   # news on/before this date may be "remembered" by the model:
                                               # it is scored but excluded from accuracy tests. Unknown cutoffs
                                               # use the model's release date (a safe upper bound).
    record_live: bool = True             # record option chain / depth during market hours (needs token)

    # ---- the user's ATM rules (strategies/atm_rules.py), all amounts in Rs for the position --------
    atm_target_inr: float = 5000.0       # take profit at +Rs 5,000 ...
    atm_extend_prob: float = 0.80        # ... unless P(move continues 30 more min) > 80%: then hold 30 min, repeat
    atm_extend_minutes: int = 30
    atm_stop_inr: float = 2000.0         # exit immediately at -Rs 2,000
    atm_against_bars: int = 2            # exit after 2 consecutive 5-min closes against the trade ...
    atm_against_scope: str = "in_profit"  # ... while in profit ("always" / "after_target" are the alternatives)
    atm_entry_threshold: float = 0.60    # enter when P(up by 15:15) >= 0.60 (call) or <= 0.40 (put)
    atm_entry_start: str = "09:20"
    atm_entry_end: str = "13:30"
    atm_exit_time: str = "15:15"
    atm_lots: int = 1
    atm_max_trades_per_day: int = 1
    atm_news_exit_bps: float = 8.0       # exit on a headline with learned impact >= 8 bps against the trade
                                         # (only active once the news-impact model passes its test)

    # ---- runtime -------------------------------------------------------------------------------------
    price_poll_seconds: float = 5.0      # option/spot price checks for open positions (stop is checked on each)
    news_poll_seconds: int = 60          # fast news sources (conditional GET, per-source backoff)
    agent_enabled: bool = True           # local-LLM trading agent (paper), decides every 5 minutes
    agent_auto: bool = True              # agent book trades automatically (paper) when enabled
    model_rules_auto: bool = True        # model + ATM rules book trades automatically (paper)

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

    @property
    def group_list(self):
        return [g.strip() for g in self.feature_groups.split(",") if g.strip()] or ["base"]

    @property
    def strategy_list(self):
        return [b.strip() for b in self.strategy_books.split(",") if b.strip()]
