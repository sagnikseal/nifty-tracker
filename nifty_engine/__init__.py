"""NIFTY v5 engine: broker-agnostic data layer, leakage-free walk-forward models,
SMC / technical / option-chain analytics, a free local-LLM news layer, multi-strategy
options paper trading with realistic Indian F&O charges, a pre-registered research
harness, and scheduled self-retraining.

Run `python -m nifty_engine --help` for the command list.
"""

__version__ = "5.0.0"

# Corporate networks often intercept TLS with their own root CA (installed in the OS store
# but unknown to certifi). truststore makes Python use the OS trust store, like browsers do.
try:  # pragma: no cover
    import truststore as _truststore

    _truststore.inject_into_ssl()
except Exception:  # noqa: BLE001
    pass

# Load .env as soon as the package is imported, so every module (e.g. data.providers used on
# its own) sees broker tokens. Previously only modules importing config triggered this.
from . import config as _config  # noqa: E402,F401
