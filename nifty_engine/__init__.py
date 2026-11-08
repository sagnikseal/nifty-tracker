"""NIFTY engine: live dashboard and paper trading for NIFTY options (ATM rules, model and local-LLM
agent books), leakage-tested walk-forward models (daily and 5-minute), news read by a local LLM,
option-chain / futures / cross-asset analytics, a pre-registered research harness, and a GitHub
Actions data collector. Paper trading only.

Run `python -m nifty_engine --help` for the command list.
"""

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
