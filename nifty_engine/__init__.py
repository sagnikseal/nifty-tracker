"""NIFTY v4 engine: broker-agnostic data layer, leakage-free walk-forward models,
options paper-trading with realistic Indian F&O charges, and scheduled self-retraining.

Run `python -m nifty_engine --help` for the command list.
"""

__version__ = "4.0.0"

# Corporate networks often intercept TLS with their own root CA (installed in the OS store
# but unknown to certifi). truststore makes Python use the OS trust store, like browsers do.
try:  # pragma: no cover
    import truststore as _truststore

    _truststore.inject_into_ssl()
except Exception:  # noqa: BLE001
    pass
