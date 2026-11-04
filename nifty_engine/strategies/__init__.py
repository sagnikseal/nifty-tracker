"""v5 strategy books. Each book turns a decision context into a concrete, pre-registered trade
plan (legs, expiry, entry/exit times, stop/target rules). The same plan objects are executed by
the live paper trader (paper/books_live.py) and replayed on real expired-option prices by the
research harness (research/option_replay.py), so live and backtest can't drift apart."""
