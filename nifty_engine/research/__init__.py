"""Research harness: replays every strategy book on real expired-option prices, tests feature
groups against the base model on identical days, and applies multiple-testing corrections
(deflated Sharpe, probability of backtest overfitting). Run `python -m nifty_engine research`."""
