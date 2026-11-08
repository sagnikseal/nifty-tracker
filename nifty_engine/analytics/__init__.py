"""Market analytics: price structure (SMC), technical indicators, option-chain microstructure
and regimes. Every function is causal: row t only uses information available at the close of
bar t (tests/test_research_books.py perturbs future bars and checks earlier rows don't change)."""
