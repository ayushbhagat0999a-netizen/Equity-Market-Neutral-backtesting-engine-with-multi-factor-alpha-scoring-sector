# EMN Multi-Factor Backtester

Equity Market-Neutral backtesting engine with multi-factor alpha scoring, sector neutrality, turnover constraints, and interactive dashboards.

## Features

- **Long/short portfolio construction** — dollar-neutral with 200% gross exposure
- **Three factor families** — Value (BP, EP), Momentum, Quality (ROE, GP, Leverage)
- **Sector neutrality** — optional per-sector dollar-neutral constraint
- **Turnover constraints** — limit single-rebalance turnover with proportional weight scaling
- **Factor attribution** — OLS-based daily return decomposition (Value / Momentum / Quality / Specific)
- **Interactive charts** — cumulative return, drawdown, rolling Sharpe, factor exposure, exposure tracking
- **Dark mode** — persisted in localStorage
- **CSV upload** — standalone version runs entirely in-browser

## Quick Start

### Standalone (no server)

Open `dashboard.html` in your browser, upload a CSV, and run.

### Flask webapp

```bash
pip install flask pandas numpy yfinance chart.js
python webapp/app.py
# Open http://localhost:5000
```

### Generate test data

```bash
python generate_test_data.py
python validate_backtest.py
```

## CSV Format

Required: `Date, Ticker, Close`
Factors: `BP_Z, EP_Z, Momentum_Z, ROE_Z, GP_Z, Leverage_Z`
Optional: `Sector` (enables sector-neutral mode)

Sample data included in `sample_data.csv`.

## Project Structure

| File | Purpose |
|---|---|
| `dashboard.html` | Self-contained interactive dashboard (JS engine + Chart.js) |
| `webapp/app.py` | Flask server |
| `webapp/engine.py` | Python backtest engine |
| `webapp/templates/index.html` | Flask dashboard template |
| `phase1-6_*.py` | Full pipeline from data acquisition to performance reporting |
| `data/scores.csv` | Pre-computed factor Z-scores for 31 large-cap US stocks |
| `generate_test_data.py` | Synthetic test dataset generator |
| `validate_backtest.py` | Validation suite for all test scenarios |

## Metrics

Total Return, Sharpe Ratio, Sortino Ratio, Max Drawdown, Annualised Volatility, Beta, Win Rate, Profit Factor, Calmar Ratio, Jensen's Alpha, Information Ratio, Factor Attribution.

## Backtest Period

2020-01-02 to 2025-12-30 (active from ~2023). Universe: 31 large-cap US stocks, monthly/quarterly rebalance. SPY used as benchmark for Beta, Alpha, IR.
