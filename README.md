# DSCI 560 Lab 4 Backtest

This project backtests three strategies on five stocks for the 2024-2025 period:

- **Moving Average Crossover** (SMA 20/50)
- **Exponential Smoothing** (single exponential smoothing with alpha=0.2)
- **ARIMA** (rolling ARIMA(1,1,1) on 60-day windows, long-only)

It compares both to **buy-and-hold** with a $100,000 portfolio split 20% per stock.

## Assumptions (per your specs)

- **Data**: `yfinance` daily data
- **Price field**: **Adjusted Close**
- **Execution**: signal on day *t*, trade at **next day Adjusted Close**
- **Initial allocation**: 20% per stock at each ticker's first available date
- **Positioning**: equal-weight across active longs, rebalance only when signals change
- **Shares**: whole shares only
- **Fees/Slippage**: none
- **Missing data**: skip trades for missing prices; valuation uses last known price
- **Initial allocation timing**: if a ticker's first buy happens later, rebalancing starts the next day
- **Sharpe**: 3-month T-bill (FRED `TB3MS`) when available, otherwise 0%
- **ARIMA long-only**: negative forecasts go to cash

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python backtest.py
```

Outputs are written to `outputs/`:

- `summary.csv`
- `portfolio_values.png`
- `*_portfolio.csv`
- `*_trades.csv`
- `report.md`

## Tickers

`BMNR`, `BE`, `OKLO`, `IREN`, `SNDK`

## Notes

If a ticker has partial 2025 data, the backtest will **skip trades** on days with missing prices (as requested).
If a ticker has no 2025 data, its 20% allocation stays as cash for the year.
