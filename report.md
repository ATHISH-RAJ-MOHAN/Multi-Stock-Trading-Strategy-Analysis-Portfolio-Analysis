# 2024-2025 Backtest Report

This report summarizes the 2024-2025 backtest for BMNR, BE, OKLO, IREN, and SNDK using:

- Moving Average Crossover (SMA 20/50)
- Exponential Smoothing (SES alpha=0.2)
- ARIMA (rolling ARIMA(1,1,1) on 60-day windows, long-only)
- Buy-and-hold baseline (20% per stock)

## Methodology

- **Data**: `yfinance` daily data (Adjusted Close)
- **Execution**: signals on day *t*, trades at next-day adjusted close
- **Initial allocation**: 20% per stock at each ticker's first available date
- **Positioning**: equal-weight across active long signals; rebalance only when signals change
- **Shares**: whole shares only
- **Fees/Slippage**: none
- **Missing data**: skip trades for missing prices; valuation uses last known price
- **Initial allocation timing**: if a ticker's first buy happens later, rebalancing starts the next day
- **Sharpe**: 3-month T-bill (FRED `TB3MS`) when available, otherwise 0%
- **ARIMA long-only**: negative forecasts go to cash

## Results

Run the backtest to populate the results table in `outputs/report.md`:

```bash
python backtest.py
```

That file contains the full results table (final value, total return, CAGR, Sharpe, max drawdown) and a chart of portfolio value over 2025 in `outputs/portfolio_values.png`.

If a ticker has no 2025 data, its 20% allocation remains as cash for the year.
