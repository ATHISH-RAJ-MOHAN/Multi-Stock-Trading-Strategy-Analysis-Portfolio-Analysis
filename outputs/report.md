# 2024-2025 Backtest Report

## Setup
- Dates: 2024-01-01 to 2025-12-31
- Tickers: BMNR, BE, OKLO, IREN, SNDK
- Initial cash: $100,000
- Initial allocation: 20% per stock at each ticker's first available date
- Signals: SMA(20/50), SES(alpha=0.2), ARIMA(1,1,1) rolling 60 days (long-only)
- Execution: signals on day t, trades at next-day Adj Close
- Positioning: equal-weight across active longs, rebalance on signal changes
- Whole shares, no fees
- ARIMA strategy is long-only (negative forecasts go to cash)
- Missing data: skip trades if a ticker has no price; valuation uses last known price
- Initial allocations trigger rebalancing starting the next trading day
- Sharpe: risk-free rate from 3M T-bill (FRED TB3MS) if available

## Results Summary

| Strategy | Final Value | Total Return | CAGR | Sharpe | Max Drawdown |
|---|---:|---:|---:|---:|---:|
| Moving Average | $984,348.48 | 884.35% | 215.90% | 1.956 | -38.95% |
| Exponential Smoothing | $848,920.49 | 748.92% | 193.23% | 1.634 | -40.20% |
| ARIMA (1,1,1) Long-Only | $305,036.66 | 205.04% | 75.24% | 1.137 | -53.61% |
| Buy & Hold | $566,009.86 | 466.01% | 139.15% | 1.370 | -47.42% |

## Notes
- Results are based on daily adjusted close prices for 2025 only.
- Trades execute at the next trading day close to avoid lookahead bias.
- If a ticker has no 2025 data, its 20% allocation remains in cash.
