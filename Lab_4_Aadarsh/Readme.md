# DSCI 560 Algorithmic Trading System

This project implements and backtests three algorithmic trading strategies on selected stock portfolios for the 2024-2025 period:

- **Baseline Strategy**: Combined technical indicators (Moving Average Crossover, RSI, MACD, Bollinger Bands)
- **LSTM-Attention Model**: Deep learning sequential model with attention mechanism
- **Reinforcement Learning Agent**: Q-learning optimization of trading decisions

All strategies are compared using identical initial conditions and transaction costs.

## Assumptions

- **Data Source**: Yahoo Finance daily OHLCV data
- **Initial Capital**: $100,000 allocated equally across selected tickers
- **Transaction Cost**: 0.1% per trade
- **Position Sizing**: 10% maximum allocation per position
- **Execution**: Signals generated at close, executed at next open
- **Evaluation Period**: January 2024 - January 2026
- **Risk-Free Rate**: 2% annual for Sharpe ratio calculations

## Quick Start

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the complete trading system
python main.py
```

## Output & Results

After execution, all generated data and visualizations are stored in the `results/` directory:

| File Type | Filename(s) | Description |
| :--- | :--- | :--- |
| **Portfolio Logs** | `portfolio_baseline.csv`, `portfolio_lstm.csv`, `portfolio_rl_agent.csv` | Detailed daily trade logs and holdings for each strategy. |
| **Metrics** | `strategy_comparison.csv` | Comparative table of KPIs (Sharpe Ratio, Max Drawdown, etc.). |
| **Visualization** | `performance_comparison.png` | Equity curve visualization comparing all three strategies. |

---

## Test Portfolios

### High-Cap Tech Basket
`NVDA`, `AAPL`, `GOOGL`, `MSFT`, `AMZN`

### High-Volatility Basket
`BMNR`, `BE`, `OKLO`, `IREN`, `SNDK`

---

## Technical Notes

> **Core System Requirements & Logic**
> * **Feature Set**: All strategies leverage a shared set of 50+ technical indicators.
> * **LSTM Requirements**: Requires a sliding window of at least 60 days of historical data for predictions.
> * **RL Training**: The agent undergoes 10 episodes of training before generating live signals.
> * **Data Integrity**: If price data is missing for a specific ticker on a given day, trading is skipped for that asset only for that session.