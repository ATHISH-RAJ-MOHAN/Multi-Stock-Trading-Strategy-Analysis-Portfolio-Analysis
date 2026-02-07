#!/usr/bin/env python3
"""
Backtest for Moving Average (SMA crossover), Exponential Smoothing (SES),
and ARIMA strategies on a 2025 calendar-year window.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Missing dependency: yfinance. Install with `pip install -r requirements.txt`.") from exc

try:
    from pandas_datareader import data as pdr
except Exception:  # pragma: no cover
    pdr = None

try:
    from statsmodels.tsa.arima.model import ARIMA
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Missing dependency: statsmodels. Install with `pip install -r requirements.txt`.") from exc

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_TICKERS = ["BMNR", "BE", "OKLO", "IREN", "SNDK"]
TRADING_DAYS = 252


@dataclass
class BacktestConfig:
    tickers: List[str]
    start: str
    end: str
    initial_cash: float
    sma_short: int
    sma_long: int
    ses_alpha: float
    use_risk_free: bool


def download_prices(tickers: List[str], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    data = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=False,
        group_by="ticker",
        progress=False,
    )

    if isinstance(data.columns, pd.MultiIndex):
        if "Adj Close" in data.columns.levels[1]:
            prices = data.xs("Adj Close", axis=1, level=1)
        elif "Close" in data.columns.levels[1]:
            prices = data.xs("Close", axis=1, level=1)
        else:
            raise ValueError("Price data missing Adj Close/Close columns.")
    else:
        col = "Adj Close" if "Adj Close" in data.columns else "Close"
        prices = data[[col]].rename(columns={col: tickers[0]})

    prices = prices.sort_index()
    prices = prices.reindex(columns=tickers)
    return prices


def compute_sma_signals(prices: pd.DataFrame, short_window: int, long_window: int) -> pd.DataFrame:
    short = prices.rolling(window=short_window, min_periods=short_window).mean()
    long = prices.rolling(window=long_window, min_periods=long_window).mean()
    signals = (short > long).astype(int)
    return signals.fillna(0)


def compute_ses_signals(prices: pd.DataFrame, alpha: float) -> pd.DataFrame:
    ses = prices.ewm(alpha=alpha, adjust=False).mean()
    signals = (prices > ses).astype(int)
    return signals.fillna(0)


def compute_arima_signals(
    prices: pd.DataFrame,
    order: Tuple[int, int, int] = (1, 1, 1),
    window: int = 60,
    allow_short: bool = False,
) -> pd.DataFrame:
    tickers = list(prices.columns)
    signals = pd.DataFrame(0, index=prices.index, columns=tickers, dtype=int)

    for t in tickers:
        series = prices[t]
        for i in range(window - 1, len(series)):
            window_slice = series.iloc[i - window + 1 : i + 1]
            current_price = window_slice.iloc[-1]

            if window_slice.isna().any() or pd.isna(current_price):
                continue

            try:
                model = ARIMA(
                    window_slice,
                    order=order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                    trend="n",
                )
                fitted = model.fit()
                forecast = float(fitted.forecast(steps=1).iloc[0])
            except Exception:
                continue

            expected_return = forecast / current_price - 1.0
            if expected_return > 0:
                signals.iat[i, signals.columns.get_loc(t)] = 1
            elif expected_return < 0 and allow_short:
                signals.iat[i, signals.columns.get_loc(t)] = -1

    return signals


def get_risk_free_daily(start: pd.Timestamp, end: pd.Timestamp) -> Optional[pd.Series]:
    if pdr is None:
        print("Warning: pandas_datareader not available. Using 0% risk-free rate.")
        return None
    try:
        rf = pdr.DataReader("TB3MS", "fred", start, end)
    except Exception as exc:  # pragma: no cover
        print(f"Warning: could not load risk-free rate from FRED: {exc}")
        return None

    rf = rf.rename(columns={"TB3MS": "rf"})
    full_index = pd.date_range(start=start, end=end, freq="D")
    rf = rf.reindex(full_index).ffill()
    rf_daily = rf["rf"] / 100.0 / TRADING_DAYS
    return rf_daily


def _value_portfolio(holdings: pd.Series, cash: float, prices_ffill: pd.Series) -> float:
    return float(cash + (holdings * prices_ffill).sum())


def simulate_strategy(
    prices: pd.DataFrame,
    signals: pd.DataFrame,
    initial_cash: float,
    initial_weights: Optional[Dict[str, float]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    tickers = list(prices.columns)
    dates = list(prices.index)
    prices_ffill = prices.ffill().fillna(0)

    holdings = pd.Series(0, index=tickers, dtype=int)
    cash = float(initial_cash)

    records: List[Dict[str, float]] = []
    trades: List[Dict[str, float]] = []

    bought = {t: False for t in tickers} if initial_weights else None
    reserved = {t: initial_cash * initial_weights[t] for t in tickers} if initial_weights else {}

    for i in range(len(dates)):
        exec_date = dates[i]
        initial_buy_today = False

        # Initial 20% allocation per ticker (on each ticker's first available date).
        if initial_weights is not None and bought is not None:
            for t in tickers:
                if bought[t]:
                    continue
                price = prices.loc[exec_date, t]
                if pd.isna(price):
                    continue
                bucket = reserved.get(t, 0.0)
                shares = int(bucket // price)
                if shares > 0:
                    cost = float(shares * price)
                    holdings[t] += shares
                    cash -= cost
                    reserved[t] = 0.0
                    bought[t] = True
                    initial_buy_today = True

        if i == 0:
            # Record initial state on the first trading day (no trades yet).
            records.append(
                {
                    "date": exec_date,
                    "cash": cash,
                    "total_value": _value_portfolio(holdings, cash, prices_ffill.loc[exec_date]),
                    **{f"shares_{t}": int(holdings[t]) for t in tickers},
                }
            )
            continue

        if initial_buy_today:
            # Avoid buying and selling the same day; rebalance on the next day.
            records.append(
                {
                    "date": exec_date,
                    "cash": cash,
                    "total_value": _value_portfolio(holdings, cash, prices_ffill.loc[exec_date]),
                    **{f"shares_{t}": int(holdings[t]) for t in tickers},
                }
            )
            continue

        signal_date = dates[i - 1]

        signal_row = signals.loc[signal_date]
        active = [t for t in tickers if signal_row.get(t, 0) != 0]
        desired_signals = {t: int(signal_row.get(t, 0)) for t in tickers}
        current_signals = {t: int(np.sign(holdings[t])) for t in tickers}

        if desired_signals != current_signals:
            total_equity = _value_portfolio(holdings, cash, prices_ffill.loc[exec_date])
            reserved_total = float(sum(reserved.values())) if reserved else 0.0
            tradable_equity = total_equity - reserved_total
            weight = 1.0 / len(active) if active else 0.0

            for t in tickers:
                price = prices.loc[exec_date, t]
                if pd.isna(price):
                    # Skip trades when the execution price is missing.
                    continue

                signal_value = int(signal_row.get(t, 0))
                if signal_value == 0:
                    target_value = 0.0
                else:
                    target_value = tradable_equity * weight * signal_value
                target_shares = int(target_value / price)
                delta = target_shares - holdings[t]

                if delta != 0:
                    cash -= float(delta * price)
                    holdings[t] += int(delta)
                    trades.append(
                        {
                            "date": exec_date,
                            "ticker": t,
                            "delta_shares": int(delta),
                            "price": float(price),
                            "cash_after": float(cash),
                        }
                    )

        records.append(
            {
                "date": exec_date,
                "cash": cash,
                "total_value": _value_portfolio(holdings, cash, prices_ffill.loc[exec_date]),
                **{f"shares_{t}": int(holdings[t]) for t in tickers},
            }
        )

    portfolio = pd.DataFrame(records).set_index("date")
    trades_df = pd.DataFrame(trades)
    return portfolio, trades_df


def simulate_buy_hold(
    prices: pd.DataFrame,
    initial_cash: float,
    weights: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    tickers = list(prices.columns)
    dates = list(prices.index)
    prices_ffill = prices.ffill().fillna(0)

    if weights is None:
        weights = {t: 1.0 / len(tickers) for t in tickers}

    holdings = pd.Series(0, index=tickers, dtype=int)
    cash = float(initial_cash)
    bought = {t: False for t in tickers}

    records: List[Dict[str, float]] = []

    for date in dates:
        for t in tickers:
            if bought[t]:
                continue
            price = prices.loc[date, t]
            if pd.isna(price):
                continue

            bucket = initial_cash * weights[t]
            shares = int(bucket // price)
            if shares > 0:
                cost = float(shares * price)
                holdings[t] += shares
                cash -= cost
                bought[t] = True

        records.append(
            {
                "date": date,
                "cash": cash,
                "total_value": _value_portfolio(holdings, cash, prices_ffill.loc[date]),
                **{f"shares_{t}": int(holdings[t]) for t in tickers},
            }
        )

    return pd.DataFrame(records).set_index("date")


def compute_metrics(portfolio: pd.DataFrame, rf_daily: Optional[pd.Series]) -> Dict[str, float]:
    values = portfolio["total_value"]
    daily_returns = values.pct_change().dropna()

    if rf_daily is None:
        excess = daily_returns
    else:
        rf_aligned = rf_daily.reindex(daily_returns.index).ffill().fillna(0)
        excess = daily_returns - rf_aligned

    total_return = float(values.iloc[-1] / values.iloc[0] - 1.0)
    if len(daily_returns) == 0:
        cagr = float("nan")
    else:
        cagr = float((values.iloc[-1] / values.iloc[0]) ** (TRADING_DAYS / len(daily_returns)) - 1.0)

    if excess.std(ddof=0) == 0:
        sharpe = float("nan")
    else:
        sharpe = float(np.sqrt(TRADING_DAYS) * excess.mean() / excess.std(ddof=0))

    running_max = values.cummax()
    drawdown = values / running_max - 1.0
    max_drawdown = float(drawdown.min())

    return {
        "final_value": float(values.iloc[-1]),
        "total_return": total_return,
        "cagr": cagr,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
    }


def run_backtest(config: BacktestConfig):
    start = pd.Timestamp(config.start)
    end_inclusive = pd.Timestamp(config.end)
    download_end = end_inclusive + pd.Timedelta(days=1)

    prices = download_prices(config.tickers, start, download_end)
    prices = prices.loc[(prices.index >= start) & (prices.index <= end_inclusive)]

    if prices.empty:
        raise ValueError("No price data returned for the requested window.")

    sma_signals = compute_sma_signals(prices, config.sma_short, config.sma_long)
    ses_signals = compute_ses_signals(prices, config.ses_alpha)
    arima_signals = compute_arima_signals(
        prices, order=(1, 1, 1), window=60, allow_short=False
    )

    weights = {t: 0.2 for t in config.tickers}
    ma_portfolio, ma_trades = simulate_strategy(
        prices, sma_signals, config.initial_cash, initial_weights=weights
    )
    es_portfolio, es_trades = simulate_strategy(
        prices, ses_signals, config.initial_cash, initial_weights=weights
    )
    arima_portfolio, arima_trades = simulate_strategy(
        prices, arima_signals, config.initial_cash, initial_weights=weights
    )

    buy_hold_portfolio = simulate_buy_hold(prices, config.initial_cash, weights)

    rf_daily = get_risk_free_daily(start, end_inclusive) if config.use_risk_free else None

    ma_metrics = compute_metrics(ma_portfolio, rf_daily)
    es_metrics = compute_metrics(es_portfolio, rf_daily)
    arima_metrics = compute_metrics(arima_portfolio, rf_daily)
    bh_metrics = compute_metrics(buy_hold_portfolio, rf_daily)

    summary = pd.DataFrame(
        [
            {"strategy": "Moving Average", **ma_metrics},
            {"strategy": "Exponential Smoothing", **es_metrics},
            {"strategy": "ARIMA (1,1,1) Long-Only", **arima_metrics},
            {"strategy": "Buy & Hold", **bh_metrics},
        ]
    ).set_index("strategy")

    portfolios = {
        "Moving Average": ma_portfolio,
        "Exponential Smoothing": es_portfolio,
        "ARIMA (1,1,1) Long-Only": arima_portfolio,
        "Buy & Hold": buy_hold_portfolio,
    }

    trades = {
        "Moving Average": ma_trades,
        "Exponential Smoothing": es_trades,
        "ARIMA (1,1,1) Long-Only": arima_trades,
    }

    return summary, portfolios, trades, prices


def write_outputs(
    summary: pd.DataFrame,
    portfolios: Dict[str, pd.DataFrame],
    trades: Dict[str, pd.DataFrame],
    prices: pd.DataFrame,
    output_dir: str,
    config: BacktestConfig,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    summary.to_csv(os.path.join(output_dir, "summary.csv"))
    prices.to_csv(os.path.join(output_dir, "prices.csv"))

    for name, df in portfolios.items():
        safe = name.lower().replace(" ", "_")
        df.to_csv(os.path.join(output_dir, f"{safe}_portfolio.csv"))

    for name, df in trades.items():
        safe = name.lower().replace(" ", "_")
        df.to_csv(os.path.join(output_dir, f"{safe}_trades.csv"), index=False)

    # Plot portfolio values
    plt.figure(figsize=(10, 6))
    for name, df in portfolios.items():
        plt.plot(df.index, df["total_value"], label=name)

    plt.title("Portfolio Value Over 2024-2025")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value ($)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "portfolio_values.png"))
    plt.close()

    # Write a short report
    report_path = os.path.join(output_dir, "report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# 2024-2025 Backtest Report\n\n")
        f.write("## Setup\n")
        f.write(f"- Dates: {config.start} to {config.end}\n")
        f.write(f"- Tickers: {', '.join(config.tickers)}\n")
        f.write(f"- Initial cash: ${config.initial_cash:,.0f}\n")
        f.write("- Initial allocation: 20% per stock at each ticker's first available date\n")
        f.write("- Signals: SMA(20/50), SES(alpha=0.2), ARIMA(1,1,1) rolling 60 days (long-only)\n")
        f.write("- Execution: signals on day t, trades at next-day Adj Close\n")
        f.write("- Positioning: equal-weight across active longs, rebalance on signal changes\n")
        f.write("- Whole shares, no fees\n")
        f.write("- ARIMA strategy is long-only (negative forecasts go to cash)\n")
        f.write("- Missing data: skip trades if a ticker has no price; valuation uses last known price\n")
        f.write("- Initial allocations trigger rebalancing starting the next trading day\n")
        f.write("- Sharpe: risk-free rate from 3M T-bill (FRED TB3MS) if available\n\n")

        f.write("## Results Summary\n\n")
        f.write("| Strategy | Final Value | Total Return | CAGR | Sharpe | Max Drawdown |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")

        for strategy, row in summary.iterrows():
            f.write(
                "| {strategy} | ${final:,.2f} | {total:.2%} | {cagr:.2%} | {sharpe:.3f} | {mdd:.2%} |\n".format(
                    strategy=strategy,
                    final=row["final_value"],
                    total=row["total_return"],
                    cagr=row["cagr"],
                    sharpe=row["sharpe"],
                    mdd=row["max_drawdown"],
                )
            )

        f.write("\n## Notes\n")
        f.write("- Results are based on daily adjusted close prices for 2025 only.\n")
        f.write("- Trades execute at the next trading day close to avoid lookahead bias.\n")
        f.write("- If a ticker has no 2025 data, its 20% allocation remains in cash.\n")


def parse_args() -> BacktestConfig:
    parser = argparse.ArgumentParser(description="Backtest SMA and SES strategies for 2025.")
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--initial-cash", type=float, default=100_000)
    parser.add_argument("--sma-short", type=int, default=20)
    parser.add_argument("--sma-long", type=int, default=50)
    parser.add_argument("--ses-alpha", type=float, default=0.2)
    parser.add_argument(
        "--risk-free",
        dest="use_risk_free",
        action="store_true",
        default=True,
        help="Use 3M T-bill as risk-free rate for Sharpe.",
    )
    parser.add_argument(
        "--no-risk-free",
        dest="use_risk_free",
        action="store_false",
        help="Disable risk-free adjustment (Sharpe uses 0%).",
    )
    args = parser.parse_args()

    return BacktestConfig(
        tickers=list(args.tickers),
        start=args.start,
        end=args.end,
        initial_cash=args.initial_cash,
        sma_short=args.sma_short,
        sma_long=args.sma_long,
        ses_alpha=args.ses_alpha,
        use_risk_free=args.use_risk_free,
    )


def main() -> None:
    config = parse_args()
    summary, portfolios, trades, prices = run_backtest(config)

    output_dir = os.path.join(os.getcwd(), "outputs")
    write_outputs(summary, portfolios, trades, prices, output_dir, config)

    print("Backtest complete. Summary:")
    print(summary)
    print(f"Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
