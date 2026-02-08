# Multi‑Stock Trading Strategy & Portfolio Analysis

This project evaluates four algorithmic trading strategies - **SMA, EMA, RSI, and ARIMA** - across five different stocks using a fixed investment of **$10,000 per ticker**.  
The objective is to compare trend‑following and forecasting‑based models, analyze portfolio growth, and understand how each strategy behaves under different market conditions.

---

##  Stocks Analyzed

- **BMNR**
- **BE**
- **OKLO**
- **IREN**
- **SNDK**

These tickers were selected to capture a mix of volatility, trend strength, and market behavior.

---

## Methodology

1. Download 2 years of daily OHLCV data using `yfinance`.  
2. Compute indicators: SMA, EMA, RSI.  
3. Build a trading environment to simulate buy/sell decisions.  
4. Apply four strategies:
   - **SMA** - simple trend‑following  
   - **EMA** - faster trend‑following  
   - **RSI** - mean‑reversion  
   - **ARIMA** - next‑day price forecasting  
5. Track portfolio value daily for each stock.  
6. Compute performance metrics:
   - Total Return  
   - Annualized Return  
   - Sharpe Ratio  
7. Plot multi‑stock portfolio comparison graphs for each strategy.

---

## Executing the Project
1) ##  How to Run the Python Script in VS Code

1. Open VS Code and load your project folder:
   - Go to **File → Open Folder**
   - Select the folder containing your `multi_stock_trading_strategy_analysis_portfolio_analysis.py` file

2. Create and activate a virtual environment:
   - Windows:
     ```
     python -m venv venv
     venv\Scripts\activate
     ```
   - Mac/Linux:
     ```
     python -m venv venv
     venv\Scripts\activate
     ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
4. Run the Main .py file
   ```
   python multi_stock_trading_strategy_analysis_portfolio_analysis.py
   ```

This will execute the full workflow: download data, compute indicators, run all strategies, simulate portfolio values, and print performance metrics.

2) ## How to Run (Google Colab)

You can run the entire project directly on **Google Colab** with no local setup.

1. Open **Google Colab** in your browser.  
2. Click **File → Upload Notebook**.  
3. Upload `Multi-Stock-Trading-Strategy-Analysis-Portfolio-Analysis.ipynb` from this repository.  
4. Run all cells from top to bottom.  

The notebook will automatically:

- Download stock data  
- Compute indicators  
- Run SMA, EMA, RSI, and ARIMA  
- Simulate portfolio values  
- Generate all graphs  
- Print performance metrics  
