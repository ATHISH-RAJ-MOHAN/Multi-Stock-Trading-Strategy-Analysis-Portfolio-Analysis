import yfinance as yf
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class StockDataCollector:
    def __init__(self):
        """Initialize data collector"""
        pass
    
    def collect_data(self, tickers, start_date, end_date):
        """
        Collect stock data from Yahoo Finance
        """
        print(f"Collecting data for {tickers}...")
        
        data_dict = {}
        
        for ticker in tickers:
            try:
                print(f"  Downloading {ticker}...")
                stock_data = yf.download(
                    ticker, 
                    start=start_date, 
                    end=end_date,
                    progress=False
                )
                
                if not stock_data.empty:
                    # FIX: Handle MultiIndex columns if present
                    if isinstance(stock_data.columns, pd.MultiIndex):
                        stock_data.columns = stock_data.columns.get_level_values(0)
                    
                    # Ensure we have a proper index
                    if not isinstance(stock_data.index, pd.DatetimeIndex):
                        stock_data.index = pd.to_datetime(stock_data.index)
                        
                    data_dict[ticker] = stock_data
                    print(f"    ✓ {ticker}: {len(stock_data)} days collected")
                else:
                    print(f"    ✗ {ticker}: No data available")
                    
            except Exception as e:
                print(f"    ✗ Error downloading {ticker}: {str(e)}")
        
        return data_dict

    def calculate_basic_features(self, data_dict):
        """
        Calculate basic features for each stock
        """
        enhanced_data = {}
        
        for ticker, df in data_dict.items():
            # Create a copy
            enhanced_df = df.copy()
            
            # Ensure Close is available (handle Adj Close fallback)
            if 'Close' not in enhanced_df.columns and 'Adj Close' in enhanced_df.columns:
                enhanced_df['Close'] = enhanced_df['Adj Close']
                
            if 'Close' in enhanced_df.columns:
                # Calculate returns
                enhanced_df['Returns'] = enhanced_df['Close'].pct_change()
                enhanced_df['Log_Returns'] = np.log(
                    enhanced_df['Close'] / enhanced_df['Close'].shift(1)
                )
                
                # Calculate volatility (20-day rolling)
                enhanced_df['Volatility'] = enhanced_df['Returns'].rolling(
                    window=20
                ).std() * np.sqrt(252)
                
                # Volume indicators
                if 'Volume' in enhanced_df.columns:
                    enhanced_df['Volume_MA'] = enhanced_df['Volume'].rolling(
                        window=20
                    ).mean()
                    enhanced_df['Volume_Ratio'] = (
                        enhanced_df['Volume'] / enhanced_df['Volume_MA']
                    )
                
                enhanced_data[ticker] = enhanced_df
            else:
                print(f"Warning: 'Close' price not found for {ticker}")
        
        return enhanced_data