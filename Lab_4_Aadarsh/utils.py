"""
Utility functions for the trading system
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle
import json
import os

def save_object(obj, filename):
    """
    Save Python object to pickle file
    
    Parameters:
    -----------
    obj : object
        Python object to save
    filename : str
        Filename for saving
    """
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

def load_object(filename):
    """
    Load Python object from pickle file
    
    Parameters:
    -----------
    filename : str
        Filename to load from
    
    Returns:
    --------
    object
        Loaded Python object
    """
    with open(filename, 'rb') as f:
        return pickle.load(f)

def calculate_returns(prices, period=1):
    """
    Calculate returns for given period
    
    Parameters:
    -----------
    prices : Series or array
        Price data
    period : int
        Return period
    
    Returns:
    --------
    Series or array
        Returns
    """
    return prices.pct_change(period)

def calculate_log_returns(prices):
    """
    Calculate log returns
    
    Parameters:
    -----------
    prices : Series or array
        Price data
    
    Returns:
    --------
    Series or array
        Log returns
    """
    return np.log(prices / prices.shift(1))

def calculate_volatility(returns, window=20):
    """
    Calculate rolling volatility
    
    Parameters:
    -----------
    returns : Series
        Return data
    window : int
        Rolling window
    
    Returns:
    --------
    Series
        Volatility
    """
    return returns.rolling(window=window).std() * np.sqrt(252)

def calculate_drawdown(portfolio_values):
    """
    Calculate drawdown series
    
    Parameters:
    -----------
    portfolio_values : Series
        Portfolio values over time
    
    Returns:
    --------
    Series
        Drawdown percentages
    """
    rolling_max = portfolio_values.expanding().max()
    drawdown = (portfolio_values - rolling_max) / rolling_max
    return drawdown

def normalize_data(data, method='standard'):
    """
    Normalize data using specified method
    
    Parameters:
    -----------
    data : array-like
        Data to normalize
    method : str
        Normalization method ('standard', 'minmax', 'robust')
    
    Returns:
    --------
    array-like
        Normalized data
    """
    if method == 'standard':
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        return (data - mean) / (std + 1e-8)
    
    elif method == 'minmax':
        min_val = np.min(data, axis=0)
        max_val = np.max(data, axis=0)
        return (data - min_val) / (max_val - min_val + 1e-8)
    
    elif method == 'robust':
        median = np.median(data, axis=0)
        q75, q25 = np.percentile(data, [75, 25], axis=0)
        iqr = q75 - q25
        return (data - median) / (iqr + 1e-8)
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def create_lagged_features(data, lags):
    """
    Create lagged features
    
    Parameters:
    -----------
    data : array-like
        Input data
    lags : list
        List of lag values
    
    Returns:
    --------
    DataFrame
        Data with lagged features
    """
    df = pd.DataFrame(data)
    for lag in lags:
        df[f'lag_{lag}'] = df.iloc[:, 0].shift(lag)
    return df.dropna()

def calculate_correlation_matrix(returns_data):
    """
    Calculate correlation matrix
    
    Parameters:
    -----------
    returns_data : DataFrame
        Returns data for multiple assets
    
    Returns:
    --------
    DataFrame
        Correlation matrix
    """
    return returns_data.corr()

def calculate_beta(asset_returns, market_returns):
    """
    Calculate beta coefficient
    
    Parameters:
    -----------
    asset_returns : Series
        Asset returns
    market_returns : Series
        Market returns
    
    Returns:
    --------
    float
        Beta coefficient
    """
    covariance = np.cov(asset_returns, market_returns)[0, 1]
    market_variance = np.var(market_returns)
    return covariance / market_variance

def calculate_alpha(asset_returns, market_returns, risk_free_rate=0.02):
    """
    Calculate alpha (excess return)
    
    Parameters:
    -----------
    asset_returns : Series
        Asset returns
    market_returns : Series
        Market returns
    risk_free_rate : float
        Risk-free rate
    
    Returns:
    --------
    float
        Alpha
    """
    beta = calculate_beta(asset_returns, market_returns)
    expected_return = risk_free_rate + beta * (market_returns.mean() - risk_free_rate)
    actual_return = asset_returns.mean()
    return actual_return - expected_return

def format_currency(value):
    """
    Format value as currency
    
    Parameters:
    -----------
    value : float
        Value to format
    
    Returns:
    --------
    str
        Formatted currency string
    """
    return f"${value:,.2f}"

def format_percentage(value):
    """
    Format value as percentage
    
    Parameters:
    -----------
    value : float
        Value to format
    
    Returns:
    --------
    str
        Formatted percentage string
    """
    return f"{value:.2%}"

def get_trading_days(start_date, end_date):
    """
    Get list of trading days between dates
    
    Parameters:
    -----------
    start_date : str
        Start date
    end_date : str
        End date
    
    Returns:
    --------
    DatetimeIndex
        Trading days
    """
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    return dates

def check_market_hours():
    """
    Check if current time is within market hours
    
    Returns:
    --------
    bool
        True if market is open
    """
    now = datetime.now()
    market_open = datetime(now.year, now.month, now.day, 9, 30)
    market_close = datetime(now.year, now.month, now.day, 16, 0)
    
    # Check if weekday (Monday=0, Friday=4)
    if now.weekday() > 4:
        return False
    
    return market_open <= now <= market_close

def ensure_directory(directory):
    """
    Ensure directory exists
    
    Parameters:
    -----------
    directory : str
        Directory path
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_config(config_file='config.json'):
    """
    Load configuration from JSON file
    
    Parameters:
    -----------
    config_file : str
        Configuration file path
    
    Returns:
    --------
    dict
        Configuration dictionary
    """
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            return json.load(f)
    else:
        return {}