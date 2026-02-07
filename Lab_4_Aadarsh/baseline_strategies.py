import pandas as pd
import numpy as np
from typing import Dict, Tuple

class BaselineStrategies:
    def __init__(self):
        """Initialize baseline trading strategies"""
        pass
    
    def generate_signals(self, features_dict: Dict) -> Dict:
        """
        Generate trading signals using baseline strategies
        
        Parameters:
        -----------
        features_dict : dict
            Dictionary with features for each ticker
            
        Returns:
        --------
        dict
            Dictionary with signals for each ticker
        """
        print("Generating baseline strategy signals...")
        
        signals_dict = {}
        
        for ticker, df in features_dict.items():
            try:
                signals_df = self._apply_all_strategies(df)
                signals_dict[ticker] = signals_df
                print(f"  ✓ {ticker}: Baseline signals generated")
            except Exception as e:
                print(f"  ✗ Error generating signals for {ticker}: {str(e)}")
        
        return signals_dict
    
    def _apply_all_strategies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all baseline strategies to generate signals
        
        Parameters:
        -----------
        df : DataFrame
            Stock data with features
            
        Returns:
        --------
        DataFrame
            DataFrame with signals from all strategies
        """
        signals_df = df.copy()
        
        # Strategy 1: Moving Average Crossover
        signals_df['MA_Signal'] = self._ma_crossover_strategy(
            signals_df['Close'],
            short_window=5,
            long_window=20
        )
        
        # Strategy 2: RSI Strategy
        if 'RSI' in signals_df.columns:
            signals_df['RSI_Signal'] = self._rsi_strategy(
                signals_df['RSI'],
                oversold=30,
                overbought=70
            )
        
        # Strategy 3: MACD Strategy
        if 'MACD' in signals_df.columns and 'MACD_Signal' in signals_df.columns:
            signals_df['MACD_Signal'] = self._macd_strategy(
                signals_df['MACD'],
                signals_df['MACD_Signal']
            )
        
        # Strategy 4: Bollinger Bands Strategy
        if all(col in signals_df.columns for col in ['BB_Upper', 'BB_Lower']):
            signals_df['BB_Signal'] = self._bollinger_bands_strategy(
                signals_df['Close'],
                signals_df['BB_Upper'],
                signals_df['BB_Lower']
            )
        
        # Combined signal (simple voting)
        signal_columns = [
            col for col in signals_df.columns 
            if col.endswith('_Signal')
        ]
        
        if signal_columns:
            signals_df['Combined_Signal'] = signals_df[signal_columns].mean(
                axis=1
            ).apply(lambda x: 1 if x > 0.3 else (-1 if x < -0.3 else 0))
        
        return signals_df
    
    def _ma_crossover_strategy(
        self, 
        prices: pd.Series, 
        short_window: int = 5, 
        long_window: int = 20
    ) -> pd.Series:
        """
        Moving Average Crossover Strategy
        
        Parameters:
        -----------
        prices : Series
            Price data
        short_window : int
            Short moving average window
        long_window : int
            Long moving average window
            
        Returns:
        --------
        Series
            Trading signals (1: buy, -1: sell, 0: hold)
        """
        ma_short = prices.rolling(window=short_window).mean()
        ma_long = prices.rolling(window=long_window).mean()
        
        signals = pd.Series(0, index=prices.index)
        
        # Buy when short MA crosses above long MA
        buy_signals = (ma_short > ma_long) & (
            ma_short.shift(1) <= ma_long.shift(1)
        )
        
        # Sell when short MA crosses below long MA
        sell_signals = (ma_short < ma_long) & (
            ma_short.shift(1) >= ma_long.shift(1)
        )
        
        signals[buy_signals] = 1
        signals[sell_signals] = -1
        
        return signals
    
    def _rsi_strategy(
        self, 
        rsi: pd.Series, 
        oversold: float = 30, 
        overbought: float = 70
    ) -> pd.Series:
        """
        RSI Strategy
        
        Parameters:
        -----------
        rsi : Series
            RSI values
        oversold : float
            Oversold threshold
        overbought : float
            Overbought threshold
            
        Returns:
        --------
        Series
            Trading signals
        """
        signals = pd.Series(0, index=rsi.index)
        
        # Buy when RSI crosses above oversold level
        buy_signals = (rsi > oversold) & (rsi.shift(1) <= oversold)
        
        # Sell when RSI crosses below overbought level
        sell_signals = (rsi < overbought) & (rsi.shift(1) >= overbought)
        
        signals[buy_signals] = 1
        signals[sell_signals] = -1
        
        return signals
    
    def _macd_strategy(
        self, 
        macd: pd.Series, 
        signal_line: pd.Series
    ) -> pd.Series:
        """
        MACD Strategy
        
        Parameters:
        -----------
        macd : Series
            MACD values
        signal_line : Series
            MACD signal line values
            
        Returns:
        --------
        Series
            Trading signals
        """
        signals = pd.Series(0, index=macd.index)
        
        # Buy when MACD crosses above signal line
        buy_signals = (macd > signal_line) & (
            macd.shift(1) <= signal_line.shift(1)
        )
        
        # Sell when MACD crosses below signal line
        sell_signals = (macd < signal_line) & (
            macd.shift(1) >= signal_line.shift(1)
        )
        
        signals[buy_signals] = 1
        signals[sell_signals] = -1
        
        return signals
    
    def _bollinger_bands_strategy(
        self, 
        prices: pd.Series, 
        upper_band: pd.Series, 
        lower_band: pd.Series
    ) -> pd.Series:
        """
        Bollinger Bands Strategy
        
        Parameters:
        -----------
        prices : Series
            Price data
        upper_band : Series
            Upper Bollinger Band
        lower_band : Series
            Lower Bollinger Band
            
        Returns:
        --------
        Series
            Trading signals
        """
        signals = pd.Series(0, index=prices.index)
        
        # Buy when price touches lower band
        buy_signals = prices <= lower_band
        
        # Sell when price touches upper band
        sell_signals = prices >= upper_band
        
        signals[buy_signals] = 1
        signals[sell_signals] = -1
        
        return signals
    
    def calculate_returns(
        self, 
        signals: pd.Series, 
        prices: pd.Series, 
        transaction_cost: float = 0.001
    ) -> pd.Series:
        """
        Calculate strategy returns
        
        Parameters:
        -----------
        signals : Series
            Trading signals
        prices : Series
            Price data
        transaction_cost : float
            Transaction cost percentage
            
        Returns:
        --------
        Series
            Strategy returns
        """
        # Calculate daily returns
        daily_returns = prices.pct_change()
        
        # Shift signals to avoid look-ahead bias
        strategy_returns = signals.shift(1) * daily_returns
        
        # Apply transaction costs when signal changes
        signal_changes = signals.diff().abs()
        strategy_returns -= signal_changes * transaction_cost
        
        return strategy_returns