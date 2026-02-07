import pandas as pd
import numpy as np
from ta import add_all_ta_features
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
from ta.volatility import BollingerBands
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    def __init__(self):
        """Initialize feature engineer"""
        pass
    
    def create_features(self, data_dict):
        """
        Create technical indicators and features
        
        Parameters:
        -----------
        data_dict : dict
            Dictionary with stock data
            
        Returns:
        --------
        dict
            Dictionary with enhanced features for each ticker
        """
        print("Creating technical features...")
        
        features_dict = {}
        
        for ticker, df in data_dict.items():
            try:
                enhanced_df = self._add_technical_indicators(df)
                enhanced_df = self._add_statistical_features(enhanced_df)
                enhanced_df = self._add_lag_features(enhanced_df)
                enhanced_df = self._add_rolling_features(enhanced_df)
                
                # Drop NaN values
                enhanced_df = enhanced_df.dropna()
                
                features_dict[ticker] = enhanced_df
                print(f"  ✓ {ticker}: {enhanced_df.shape[1]} features created")
                
            except Exception as e:
                print(f"  ✗ Error processing {ticker}: {str(e)}")
        
        return features_dict
    
    def _add_technical_indicators(self, df):
        """
        Add common technical indicators
        """
        df_copy = df.copy()
        
        # Moving Averages
        df_copy['MA_5'] = df_copy['Close'].rolling(window=5).mean()
        df_copy['MA_20'] = df_copy['Close'].rolling(window=20).mean()
        df_copy['MA_50'] = df_copy['Close'].rolling(window=50).mean()
        df_copy['MA_200'] = df_copy['Close'].rolling(window=200).mean()
        
        # RSI
        rsi_indicator = RSIIndicator(close=df_copy['Close'], window=14)
        df_copy['RSI'] = rsi_indicator.rsi()
        
        # MACD
        macd_indicator = MACD(close=df_copy['Close'])
        df_copy['MACD'] = macd_indicator.macd()
        df_copy['MACD_Signal'] = macd_indicator.macd_signal()
        df_copy['MACD_Diff'] = macd_indicator.macd_diff()
        
        # Bollinger Bands
        bb_indicator = BollingerBands(close=df_copy['Close'])
        df_copy['BB_Upper'] = bb_indicator.bollinger_hband()
        df_copy['BB_Lower'] = bb_indicator.bollinger_lband()
        df_copy['BB_Width'] = bb_indicator.bollinger_wband()
        
        # Price position relative to indicators
        df_copy['Price_vs_MA20'] = (
            df_copy['Close'] / df_copy['MA_20'] - 1
        )
        df_copy['MA_Crossover'] = (
            df_copy['MA_5'] - df_copy['MA_20']
        )
        
        return df_copy
    
    def _add_statistical_features(self, df):
        """
        Add statistical features
        """
        df_copy = df.copy()
        
        # Volatility measures
        returns = df_copy['Close'].pct_change()
        df_copy['Returns_Std_20'] = returns.rolling(window=20).std()
        df_copy['Returns_Skew_20'] = returns.rolling(window=20).skew()
        df_copy['Returns_Kurt_20'] = returns.rolling(window=20).kurt()
        
        # Price statistics
        df_copy['High_Low_Ratio'] = (
            df_copy['High'] / df_copy['Low']
        )
        df_copy['Close_Open_Ratio'] = (
            df_copy['Close'] / df_copy['Open']
        )
        
        return df_copy
    
    def _add_lag_features(self, df, lags=[1, 2, 3, 5, 10]):
        """
        Add lagged features
        """
        df_copy = df.copy()
        
        for lag in lags:
            df_copy[f'Close_Lag_{lag}'] = df_copy['Close'].shift(lag)
            df_copy[f'Volume_Lag_{lag}'] = df_copy['Volume'].shift(lag)
            df_copy[f'Returns_Lag_{lag}'] = df_copy['Returns'].shift(lag)
        
        return df_copy
    
    def _add_rolling_features(self, df):
        """
        Add rolling window features
        """
        df_copy = df.copy()
        
        windows = [5, 10, 20, 50]
        
        for window in windows:
            # Rolling statistics
            df_copy[f'Roll_Mean_{window}'] = (
                df_copy['Close'].rolling(window=window).mean()
            )
            df_copy[f'Roll_Std_{window}'] = (
                df_copy['Close'].rolling(window=window).std()
            )
            df_copy[f'Roll_Min_{window}'] = (
                df_copy['Close'].rolling(window=window).min()
            )
            df_copy[f'Roll_Max_{window}'] = (
                df_copy['Close'].rolling(window=window).max()
            )
            
            # Rolling returns
            df_copy[f'Roll_Return_{window}'] = (
                df_copy['Close'].pct_change(window)
            )
        
        return df_copy
    
    def prepare_features_for_model(self, features_dict, target_ticker='AAPL'):
        """
        Prepare features for machine learning model
        
        Parameters:
        -----------
        features_dict : dict
            Dictionary with features
        target_ticker : str
            Target ticker for prediction
            
        Returns:
        --------
        tuple
            (X_features, y_target, feature_names)
        """
        if target_ticker not in features_dict:
            target_ticker = list(features_dict.keys())[0]
        
        df = features_dict[target_ticker].copy()
        
        # Define target: Next day's return
        df['Target'] = df['Close'].shift(-1) / df['Close'] - 1
        
        # Remove columns with too many NaN values
        df = df.dropna(axis=1, thresh=int(0.8 * len(df)))
        
        # Fill remaining NaN values
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # Separate features and target
        exclude_columns = ['Target', 'Returns', 'Log_Returns']
        feature_columns = [
            col for col in df.columns 
            if col not in exclude_columns and 'Target' not in col
        ]
        
        X = df[feature_columns].values
        y = df['Target'].values
        
        return X, y, feature_columns