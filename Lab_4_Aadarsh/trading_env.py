import pandas as pd
import numpy as np
from datetime import datetime

class TradingEnvironment:
    def __init__(self, initial_capital=100000, transaction_cost=0.001):
        """
        Initialize trading environment
        
        Parameters:
        -----------
        initial_capital : float
            Initial capital in dollars
        transaction_cost : float
            Transaction cost as percentage of trade value
        """
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.reset()
    
    def reset(self):
        """Reset environment to initial state"""
        self.capital = self.initial_capital
        self.positions = {}  # {ticker: {'shares': quantity, 'avg_price': price}}
        self.portfolio_value = self.initial_capital
        self.trades_history = []
        self.portfolio_history = []
        
    def simulate_trades(self, features_dict, signals_dict):
        """
        Simulate trading based on signals
        
        Parameters:
        -----------
        features_dict : dict
            Dictionary with features for each ticker
        signals_dict : dict
            Dictionary with trading signals
            
        Returns:
        --------
        DataFrame
            Portfolio history with daily values
        """
        print("Simulating trades...")
        
        # Get common dates across all tickers
        all_dates = set()
        for ticker in features_dict.keys():
            if ticker in signals_dict:
                all_dates.update(features_dict[ticker].index)
        
        all_dates = sorted(list(all_dates))
        
        # Initialize portfolio tracking
        portfolio_history = []
        
        for current_date in all_dates:
            # Update portfolio value based on current prices
            self._update_portfolio_value(features_dict, current_date)
            
            # Execute trades for each ticker
            for ticker in features_dict.keys():
                if ticker in signals_dict:
                    if current_date in signals_dict[ticker].index:
                        signal = signals_dict[ticker].loc[current_date, 'final_signal']
                        
                        if signal != 0:
                            current_price = self._get_current_price(
                                features_dict[ticker], current_date
                            )
                            
                            if current_price:
                                self._execute_trade(
                                    ticker, signal, current_price, current_date
                                )
            
            # Record portfolio state
            portfolio_record = {
                'date': current_date,
                'portfolio_value': self.portfolio_value,
                'capital': self.capital,
                'positions_value': self.portfolio_value - self.capital,
                'num_positions': len(self.positions)
            }
            
            # Add position details
            for ticker, position in self.positions.items():
                portfolio_record[f'{ticker}_shares'] = position['shares']
                portfolio_record[f'{ticker}_value'] = (
                    position['shares'] * 
                    self._get_current_price(features_dict[ticker], current_date)
                )
            
            portfolio_history.append(portfolio_record)
        
        # Create portfolio DataFrame
        portfolio_df = pd.DataFrame(portfolio_history)
        portfolio_df.set_index('date', inplace=True)
        
        # Calculate returns
        portfolio_df['daily_returns'] = portfolio_df['portfolio_value'].pct_change()
        portfolio_df['cumulative_returns'] = (
            portfolio_df['portfolio_value'] / self.initial_capital - 1
        )
        
        # Calculate drawdown
        portfolio_df['drawdown'] = self._calculate_drawdown(
            portfolio_df['portfolio_value']
        )
        
        print(f"  ✓ Simulation complete")
        print(f"    Final Portfolio Value: ${self.portfolio_value:,.2f}")
        print(f"    Total Return: {(self.portfolio_value/self.initial_capital - 1):.2%}")
        print(f"    Number of Trades: {len(self.trades_history)}")
        
        return portfolio_df
    
    def _get_current_price(self, features_df, date):
        """Get current price for a ticker"""
        if date in features_df.index:
            return features_df.loc[date, 'Close']
        return None
    
    def _update_portfolio_value(self, features_dict, date):
        """Update portfolio value based on current prices"""
        positions_value = 0
        
        for ticker, position in self.positions.items():
            if ticker in features_dict:
                current_price = self._get_current_price(
                    features_dict[ticker], date
                )
                if current_price:
                    positions_value += position['shares'] * current_price
        
        self.portfolio_value = self.capital + positions_value
    
    def _execute_trade(self, ticker, signal, price, date):
        """
        Execute a trade
        
        Parameters:
        -----------
        ticker : str
            Stock ticker
        signal : int
            Trading signal (1: buy, -1: sell)
        price : float
            Current price
        date : datetime
            Trade date
        """
        # Calculate position size (simple fixed fraction)
        position_size = self.portfolio_value * 0.1  # 10% per position
        
        if signal == 1:  # Buy
            # Calculate number of shares to buy
            shares_to_buy = int(position_size / price)
            
            if shares_to_buy > 0 and self.capital >= shares_to_buy * price:
                # Execute buy
                cost = shares_to_buy * price * (1 + self.transaction_cost)
                
                if ticker in self.positions:
                    # Average down/up existing position
                    old_shares = self.positions[ticker]['shares']
                    old_avg_price = self.positions[ticker]['avg_price']
                    total_shares = old_shares + shares_to_buy
                    new_avg_price = (
                        (old_shares * old_avg_price) + 
                        (shares_to_buy * price)
                    ) / total_shares
                    
                    self.positions[ticker] = {
                        'shares': total_shares,
                        'avg_price': new_avg_price
                    }
                else:
                    # New position
                    self.positions[ticker] = {
                        'shares': shares_to_buy,
                        'avg_price': price
                    }
                
                self.capital -= cost
                
                # Record trade
                self.trades_history.append({
                    'date': date,
                    'ticker': ticker,
                    'action': 'BUY',
                    'shares': shares_to_buy,
                    'price': price,
                    'cost': cost
                })
        
        elif signal == -1:  # Sell
            if ticker in self.positions:
                # Sell entire position
                shares_to_sell = self.positions[ticker]['shares']
                proceeds = shares_to_sell * price * (1 - self.transaction_cost)
                
                # Update capital and remove position
                self.capital += proceeds
                del self.positions[ticker]
                
                # Record trade
                self.trades_history.append({
                    'date': date,
                    'ticker': ticker,
                    'action': 'SELL',
                    'shares': shares_to_sell,
                    'price': price,
                    'proceeds': proceeds
                })
    
    def _calculate_drawdown(self, portfolio_values):
        """Calculate drawdown series"""
        rolling_max = portfolio_values.expanding().max()
        drawdown = (portfolio_values - rolling_max) / rolling_max * 100
        return drawdown
    
    def get_trades_summary(self):
        """Get summary of all trades"""
        if not self.trades_history:
            return pd.DataFrame()
        
        trades_df = pd.DataFrame(self.trades_history)
        
        # Calculate trade metrics
        buy_trades = trades_df[trades_df['action'] == 'BUY']
        sell_trades = trades_df[trades_df['action'] == 'SELL']
        
        summary = {
            'total_trades': len(trades_df),
            'buy_trades': len(buy_trades),
            'sell_trades': len(sell_trades),
            'total_volume': trades_df['shares'].sum() if 'shares' in trades_df.columns else 0
        }
        
        return summary