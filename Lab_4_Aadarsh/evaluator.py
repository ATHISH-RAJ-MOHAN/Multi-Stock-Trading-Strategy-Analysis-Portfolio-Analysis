import numpy as np
import pandas as pd
from scipy import stats

class PortfolioEvaluator:
    def __init__(self, risk_free_rate=0.02):
        """
        Initialize portfolio evaluator
        
        Parameters:
        -----------
        risk_free_rate : float
            Annual risk-free rate for Sharpe ratio
        """
        self.risk_free_rate = risk_free_rate
        
    def evaluate_performance(self, portfolio_history):
        """
        Evaluate portfolio performance
        
        Parameters:
        -----------
        portfolio_history : DataFrame
            Portfolio history with daily values
            
        Returns:
        --------
        dict
            Dictionary with performance metrics
        """
        print("Evaluating portfolio performance...")
        
        if portfolio_history.empty:
            return {}
        
        # Calculate metrics
        metrics = {}
        
        # Basic metrics
        initial_value = portfolio_history['portfolio_value'].iloc[0]
        final_value = portfolio_history['portfolio_value'].iloc[-1]
        total_return = (final_value / initial_value) - 1
        
        metrics['initial_value'] = initial_value
        metrics['final_value'] = final_value
        metrics['total_return'] = total_return
        
        # Daily returns
        daily_returns = portfolio_history['daily_returns'].dropna()
        
        if len(daily_returns) > 0:
            # Annualized metrics
            trading_days = 252  # Approximate trading days in a year
            annualized_return = self._calculate_annualized_return(
                daily_returns, trading_days
            )
            annualized_volatility = daily_returns.std() * np.sqrt(trading_days)
            
            # Sharpe ratio
            sharpe_ratio = self._calculate_sharpe_ratio(
                daily_returns, trading_days
            )
            
            # Sortino ratio
            sortino_ratio = self._calculate_sortino_ratio(
                daily_returns, trading_days
            )
            
            # Maximum drawdown
            max_drawdown = portfolio_history['drawdown'].min()
            
            # Calmar ratio
            calmar_ratio = (
                annualized_return / abs(max_drawdown) 
                if max_drawdown != 0 else np.nan
            )
            
            # Win rate
            win_rate = (daily_returns > 0).sum() / len(daily_returns)
            
            # Profit factor
            profit_factor = self._calculate_profit_factor(daily_returns)
            
            # Skewness and kurtosis
            skewness = daily_returns.skew()
            kurtosis = daily_returns.kurtosis()
            
            # Value at Risk (VaR)
            var_95 = np.percentile(daily_returns, 5)
            cvar_95 = daily_returns[daily_returns <= var_95].mean()
            
            # Update metrics
            metrics.update({
                'annualized_return': annualized_return,
                'annualized_volatility': annualized_volatility,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'max_drawdown': max_drawdown,
                'calmar_ratio': calmar_ratio,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'skewness': skewness,
                'kurtosis': kurtosis,
                'var_95': var_95,
                'cvar_95': cvar_95
            })
        
        # Print summary
        self._print_summary(metrics)
        
        return metrics
    
    def _calculate_annualized_return(self, daily_returns, trading_days):
        """Calculate annualized return"""
        cumulative_return = (1 + daily_returns).prod() - 1
        years = len(daily_returns) / trading_days
        annualized_return = (1 + cumulative_return) ** (1/years) - 1
        return annualized_return
    
    def _calculate_sharpe_ratio(self, daily_returns, trading_days):
        """Calculate Sharpe ratio"""
        excess_returns = daily_returns - (self.risk_free_rate / trading_days)
        sharpe = (
            excess_returns.mean() / daily_returns.std() * np.sqrt(trading_days)
        )
        return sharpe
    
    def _calculate_sortino_ratio(self, daily_returns, trading_days):
        """Calculate Sortino ratio"""
        downside_returns = daily_returns[daily_returns < 0]
        if len(downside_returns) == 0:
            return np.nan
        
        downside_deviation = downside_returns.std()
        if downside_deviation == 0:
            return np.nan
        
        excess_returns = daily_returns.mean() - (self.risk_free_rate / trading_days)
        sortino = excess_returns / downside_deviation * np.sqrt(trading_days)
        return sortino
    
    def _calculate_profit_factor(self, daily_returns):
        """Calculate profit factor"""
        winning_trades = daily_returns[daily_returns > 0].sum()
        losing_trades = abs(daily_returns[daily_returns < 0].sum())
        
        if losing_trades == 0:
            return np.inf
        
        return winning_trades / losing_trades
    
    def _print_summary(self, metrics):
        """Print performance summary"""
        print("\nPerformance Summary:")
        print("-" * 40)
        
        print(f"Total Return: {metrics.get('total_return', 0):.2%}")
        print(f"Annualized Return: {metrics.get('annualized_return', 0):.2%}")
        print(f"Annualized Volatility: {metrics.get('annualized_volatility', 0):.2%}")
        print(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}")
        print(f"Sortino Ratio: {metrics.get('sortino_ratio', 0):.3f}")
        print(f"Maximum Drawdown: {metrics.get('max_drawdown', 0):.2f}%")
        print(f"Calmar Ratio: {metrics.get('calmar_ratio', 0):.3f}")
        print(f"Win Rate: {metrics.get('win_rate', 0):.2%}")
        print(f"Profit Factor: {metrics.get('profit_factor', 0):.3f}")
        print(f"Value at Risk (95%): {metrics.get('var_95', 0):.4f}")
        
        if 'skewness' in metrics:
            print(f"Return Skewness: {metrics['skewness']:.3f}")
        if 'kurtosis' in metrics:
            print(f"Return Kurtosis: {metrics['kurtosis']:.3f}")
        
        print("-" * 40)
    
    def compare_strategies(self, portfolio_histories, strategy_names):
        """
        Compare multiple trading strategies
        
        Parameters:
        -----------
        portfolio_histories : list
            List of portfolio history DataFrames
        strategy_names : list
            List of strategy names
            
        Returns:
        --------
        DataFrame
            Comparison table
        """
        comparison_data = []
        
        for history, name in zip(portfolio_histories, strategy_names):
            metrics = self.evaluate_performance(history)
            metrics['strategy'] = name
            comparison_data.append(metrics)
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df.set_index('strategy', inplace=True)
        
        # Select key metrics for display
        key_metrics = [
            'total_return', 'annualized_return', 'sharpe_ratio',
            'max_drawdown', 'win_rate', 'profit_factor'
        ]
        
        available_metrics = [m for m in key_metrics if m in comparison_df.columns]
        
        return comparison_df[available_metrics]