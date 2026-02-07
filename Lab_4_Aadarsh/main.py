import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# Import your custom modules
from config import *
from data_collector import StockDataCollector
from feature_engineer import FeatureEngineer
from baseline_strategies import BaselineStrategies
from advanced_model import LSTMAttentionModel
from rl_agent import TradingAgent
from trading_env import TradingEnvironment
from evaluator import PortfolioEvaluator

def ensure_output_dirs():
    if not os.path.exists('results'):
        os.makedirs('results')

def main():
    print("="*60)
    print("ALGORITHMIC TRADING SYSTEM - MODULAR EXECUTION")
    print("="*60)
    
    ensure_output_dirs()
    
    # 1. Data Collection
    # -----------------------------------------------------------
    print("\n--- Phase 1: Data Collection ---")
    collector = StockDataCollector()
    # Using a subset of tickers for faster testing
    tickers = DEFAULT_TICKERS 
    data_dict = collector.collect_data(tickers, DEFAULT_START_DATE, DEFAULT_END_DATE)
    data_dict = collector.calculate_basic_features(data_dict)
    
    if not data_dict:
        print("No data collected. Exiting.")
        return

    # 2. Feature Engineering
    # -----------------------------------------------------------
    print("\n--- Phase 2: Feature Engineering ---")
    fe = FeatureEngineer()
    features_dict = fe.create_features(data_dict)

    filtered_features_dict = {}
    for ticker, df in features_dict.items():
        if len(df) > SEQUENCE_LENGTH:
            filtered_features_dict[ticker] = df
        else:
            print(f"  ⚠️ {ticker}: Skipped (only {len(df)} days, need {SEQUENCE_LENGTH}+)")

    # 3. Strategy Execution
    # -----------------------------------------------------------
    print("\n--- Phase 3: Strategy Generation ---")
    
    # A. Baseline Strategy
    print("\n[A] Running Baseline Strategies...")
    baseline = BaselineStrategies()
    baseline_signals = baseline.generate_signals(filtered_features_dict)
    
    # Prepare baseline signals for the environment (Standardize column name)
    baseline_ready = {}
    for ticker, df in baseline_signals.items():
        df_copy = df.copy()
        # The environment looks for 'final_signal'
        if 'Combined_Signal' in df_copy.columns:
            df_copy['final_signal'] = df_copy['Combined_Signal']
            baseline_ready[ticker] = df_copy

    # B. Advanced LSTM Model
    print("\n[B] Running LSTM-Attention Model...")
    lstm = LSTMAttentionModel(sequence_length=60)
    lstm_predictions = lstm.train_and_predict(filtered_features_dict)
    
    # Prepare LSTM signals
    lstm_ready = {}
    for ticker, df in lstm_predictions.items():
        df_copy = df.copy()
        # LSTM output is in 'Signal', map to 'final_signal'
        df_copy['final_signal'] = df_copy['Signal']
        lstm_ready[ticker] = df_copy

    # C. RL Agent Optimization (Optional - Applied on top of LSTM)
    print("\n[C] Running RL Agent Optimization...")
    rl_agent = TradingAgent()
    rl_ready = rl_agent.optimize_signals(filtered_features_dict, lstm_predictions)
    # rl_ready already has 'final_signal' column from the agent code

    # 4. Simulation & Backtesting
    # -----------------------------------------------------------
    print("\n--- Phase 4: Mock Trading Simulation ---")
    
    strategies = {
        'Baseline': baseline_ready,
        'LSTM': lstm_ready,
        'RL_Agent': rl_ready
    }
    
    results = {}
    
    for name, signals in strategies.items():
        print(f"\nSimulating: {name}")
        env = TradingEnvironment(initial_capital=INITIAL_CAPITAL)
        portfolio_history = env.simulate_trades(features_dict, signals)
        results[name] = portfolio_history
        
        # Save individual result
        portfolio_history.to_csv(f'results/portfolio_{name}.csv')

    # 5. Evaluation
    # -----------------------------------------------------------
    print("\n--- Phase 5: Performance Evaluation ---")
    evaluator = PortfolioEvaluator()
    
    # Compare all strategies
    comparison_df = evaluator.compare_strategies(
        list(results.values()), 
        list(results.keys())
    )
    
    print("\nStrategy Comparison:")
    print(comparison_df)
    comparison_df.to_csv('results/strategy_comparison.csv')
    
    # Plotting Comparison
    plt.figure(figsize=(12, 6))
    for name, history in results.items():
        plt.plot(history.index, history['portfolio_value'], label=name)
    
    plt.title('Portfolio Value Comparison')
    plt.xlabel('Date')
    plt.ylabel('Value ($)')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/performance_comparison.png')
    print("\n✓ Comparison plot saved to results/performance_comparison.png")

if __name__ == "__main__":
    main()