import numpy as np
import pandas as pd
import random

class TradingAgent:
    def __init__(self, state_size=10, action_size=3):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.95 
        self.epsilon = 1.0 
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.q_table = {} 
        
    def get_state(self, row):
        """
        Discretize features into a state string. 
        FIXED: Uses flexible column lookups to avoid KeyError.
        """
        # 1. Determine RSI state (Overbought/Oversold)
        # Look for any column containing 'RSI'
        rsi_col = [c for c in row.index if 'RSI' in c]
        rsi_val = row[rsi_col[0]] if rsi_col else 50
        rsi_state = 'L' if rsi_val < 30 else 'H' if rsi_val > 70 else 'N'
        
        # 2. Determine Trend state (Price vs Moving Average)
        # Look for EMA or MA columns
        ma_col = [c for c in row.index if 'EMA' in c or 'MA' in c]
        trend_state = 'UP'
        if ma_col and row['Close'] < row[ma_col[0]]:
            trend_state = 'DOWN'
            
        # 3. Determine Momentum (Current price vs Previous price)
        # We use the 'Predicted' value from LSTM if available
        pred_state = 'NEU'
        if 'Predicted' in row:
            if row['Predicted'] > 0.001: pred_state = 'BULL'
            elif row['Predicted'] < -0.001: pred_state = 'BEAR'

        return f"{rsi_state}_{trend_state}_{pred_state}"

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        return np.argmax(self.q_table.get(state, np.zeros(self.action_size)))

    def learn(self, state, action, reward, next_state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_size)
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(self.action_size)
            
        old_value = self.q_table[state][action]
        next_max = np.max(self.q_table[next_state])
        new_value = (1 - self.learning_rate) * old_value + \
                   self.learning_rate * (reward + self.gamma * next_max)
        self.q_table[state][action] = new_value

    def optimize_signals(self, features_dict, predictions_dict):
        print("Optimizing signals with RL agent...")
        optimized_signals = {}
        
        for ticker, df in features_dict.items():
            print(f"  Optimizing {ticker}...")
            
            # Merge LSTM predictions into features so the RL agent can use them
            working_df = df.copy()
            if ticker in predictions_dict:
                # Align indices and grab 'Predicted' column
                lstm_data = predictions_dict[ticker][['Predicted']]
                working_df = working_df.join(lstm_data, how='inner')
            
            # Pre-training: Iterate through history to fill Q-table
            for episode in range(10): 
                state = self.get_state(working_df.iloc[0])
                for i in range(len(working_df) - 1):
                    row = working_df.iloc[i]
                    next_row = working_df.iloc[i+1]
                    
                    action = self.get_action(state)
                    
                    # Reward: Profitability - Transaction Costs
                    daily_return = (next_row['Close'] - row['Close']) / row['Close']
                    
                    if action == 1: # BUY
                        reward = (daily_return * 100) - 0.1 # Penalty for cost
                    elif action == 2: # SELL
                        reward = (-daily_return * 100) - 0.1 # Penalty for cost
                    else: # HOLD
                        reward = 0
                        
                    next_state = self.get_state(next_row)
                    self.learn(state, action, reward, next_state)
                    state = next_state
                
                self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            # Final Signal Generation
            final_actions = []
            for i in range(len(working_df)):
                state = self.get_state(working_df.iloc[i])
                # Take best learned action (epsilon=0)
                action = np.argmax(self.q_table.get(state, np.zeros(self.action_size)))
                
                signal = 0
                if action == 1: signal = 1
                elif action == 2: signal = -1
                final_actions.append(signal)
            
            result_df = working_df.copy()
            result_df['final_signal'] = final_actions
            optimized_signals[ticker] = result_df
            
        return optimized_signals