import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, Input, 
    Attention, Concatenate, LayerNormalization,
    Bidirectional, Conv1D, MaxPooling1D, Flatten
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)
import warnings
warnings.filterwarnings('ignore')

class LSTMAttentionModel:
    def __init__(self, sequence_length=60, n_features=50):
        """
        Initialize LSTM-Attention model
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = MinMaxScaler(feature_range=(-1, 1))
        
    def build_model(self):
        """
        Build LSTM-Attention model architecture
        """
        # Input layer
        inputs = Input(shape=(self.sequence_length, self.n_features))
        
        # Bidirectional LSTM layers
        lstm1 = Bidirectional(LSTM(128, return_sequences=True))(inputs)
        lstm1 = Dropout(0.2)(lstm1)
        lstm1 = LayerNormalization()(lstm1)
        
        lstm2 = Bidirectional(LSTM(64, return_sequences=True))(lstm1)
        lstm2 = Dropout(0.2)(lstm2)
        lstm2 = LayerNormalization()(lstm2)
        
        # Attention Mechanism
        # Self-attention
        attention = Attention()([lstm2, lstm2])
        attention = LayerNormalization()(attention)
        
        # Combine LSTM and Attention
        context = Concatenate()([lstm2, attention])
        context = Flatten()(context)
        
        # Dense layers for prediction
        x = Dense(64, activation='relu')(context)
        x = Dropout(0.2)(x)
        x = Dense(32, activation='relu')(x)
        
        # Output layer (Predicting returns)
        outputs = Dense(1, activation='linear')(x)
        
        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        
    def prepare_data(self, features_dict):
        """
        Prepare data for training
        """
        # Combine all data
        all_X = []
        all_y = []
        
        for ticker, df in features_dict.items():
            # Create sequences
            data = df.drop(['Target'], axis=1, errors='ignore')
            
            # Select numeric columns only
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            data = data[numeric_cols]
            
            # Ensure target exists (Next day return)
            if 'Target' not in df.columns:
                target = df['Close'].shift(-1) / df['Close'] - 1
                target = target.fillna(0)
            else:
                target = df['Target']
            
            values = data.values
            target_values = target.values
            
            # Create sequences
            for i in range(self.sequence_length, len(values)):
                all_X.append(values[i-self.sequence_length:i])
                all_y.append(target_values[i-1]) # Use aligned target
                
        return np.array(all_X), np.array(all_y)
        
    def train_and_predict(self, features_dict):
        """
        Train model and generate predictions
        """
        print("Training LSTM-Attention model...")
        
        predictions_dict = {}
        
        # Prepare data
        X, y = self.prepare_data(features_dict)
        
        # Scale data
        # Reshape for scaling: (samples * seq_len, features)
        n_samples, seq_len, n_feat = X.shape
        X_reshaped = X.reshape(-1, n_feat)
        
        # Fit scaler on all data
        self.scaler_X.fit(X_reshaped)
        
        # Train/Test split
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Reshape and transform
        X_train_scaled = self.scaler_X.transform(X_train.reshape(-1, n_feat)).reshape(X_train.shape)
        X_test_scaled = self.scaler_X.transform(X_test.reshape(-1, n_feat)).reshape(X_test.shape)
        
        # Update n_features based on actual data
        self.n_features = X.shape[2]
        
        # Build and train model
        self.build_model()
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss', 
                patience=5,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-6
            )
        ]
        
        self.model.fit(
            X_train_scaled, y_train,
            validation_data=(X_test_scaled, y_test),
            epochs=20, # Reduced for speed in assignment
            batch_size=64,
            callbacks=callbacks,
            verbose=1
        )
        
        # Generate predictions for each ticker
        for ticker, df in features_dict.items():
            print(f"  Processing {ticker}...")
            
            # Prepare sequences for this ticker
            X_ticker = []
            data = df.select_dtypes(include=[np.number])
            if 'Target' in data.columns:
                data = data.drop('Target', axis=1)
                
            values = data.values
            
            # Need at least sequence_length data points
            if len(values) > self.sequence_length:
                for i in range(self.sequence_length, len(values)):
                    X_ticker.append(values[i-self.sequence_length:i])
                
                X_ticker = np.array(X_ticker)
                
                # Scale
                X_ticker_scaled = self.scaler_X.transform(
                    X_ticker.reshape(-1, self.n_features)
                ).reshape(X_ticker.shape)
                
                # Predict
                preds = self.model.predict(X_ticker_scaled, verbose=0)
                
                # Create result DataFrame (align with original index)
                result_df = df.iloc[self.sequence_length:].copy()
                result_df['Predicted'] = preds.flatten()
                
                # Generate Signals
                # FIX: Lower threshold significantly to trigger trades
                result_df = self.generate_signals(result_df, threshold=0.001) # 0.1% threshold
                
                predictions_dict[ticker] = result_df
                
        return predictions_dict
    
    def generate_signals(self, predictions_df, threshold=0.001):
        """
        Generate trading signals with lower threshold
        """
        signals_df = predictions_df.copy()
        signals_df['Signal'] = 0
        
        # Buy when predicted return > threshold
        buy_signals = signals_df['Predicted'] > threshold
        signals_df.loc[buy_signals, 'Signal'] = 1
        
        # Sell when predicted return < -threshold
        sell_signals = signals_df['Predicted'] < -threshold
        signals_df.loc[sell_signals, 'Signal'] = -1
        
        return signals_df