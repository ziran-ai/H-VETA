"""
Utility functions for H-VEDA (Rg-MoE) model
"""

import numpy as np
import pandas as pd
from scipy.stats import linregress
import torch
import random
import os


def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_market_regime(df, window=20, r2_threshold=0.8, vol_multiplier=1.5):
    """
    Calculate market regime labels for each timestamp
    
    Args:
        df: DataFrame with 'Close' and 'Log_Ret' columns
        window: lookback window for regime calculation
        r2_threshold: R-squared threshold for trend detection
        vol_multiplier: multiplier for high volatility detection
    
    Returns:
        DataFrame with added 'Regime' column (0: Trend, 1: Range, 2: Panic/HighVol)
    """
    regimes = []
    global_vol_std = df['Log_Ret'].std()
    
    for i in range(len(df)):
        if i < window:
            regimes.append(1)  # Default to range regime
            continue
            
        slice_data = df['Close'].iloc[i-window:i].values
        x = np.arange(window)
        
        # Calculate linear regression (trend strength)
        slope, intercept, r_value, p_value, std_err = linregress(x, slice_data)
        r_squared = r_value ** 2
        
        # Calculate volatility (log return std)
        volatility = df['Log_Ret'].iloc[i-window:i].std()
        
        # Regime classification logic
        if r_squared > r2_threshold:
            regimes.append(0)  # Strong trend
        elif volatility > global_vol_std * vol_multiplier:
            regimes.append(2)  # Panic/High volatility
        else:
            regimes.append(1)  # Trading range
            
    df['Regime'] = regimes
    return df


def calculate_technical_indicators(df):
    """
    Calculate technical indicators for the dataframe
    
    Args:
        df: DataFrame with OHLCV data
    
    Returns:
        DataFrame with added technical indicators
    """
    # Log returns
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Log_Ret'].fillna(0, inplace=True)
    
    # Rolling volatility (GARCH-like approximation)
    df['Volatility'] = df['Log_Ret'].rolling(window=20).std()
    df['Volatility'].fillna(df['Volatility'].mean(), inplace=True)
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['RSI'].fillna(50, inplace=True)
    
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
    
    # Fill NaN values
    df['BB_Middle'].fillna(method='bfill', inplace=True)
    df['BB_Upper'].fillna(method='bfill', inplace=True)
    df['BB_Lower'].fillna(method='bfill', inplace=True)
    df['BB_Width'].fillna(df['BB_Width'].mean(), inplace=True)
    
    # ADX (Average Directional Index)
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    plus_dm = high.diff()
    minus_dm = low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    
    tr1 = pd.DataFrame(high - low)
    tr2 = pd.DataFrame(abs(high - close.shift(1)))
    tr3 = pd.DataFrame(abs(low - close.shift(1)))
    frames = [tr1, tr2, tr3]
    tr = pd.concat(frames, axis=1, join='inner').max(axis=1)
    atr = tr.rolling(14).mean()
    
    plus_di = 100 * (plus_dm.ewm(alpha=1/14).mean() / atr)
    minus_di = abs(100 * (minus_dm.ewm(alpha=1/14).mean() / atr))
    dx = (abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100
    df['ADX'] = dx.rolling(14).mean()
    df['ADX'].fillna(25, inplace=True)
    
    return df


def calculate_entropy(probabilities):
    """
    Calculate Shannon entropy of probability distribution
    
    Args:
        probabilities: torch.Tensor of shape (batch_size, num_classes)
    
    Returns:
        torch.Tensor of shape (batch_size,) with entropy values
    """
    eps = 1e-10
    entropy = -torch.sum(probabilities * torch.log(probabilities + eps), dim=1)
    return entropy


def create_sequences(data, seq_length, target_col='Close'):
    """
    Create sequences for time series prediction
    
    Args:
        data: numpy array of features
        seq_length: length of input sequence
        target_col: name of target column
    
    Returns:
        X: input sequences
        y: target values
    """
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length, -1])  # Assuming last column is target
    return np.array(X), np.array(y)


def save_checkpoint(model, optimizer, epoch, loss, path):
    """Save model checkpoint"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)


def load_checkpoint(model, optimizer, path):
    """Load model checkpoint"""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model, optimizer, epoch, loss


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve"""
    
    def __init__(self, patience=10, min_delta=0, mode='min'):
        """
        Args:
            patience: number of epochs to wait before stopping
            min_delta: minimum change to qualify as improvement
            mode: 'min' or 'max' for loss or metric
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif self.mode == 'min':
            if score > self.best_score - self.min_delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.counter = 0
        else:  # mode == 'max'
            if score < self.best_score + self.min_delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.counter = 0
        
        return self.early_stop
