"""
FINAL FIX: Predict returns instead of raw prices
This is the standard approach in financial modeling
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader
try:
    from PyEMD import CEEMDAN
except ImportError:
    from emd import CEEMDAN
import warnings
warnings.filterwarnings('ignore')

from utils import calculate_market_regime, calculate_technical_indicators


class CEEMDDecomposer:
    """CEEMD with proper train/test separation"""
    
    def __init__(self, trials=100, noise_std=0.2, max_imf=10):
        self.trials = trials
        self.noise_std = noise_std
        self.max_imf = max_imf
        
    def decompose(self, signal):
        """Decompose signal using CEEMDAN"""
        ceemdan = CEEMDAN(trials=self.trials, noise_width=self.noise_std, max_imf=self.max_imf)
        try:
            imfs = ceemdan(signal)
            return imfs
        except Exception as e:
            print(f"CEEMDAN decomposition failed: {e}")
            return np.array([signal])
    
    def reconstruct(self, imfs, exclude_first_n=1):
        """Reconstruct signal by excluding high-frequency IMFs"""
        if len(imfs) <= exclude_first_n:
            return imfs[-1]
        reconstructed = np.sum(imfs[exclude_first_n:], axis=0)
        return reconstructed


class FinancialDataPreprocessor:
    """FIXED: Predicts returns instead of prices"""
    
    def __init__(self, config, dev_mode=False):
        self.config = config
        self.dev_mode = dev_mode
        self.ceemdan = CEEMDDecomposer(
            trials=config.ceemd_trials,
            noise_std=config.ceemd_noise_std,
            max_imf=config.ceemd_max_imf
        )
        self.pca = None
        self.scaler = StandardScaler()
        
    def load_data(self, file_path):
        """Load financial data from CSV"""
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        
        # Calculate returns (key fix!)
        df['Returns'] = df['Close'].pct_change()
        df = df.dropna()
        
        if self.dev_mode:
            print(f"[Data Loader] Loaded data shape: {df.shape}")
            print(f"[Data Loader] Date range: {df['Date'].min()} to {df['Date'].max()}")
            print(f"[Data Loader] Returns mean: {df['Returns'].mean():.6f}, std: {df['Returns'].std():.6f}")
        
        return df
    
    def preprocess_split(self, file_path, train_ratio=0.7, val_ratio=0.15):
        """
        FIXED: Preprocess with returns prediction
        """
        # Load raw data
        df = self.load_data(file_path)
        
        # Add technical indicators
        df = calculate_technical_indicators(df)
        df = calculate_market_regime(df,
                                      window=self.config.regime_window,
                                      r2_threshold=self.config.trend_r2_threshold,
                                      vol_multiplier=self.config.high_volatility_multiplier)
        
        # Split data BEFORE any fitting
        n_total = len(df)
        train_size = int(n_total * train_ratio)
        val_size = int(n_total * val_ratio)
        
        df_train = df.iloc[:train_size].copy()
        df_val = df.iloc[train_size:train_size + val_size].copy()
        df_test = df.iloc[train_size + val_size:].copy()
        
        if self.dev_mode:
            print(f"\n[Split] Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")
        
        # Apply CEEMD on returns (not prices!)
        if self.dev_mode:
            print(f"[CEEMD] Decomposing TRAIN returns...")
        
        train_signal = df_train['Returns'].values
        train_imfs = self.ceemdan.decompose(train_signal)
        train_reconstructed = self.ceemdan.reconstruct(train_imfs, exclude_first_n=1)
        df_train['Returns_Denoised'] = train_reconstructed
        
        # For val/test: use simple moving average
        window = 5
        df_val['Returns_Denoised'] = df_val['Returns'].rolling(window=window, min_periods=1).mean()
        df_test['Returns_Denoised'] = df_test['Returns'].rolling(window=window, min_periods=1).mean()
        
        # Create windowed features from denoised returns
        seq_len = self.config.sequence_length
        
        train_windowed = self._create_windows(df_train['Returns_Denoised'].values, seq_len)
        val_windowed = self._create_windows(df_val['Returns_Denoised'].values, seq_len)
        test_windowed = self._create_windows(df_test['Returns_Denoised'].values, seq_len)
        
        # Fit PCA ONLY on training data
        if self.dev_mode:
            print(f"[PCA] Fitting on TRAIN data only...")
        
        self.pca = PCA(n_components=self.config.pca_components)
        train_pca = self.pca.fit_transform(train_windowed)
        val_pca = self.pca.transform(val_windowed)
        test_pca = self.pca.transform(test_windowed)
        
        if self.dev_mode:
            print(f"[PCA] Train: {train_pca.shape}, Val: {val_pca.shape}, Test: {test_pca.shape}")
            print(f"[PCA] Explained variance: {self.pca.explained_variance_ratio_.sum():.4f}")
        
        # Adjust dataframes
        df_train = df_train.iloc[seq_len:].reset_index(drop=True)
        df_val = df_val.iloc[seq_len:].reset_index(drop=True)
        df_test = df_test.iloc[seq_len:].reset_index(drop=True)
        
        return (train_pca, df_train), (val_pca, df_val), (test_pca, df_test)
    
    def _create_windows(self, data, window):
        """Create windowed features"""
        windowed = []
        for i in range(window, len(data)):
            windowed.append(data[i-window:i])
        return np.array(windowed)


class FinancialTimeSeriesDataset(Dataset):
    """PyTorch Dataset - predicts returns"""
    
    def __init__(self, pca_features, df, sequence_length, dev_mode=False):
        self.pca_features = pca_features
        self.df = df
        self.sequence_length = sequence_length
        self.dev_mode = dev_mode
        
        # Create sequences
        self.X, self.y, self.risk, self.regime = self._create_sequences()
        
        if dev_mode:
            print(f"[Dataset] Created {len(self.X)} sequences")
            print(f"[Dataset] X shape: {self.X.shape}")
            print(f"[Dataset] y shape: {self.y.shape}")
            print(f"[Dataset] y (returns) range: [{self.y.min():.6f}, {self.y.max():.6f}]")
    
    def _create_sequences(self):
        """Create sequences - target is RETURNS not prices"""
        X = []
        y = []
        risk = []
        regime = []
        
        for i in range(self.sequence_length, len(self.pca_features)):
            # Input: sequence of PCA features
            X.append(self.pca_features[i-self.sequence_length:i])
            
            # Target: next return (KEY FIX!)
            y.append(self.df.iloc[i]['Returns'])
            
            # Risk: current volatility
            risk.append(self.df.iloc[i]['Volatility'])
            
            # Regime label
            regime.append(self.df.iloc[i]['Regime'])
        
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32).reshape(-1, 1)
        risk = np.array(risk, dtype=np.float32).reshape(-1, 1)
        regime = np.array(regime, dtype=np.int64)
        
        return X, y, risk, regime
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return {
            'x': torch.FloatTensor(self.X[idx]),
            'y': torch.FloatTensor(self.y[idx]),
            'risk': torch.FloatTensor(self.risk[idx]),
            'regime': torch.LongTensor([self.regime[idx]])[0]
        }


def create_data_loaders(config, dev_mode=False):
    """
    FIXED: Create data loaders predicting returns
    """
    preprocessor = FinancialDataPreprocessor(config, dev_mode=dev_mode)
    
    # Preprocess with proper split
    (train_pca, df_train), (val_pca, df_val), (test_pca, df_test) = \
        preprocessor.preprocess_split(config.data_path, config.train_ratio, config.val_ratio)
    
    # Calculate regime distribution
    regime_counts = np.bincount(df_train['Regime'].values)
    regime_weights = len(df_train) / (len(regime_counts) * regime_counts)
    
    if dev_mode:
        print(f"\n[Regime Distribution in TRAIN set]:")
        regime_names = ['Trend', 'Range', 'Panic']
        for i, (count, weight) in enumerate(zip(regime_counts, regime_weights)):
            pct = count / len(df_train) * 100
            print(f"  {regime_names[i]:6s}: {count:4d} ({pct:5.2f}%) - Weight: {weight:.3f}")
    
    # Create datasets
    train_dataset = FinancialTimeSeriesDataset(train_pca, df_train, config.sequence_length, dev_mode)
    val_dataset = FinancialTimeSeriesDataset(val_pca, df_val, config.sequence_length, dev_mode)
    test_dataset = FinancialTimeSeriesDataset(test_pca, df_test, config.sequence_length, dev_mode)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
    
    if dev_mode:
        print(f"\n[Data Loaders] Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader, preprocessor, regime_weights
