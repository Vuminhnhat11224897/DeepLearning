"""
PyTorch Dataset for Seq2Seq Time Series
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple

class TimeSeriesDataset(Dataset):
    """
    Dataset for Encoder-Decoder sequence-to-sequence model
    
    Args:
        X: Input sequences, shape (n_samples, input_seq_len, n_features)
        y: Target sequences, shape (n_samples, output_seq_len)
    """
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def create_sequences(data: np.ndarray,
                     target_idx: int,
                     input_seq_len: int = 24,
                     output_seq_len: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create input-output sequences using sliding window
    
    Args:
        data: Array of shape (n_timesteps, n_features)
        target_idx: Index of target column in data
        input_seq_len: Number of historical timesteps for input
        output_seq_len: Number of future timesteps to predict
    
    Returns:
        X: Input sequences, shape (n_samples, input_seq_len, n_features)
        y: Target sequences, shape (n_samples, output_seq_len)
    """
    X, y = [], []
    
    for i in range(len(data) - input_seq_len - output_seq_len + 1):
        # Input: all features for input_seq_len timesteps
        X.append(data[i:i + input_seq_len])
        
        # Output: target column for next output_seq_len timesteps
        y.append(data[i + input_seq_len:i + input_seq_len + output_seq_len, target_idx])
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"Created sequences: X shape = {X.shape}, y shape = {y.shape}")
    return X, y

def train_val_test_split(X: np.ndarray, 
                         y: np.ndarray,
                         train_ratio: float = 0.7,
                         val_ratio: float = 0.15) -> Tuple:
    """
    Split data into train, validation, test sets (time-based, no shuffle)
    
    Args:
        X: Input sequences
        y: Target sequences
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
    
    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test
    """
    n_samples = len(X)
    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))
    
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    
    print(f"Train: {len(X_train)} samples")
    print(f"Val:   {len(X_val)} samples")
    print(f"Test:  {len(X_test)} samples")
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def create_dataloaders(X_train: np.ndarray, y_train: np.ndarray,
                       X_val: np.ndarray, y_val: np.ndarray,
                       X_test: np.ndarray, y_test: np.ndarray,
                       batch_size: int = 64,
                       num_workers: int = 0) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        X_test, y_test: Test data
        batch_size: Batch size
        num_workers: Number of workers for data loading
    
    Returns:
        train_loader, val_loader, test_loader
    """
    train_dataset = TimeSeriesDataset(X_train, y_train)
    val_dataset = TimeSeriesDataset(X_val, y_val)
    test_dataset = TimeSeriesDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    
    print(f"Created DataLoaders with batch_size={batch_size}")
    return train_loader, val_loader, test_loader
