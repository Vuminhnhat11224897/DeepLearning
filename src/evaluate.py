"""
Evaluation Metrics and Functions
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from typing import Dict, Tuple, List
import pandas as pd

def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Root Mean Squared Error"""
    return np.sqrt(mean_squared_error(y_true, y_pred))

def calculate_nse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Nash-Sutcliffe Efficiency (NSE)
    
    NSE = 1 - [Σ(y_true - y_pred)² / Σ(y_true - y_mean)²]
    
    Range: (-∞, 1], where:
    - NSE = 1: Perfect prediction
    - NSE = 0: Model is as good as using mean
    - NSE < 0: Model is worse than using mean
    
    Interpretation:
    - NSE > 0.75: Very good
    - NSE > 0.65: Good
    - NSE > 0.50: Satisfactory
    - NSE < 0.50: Unsatisfactory
    """
    y_mean = np.mean(y_true)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_mean) ** 2)
    
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else -np.inf
    
    return 1 - (ss_res / ss_tot)

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate all metrics: R², NSE, MAE, RMSE
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
    
    Returns:
        Dictionary with all metrics
    """
    metrics = {
        'R2': r2_score(y_true, y_pred),
        'NSE': calculate_nse(y_true, y_pred),
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': calculate_rmse(y_true, y_pred)
    }
    return metrics

def calculate_metrics_per_step(y_true: np.ndarray, 
                                y_pred: np.ndarray,
                                step_names: List[str] = None) -> pd.DataFrame:
    """
    Calculate metrics for each prediction step (t+1, t+2, ..., t+n)
    
    Args:
        y_true: Ground truth, shape (n_samples, n_steps)
        y_pred: Predictions, shape (n_samples, n_steps)
        step_names: Names for each step, e.g., ['t+1', 't+2', ...]
    
    Returns:
        DataFrame with metrics for each step
    """
    n_steps = y_true.shape[1]
    
    if step_names is None:
        step_names = [f't+{i+1}' for i in range(n_steps)]
    
    results = []
    
    for i in range(n_steps):
        metrics = calculate_metrics(y_true[:, i], y_pred[:, i])
        metrics['Step'] = step_names[i]
        results.append(metrics)
    
    # Add average
    avg_metrics = calculate_metrics(y_true.flatten(), y_pred.flatten())
    avg_metrics['Step'] = 'Average'
    results.append(avg_metrics)
    
    df = pd.DataFrame(results)
    df = df[['Step', 'R2', 'NSE', 'MAE', 'RMSE']]
    
    return df

def predict(model: nn.Module,
            dataloader: DataLoader,
            device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate predictions
    
    Args:
        model: Trained model
        dataloader: DataLoader for prediction
        device: Computation device
    
    Returns:
        y_true: Ground truth values
        y_pred: Predicted values
    """
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            
            # Forward pass (no teacher forcing)
            outputs = model(batch_x, None, 0.0)
            
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(batch_y.numpy())
    
    y_pred = np.concatenate(all_predictions, axis=0)
    y_true = np.concatenate(all_targets, axis=0)
    
    return y_true, y_pred

def evaluate_model(model: nn.Module,
                   test_loader: DataLoader,
                   scaler,
                   target_idx: int,
                   device: torch.device) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Complete model evaluation
    
    Args:
        model: Trained model
        test_loader: Test DataLoader
        scaler: Fitted scaler for inverse transform
        target_idx: Index of target column
        device: Computation device
    
    Returns:
        metrics_df: DataFrame with metrics per step
        y_true_original: Ground truth in original scale
        y_pred_original: Predictions in original scale
    """
    # Get predictions
    y_true, y_pred = predict(model, test_loader, device)
    
    # Inverse transform if scaler provided
    if scaler is not None:
        # Create dummy arrays for inverse transform
        n_samples, n_steps = y_true.shape
        n_features = scaler.n_features_in_
        
        y_true_original = np.zeros_like(y_true)
        y_pred_original = np.zeros_like(y_pred)
        
        for i in range(n_steps):
            # Create dummy array with zeros
            dummy = np.zeros((n_samples, n_features))
            
            # Put true values in target column
            dummy[:, target_idx] = y_true[:, i]
            y_true_original[:, i] = scaler.inverse_transform(dummy)[:, target_idx]
            
            # Put predicted values in target column
            dummy[:, target_idx] = y_pred[:, i]
            y_pred_original[:, i] = scaler.inverse_transform(dummy)[:, target_idx]
    else:
        y_true_original = y_true
        y_pred_original = y_pred
    
    # Calculate metrics per step
    metrics_df = calculate_metrics_per_step(y_true_original, y_pred_original)
    
    return metrics_df, y_true_original, y_pred_original

def print_evaluation_report(metrics_df: pd.DataFrame):
    """Print formatted evaluation report"""
    print("\n" + "=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)
    
    print("\nMetrics per Prediction Step:")
    print("-" * 60)
    print(metrics_df.to_string(index=False, float_format='{:.4f}'.format))
    print("-" * 60)
    
    # Interpretation
    avg_nse = metrics_df[metrics_df['Step'] == 'Average']['NSE'].values[0]
    avg_r2 = metrics_df[metrics_df['Step'] == 'Average']['R2'].values[0]
    
    print("\nInterpretation:")
    print(f"  - Average R²: {avg_r2:.4f}", end="")
    if avg_r2 > 0.9:
        print(" (Excellent)")
    elif avg_r2 > 0.7:
        print(" (Good)")
    elif avg_r2 > 0.5:
        print(" (Moderate)")
    else:
        print(" (Poor)")
    
    print(f"  - Average NSE: {avg_nse:.4f}", end="")
    if avg_nse > 0.75:
        print(" (Very Good)")
    elif avg_nse > 0.65:
        print(" (Good)")
    elif avg_nse > 0.50:
        print(" (Satisfactory)")
    else:
        print(" (Unsatisfactory)")
    
    print("=" * 60)
