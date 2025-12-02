"""
Data Preprocessing Functions
"""
import pandas as pd
import numpy as np
from typing import Tuple, Optional

def load_data(path: str) -> pd.DataFrame:
    """Load raw data from CSV"""
    df = pd.read_csv(path)
    print(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def convert_datetime(df: pd.DataFrame, date_col: str = 'date_time') -> pd.DataFrame:
    """Convert date column to datetime and sort"""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)
    print(f"Date range: {df[date_col].min()} to {df[date_col].max()}")
    return df

def check_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Check and return missing values summary"""
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({
        'Missing Count': missing,
        'Missing %': missing_pct
    })
    return missing_df[missing_df['Missing Count'] > 0]

def handle_missing_values(df: pd.DataFrame, 
                          numerical_strategy: str = 'interpolate',
                          categorical_strategy: str = 'ffill') -> pd.DataFrame:
    """Handle missing values"""
    df = df.copy()
    
    # Numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        if df[col].isnull().sum() > 0:
            if numerical_strategy == 'interpolate':
                df[col] = df[col].interpolate(method='linear')
            elif numerical_strategy == 'mean':
                df[col] = df[col].fillna(df[col].mean())
            elif numerical_strategy == 'median':
                df[col] = df[col].fillna(df[col].median())
    
    # Categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            if categorical_strategy == 'ffill':
                df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
            elif categorical_strategy == 'mode':
                df[col] = df[col].fillna(df[col].mode()[0])
    
    print(f"Missing values handled. Remaining: {df.isnull().sum().sum()}")
    return df

def handle_duplicates(df: pd.DataFrame, 
                      date_col: str = 'date_time',
                      keep: str = 'first') -> pd.DataFrame:
    """Remove duplicate timestamps"""
    n_before = len(df)
    df = df.drop_duplicates(subset=[date_col], keep=keep)
    n_removed = n_before - len(df)
    print(f"Removed {n_removed} duplicate rows")
    return df.reset_index(drop=True)

def handle_outliers_iqr(df: pd.DataFrame, 
                        column: str,
                        factor: float = 1.5,
                        method: str = 'clip') -> pd.DataFrame:
    """Handle outliers using IQR method"""
    df = df.copy()
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - factor * IQR
    upper = Q3 + factor * IQR
    
    n_outliers = ((df[column] < lower) | (df[column] > upper)).sum()
    
    if method == 'clip':
        df[column] = df[column].clip(lower=lower, upper=upper)
    elif method == 'remove':
        df = df[(df[column] >= lower) & (df[column] <= upper)]
    
    print(f"Column '{column}': {n_outliers} outliers handled (method={method})")
    return df

def handle_outliers_zscore(df: pd.DataFrame,
                           column: str,
                           threshold: float = 3.0,
                           method: str = 'clip') -> pd.DataFrame:
    """Handle outliers using Z-score method"""
    df = df.copy()
    mean = df[column].mean()
    std = df[column].std()
    z_scores = np.abs((df[column] - mean) / std)
    
    n_outliers = (z_scores > threshold).sum()
    
    if method == 'clip':
        lower = mean - threshold * std
        upper = mean + threshold * std
        df[column] = df[column].clip(lower=lower, upper=upper)
    elif method == 'remove':
        df = df[z_scores <= threshold]
    
    print(f"Column '{column}': {n_outliers} outliers handled (Z-score, method={method})")
    return df

def check_time_continuity(df: pd.DataFrame, 
                          date_col: str = 'date_time',
                          freq: str = 'H') -> Tuple[pd.DataFrame, int]:
    """Check for missing timestamps"""
    df = df.set_index(date_col)
    full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq)
    missing_timestamps = full_range.difference(df.index)
    df = df.reset_index()
    print(f"Missing timestamps: {len(missing_timestamps)}")
    return df, len(missing_timestamps)

def resample_hourly(df: pd.DataFrame,
                    date_col: str = 'date_time',
                    target_col: str = 'traffic_volume') -> pd.DataFrame:
    """Resample to ensure hourly continuity"""
    df = df.copy()
    df = df.set_index(date_col)
    
    # Create full hourly range
    full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='H')
    df = df.reindex(full_range)
    
    # Fill missing values
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    df[numerical_cols] = df[numerical_cols].interpolate(method='linear')
    
    categorical_cols = df.select_dtypes(include=['object']).columns
    df[categorical_cols] = df[categorical_cols].fillna(method='ffill').fillna(method='bfill')
    
    df = df.reset_index().rename(columns={'index': date_col})
    print(f"Resampled to hourly: {len(df)} rows")
    return df

def preprocess_pipeline(df: pd.DataFrame,
                        date_col: str = 'date_time',
                        target_col: str = 'traffic_volume') -> pd.DataFrame:
    """Complete preprocessing pipeline"""
    print("=" * 50)
    print("Starting Preprocessing Pipeline")
    print("=" * 50)
    
    # 1. Convert datetime
    df = convert_datetime(df, date_col)
    
    # 2. Handle duplicates
    df = handle_duplicates(df, date_col)
    
    # 3. Handle missing values
    df = handle_missing_values(df)
    
    # 4. Handle outliers in target
    df = handle_outliers_iqr(df, target_col, factor=1.5, method='clip')
    
    # 5. Resample to hourly
    df = resample_hourly(df, date_col, target_col)
    
    print("=" * 50)
    print(f"Preprocessing complete: {df.shape}")
    print("=" * 50)
    
    return df
