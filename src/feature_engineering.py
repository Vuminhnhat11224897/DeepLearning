"""
Feature Engineering Functions for LSTM Time Series Forecasting

Note: Lag, Rolling, and Diff features are NOT included by design.
LSTM models learn temporal patterns directly from sequence data.
Adding these features manually can cause data leakage.
"""
import pandas as pd
import numpy as np
from typing import List, Optional


def add_temporal_features(df: pd.DataFrame, date_col: str = 'date_time') -> pd.DataFrame:
    """Add temporal features from datetime column"""
    df = df.copy()
    
    # Basic temporal features
    df['hour'] = df[date_col].dt.hour
    df['day_of_week'] = df[date_col].dt.dayofweek  # Monday=0, Sunday=6
    df['day_of_month'] = df[date_col].dt.day
    df['month'] = df[date_col].dt.month
    df['year'] = df[date_col].dt.year
    df['week_of_year'] = df[date_col].dt.isocalendar().week.astype(int)
    df['quarter'] = df[date_col].dt.quarter
    
    # Binary features
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_rush_hour'] = df['hour'].apply(lambda x: 1 if (7 <= x <= 9) or (16 <= x <= 18) else 0)
    
    # Season (Northern Hemisphere)
    def get_season(month):
        if month in [3, 4, 5]:
            return 0  # Spring
        elif month in [6, 7, 8]:
            return 1  # Summer
        elif month in [9, 10, 11]:
            return 2  # Fall
        else:
            return 3  # Winter
    
    df['season'] = df['month'].apply(get_season)
    
    print(f"Added temporal features: hour, day_of_week, day_of_month, month, year, week_of_year, quarter, is_weekend, is_rush_hour, season")
    return df


def add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical encoding for temporal features"""
    df = df.copy()
    
    # Hour (24-hour cycle)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # Day of week (7-day cycle)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    # Month (12-month cycle)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    print(f"Added cyclical features: hour_sin/cos, day_sin/cos, month_sin/cos")
    return df


def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add weather-derived features"""
    df = df.copy()

    # Kelvin → Celsius
    df['temp_celsius'] = df['temp'] - 273.15

    # Binary indicators
    df['is_rainy'] = (df['rain_1h'] > 0).astype(int)
    df['is_snowy'] = (df['snow_1h'] > 0).astype(int)

    print("Added weather features: temp_celsius, is_rainy, is_snowy")
    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add interaction features between existing features"""
    df = df.copy()

    if 'is_rush_hour' in df.columns and 'is_rainy' in df.columns:
        df['rush_rain'] = df['is_rush_hour'] * df['is_rainy']

    print("Added interaction feature: rush_rain")
    return df


def feature_engineering_pipeline(df: pd.DataFrame,
                                 date_col: str = 'date_time',
                                 target_col: str = 'traffic_volume') -> pd.DataFrame:
    """
    Complete feature engineering pipeline for LSTM time series forecasting.
    
    Note: Lag, Rolling, and Diff features are NOT included.
    LSTM models learn temporal patterns directly from sequence data.
    Adding these features manually can cause data leakage.
    
    Args:
        df: Input DataFrame
        date_col: Name of datetime column
        target_col: Name of target column
        
    Returns:
        DataFrame with engineered features
    """
    print("=" * 50)
    print("Feature Engineering Pipeline for LSTM")
    print("=" * 50)
    
    # 1. Temporal features
    df = add_temporal_features(df, date_col)
    
    # 2. Cyclical features
    df = add_cyclical_features(df)
    
    # 3. Weather features
    df = add_weather_features(df)
    
    # 4. Interaction features
    df = add_interaction_features(df)
    
    print("=" * 50)
    print(f"Feature engineering complete: {df.shape[1]} features")
    print("Note: No lag/rolling/diff features - LSTM learns from sequences")
    print("=" * 50)
    
    return df
