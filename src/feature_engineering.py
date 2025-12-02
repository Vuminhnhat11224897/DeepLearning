"""
Feature Engineering Functions
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

def add_lag_features(df: pd.DataFrame, 
                     target_col: str = 'traffic_volume',
                     lags: List[int] = [1, 2, 3, 6, 12, 24, 168]) -> pd.DataFrame:
    """Add lag features for target column"""
    df = df.copy()
    
    for lag in lags:
        df[f'{target_col}_lag_{lag}h'] = df[target_col].shift(lag)
    
    print(f"Added lag features: {lags}")
    return df

def add_rolling_features(df: pd.DataFrame,
                         target_col: str = 'traffic_volume',
                         windows: List[int] = [3, 6, 12, 24]) -> pd.DataFrame:
    """Add rolling statistics features"""
    df = df.copy()
    
    for window in windows:
        df[f'rolling_mean_{window}h'] = df[target_col].shift(1).rolling(window=window).mean()
        df[f'rolling_std_{window}h'] = df[target_col].shift(1).rolling(window=window).std()
    
    # Rolling min/max for 24h
    df['rolling_min_24h'] = df[target_col].shift(1).rolling(window=24).min()
    df['rolling_max_24h'] = df[target_col].shift(1).rolling(window=24).max()
    
    # Exponential weighted mean
    df['ewm_mean'] = df[target_col].shift(1).ewm(span=12).mean()
    
    print(f"Added rolling features for windows: {windows}")
    return df

def add_diff_features(df: pd.DataFrame,
                      target_col: str = 'traffic_volume') -> pd.DataFrame:
    """Add difference features"""
    df = df.copy()
    
    # Absolute difference
    df['diff_1h'] = df[target_col].diff(1)
    df['diff_24h'] = df[target_col].diff(24)
    
    # Percentage change
    df['pct_change_1h'] = df[target_col].pct_change(1)
    df['pct_change_24h'] = df[target_col].pct_change(24)
    
    print(f"Added difference features: diff_1h, diff_24h, pct_change_1h, pct_change_24h")
    return df

def add_holiday_features(df: pd.DataFrame, 
                         holiday_col: str = 'holiday') -> pd.DataFrame:
    """Add holiday-related features"""
    df = df.copy()
    
    # Binary is_holiday
    df['is_holiday'] = (df[holiday_col] != 'None').astype(int)
    
    # Holiday type encoding (simplified)
    holiday_types = df[holiday_col].unique()
    holiday_map = {h: i for i, h in enumerate(holiday_types)}
    df['holiday_encoded'] = df[holiday_col].map(holiday_map)
    
    print(f"Added holiday features: is_holiday, holiday_encoded")
    return df

def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add weather-related engineered features"""
    df = df.copy()
    
    # Temperature: Convert Kelvin to Celsius
    if 'temp' in df.columns:
        df['temp_celsius'] = df['temp'] - 273.15
        
        # Temperature categories
        def temp_category(temp):
            if temp < 0:
                return 0  # Freezing
            elif temp < 10:
                return 1  # Cold
            elif temp < 20:
                return 2  # Mild
            elif temp < 30:
                return 3  # Warm
            else:
                return 4  # Hot
        
        df['temp_category'] = df['temp_celsius'].apply(temp_category)
    
    # Rain/Snow indicators
    if 'rain_1h' in df.columns:
        df['is_rainy'] = (df['rain_1h'] > 0).astype(int)
    
    if 'snow_1h' in df.columns:
        df['is_snowy'] = (df['snow_1h'] > 0).astype(int)
    
    # Cloud categories
    if 'clouds_all' in df.columns:
        def cloud_category(clouds):
            if clouds < 20:
                return 0  # Clear
            elif clouds < 50:
                return 1  # Partly cloudy
            elif clouds < 80:
                return 2  # Cloudy
            else:
                return 3  # Overcast
        
        df['cloud_category'] = df['clouds_all'].apply(cloud_category)
    
    # Weather main encoding
    if 'weather_main' in df.columns:
        weather_types = df['weather_main'].unique()
        weather_map = {w: i for i, w in enumerate(weather_types)}
        df['weather_encoded'] = df['weather_main'].map(weather_map)
    
    print(f"Added weather features: temp_celsius, temp_category, is_rainy, is_snowy, cloud_category, weather_encoded")
    return df

def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add interaction features between variables"""
    df = df.copy()
    
    # Hour x Weekend interaction
    if 'hour' in df.columns and 'is_weekend' in df.columns:
        df['hour_weekend'] = df['hour'] * df['is_weekend']
    
    # Hour x Holiday interaction
    if 'hour' in df.columns and 'is_holiday' in df.columns:
        df['hour_holiday'] = df['hour'] * df['is_holiday']
    
    # Rush hour x Rain interaction
    if 'is_rush_hour' in df.columns and 'is_rainy' in df.columns:
        df['rush_rain'] = df['is_rush_hour'] * df['is_rainy']
    
    # Temperature x Rush hour
    if 'temp_celsius' in df.columns and 'is_rush_hour' in df.columns:
        df['temp_rush'] = df['temp_celsius'] * df['is_rush_hour']
    
    print(f"Added interaction features: hour_weekend, hour_holiday, rush_rain, temp_rush")
    return df

def feature_engineering_pipeline(df: pd.DataFrame,
                                 date_col: str = 'date_time',
                                 target_col: str = 'traffic_volume',
                                 holiday_col: str = 'holiday') -> pd.DataFrame:
    """Complete feature engineering pipeline"""
    print("=" * 50)
    print("Starting Feature Engineering Pipeline")
    print("=" * 50)
    
    # 1. Temporal features
    df = add_temporal_features(df, date_col)
    
    # 2. Cyclical features
    df = add_cyclical_features(df)
    
    # 3. Lag features
    df = add_lag_features(df, target_col, lags=[1, 2, 3, 6, 12, 24, 168])
    
    # 4. Rolling features
    df = add_rolling_features(df, target_col, windows=[3, 6, 12, 24])
    
    # 5. Difference features
    df = add_diff_features(df, target_col)
    
    # 6. Holiday features
    df = add_holiday_features(df, holiday_col)
    
    # 7. Weather features
    df = add_weather_features(df)
    
    # 8. Interaction features
    df = add_interaction_features(df)
    
    # Drop rows with NaN from lag/rolling features
    initial_len = len(df)
    df = df.dropna().reset_index(drop=True)
    print(f"Dropped {initial_len - len(df)} rows with NaN values")
    
    print("=" * 50)
    print(f"Feature engineering complete: {df.shape[1]} features")
    print("=" * 50)
    
    return df
