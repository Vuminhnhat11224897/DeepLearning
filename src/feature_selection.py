"""
Feature Selection Functions for LSTM Encoder-Decoder

Optimized for time series forecasting with LSTM:
- No lag/rolling/diff features (LSTM learns from sequences)
- Focus on temporal, weather, and context features
- Avoid data leakage
"""
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict


# ============================================================
# LSTM-SPECIFIC FEATURE CATEGORIES
# ============================================================

# Features that LSTM can use safely (no data leakage)
LSTM_SAFE_FEATURES = {
    # Cyclical temporal features (encode time patterns)
    'cyclical': ['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos'],
    
    # Categorical temporal features
    'temporal': ['hour', 'day_of_week', 'day_of_month', 'month', 'year', 
                 'week_of_year', 'quarter', 'season'],
    
    # Binary context features
    'context': ['is_weekend', 'is_rush_hour'],
    
    # Weather features (known at prediction time or forecasted)
    'weather': ['temp', 'temp_celsius', 'clouds_all', 'rain_1h', 'snow_1h', 
                'is_rainy', 'is_snowy'],
    
    # Interaction features
    'interaction': ['rush_rain'],
}

# Features to NEVER use with LSTM (cause data leakage)
LEAKAGE_FEATURES = [
    # Lag features - LSTM handles this via sequence input
    'lag', '_lag_',
    # Rolling statistics - uses future data in training
    'rolling_', 'ewm_',
    # Difference features - can leak information
    'diff_', 'pct_change',
]


def get_lstm_feature_categories() -> Dict[str, List[str]]:
    """Return feature categories safe for LSTM"""
    return LSTM_SAFE_FEATURES.copy()


def check_for_leakage_features(features: List[str]) -> List[str]:
    """Check if any features might cause data leakage"""
    leakage = []
    for feat in features:
        for pattern in LEAKAGE_FEATURES:
            if pattern in feat.lower():
                leakage.append(feat)
                break
    return leakage


def get_available_features(df: pd.DataFrame, 
                           target_col: str = 'traffic_volume',
                           date_col: str = 'date_time') -> Dict[str, List[str]]:
    """
    Get available features from dataframe, categorized by type.
    Only returns features that exist in the dataframe.
    """
    available = {}
    
    for category, feature_list in LSTM_SAFE_FEATURES.items():
        available[category] = [f for f in feature_list if f in df.columns]
    
    # Check for any leakage features that shouldn't be there
    all_cols = df.columns.tolist()
    exclude = [target_col, date_col]
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numerical_cols = [c for c in numerical_cols if c not in exclude]
    
    leakage = check_for_leakage_features(numerical_cols)
    if leakage:
        print(f"⚠️ Warning: Found {len(leakage)} potential leakage features: {leakage}")
        print("  These should NOT be used with LSTM!")
    
    return available


def correlation_with_target(df: pd.DataFrame, 
                            target_col: str = 'traffic_volume',
                            threshold: float = 0.05) -> Tuple[pd.Series, List[str]]:
    """Calculate correlation with target and filter by threshold"""
    # Get numerical columns only
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numerical_cols:
        numerical_cols.remove(target_col)
    
    # Filter out leakage features
    safe_cols = [c for c in numerical_cols if not check_for_leakage_features([c])]
    
    correlations = df[safe_cols].corrwith(df[target_col]).abs().sort_values(ascending=False)
    
    # Filter by threshold
    selected = correlations[correlations >= threshold].index.tolist()
    
    print(f"Features with |correlation| >= {threshold}: {len(selected)}")
    return correlations, selected


def remove_highly_correlated(df: pd.DataFrame,
                             features: List[str],
                             threshold: float = 0.95) -> List[str]:
    """Remove features that are highly correlated with each other"""
    if len(features) <= 1:
        return features
        
    corr_matrix = df[features].corr().abs()
    
    # Upper triangle
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Find features to drop
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    
    selected = [f for f in features if f not in to_drop]
    
    if to_drop:
        print(f"Removed {len(to_drop)} highly correlated features (>{threshold})")
        print(f"Dropped: {to_drop}")
    else:
        print("No highly correlated features to remove")
    
    return selected


def select_features_for_lstm(df: pd.DataFrame,
                             target_col: str = 'traffic_volume',
                             date_col: str = 'date_time',
                             include_categories: List[str] = None,
                             multicollinearity_threshold: float = 0.95) -> Tuple[List[str], Dict]:
    """
    Select features optimized for LSTM Encoder-Decoder.
    
    Args:
        df: Input DataFrame
        target_col: Target column name
        date_col: Date column name  
        include_categories: Which feature categories to include
                           Options: 'cyclical', 'temporal', 'context', 'weather', 'interaction'
                           Default: all categories
        multicollinearity_threshold: Threshold for removing correlated features
        
    Returns:
        Tuple of (selected_features, info_dict)
    """
    print("=" * 60)
    print("LSTM Feature Selection")
    print("=" * 60)
    
    # Default: include all categories
    if include_categories is None:
        include_categories = list(LSTM_SAFE_FEATURES.keys())
    
    # Get available features by category
    available = get_available_features(df, target_col, date_col)
    
    # Collect features from selected categories
    selected_features = []
    category_info = {}
    
    for category in include_categories:
        if category in available:
            feats = available[category]
            selected_features.extend(feats)
            category_info[category] = feats
            print(f"  {category}: {len(feats)} features - {feats}")
    
    print(f"\nTotal features before filtering: {len(selected_features)}")
    
    # Remove multicollinearity
    if len(selected_features) > 1:
        selected_features = remove_highly_correlated(
            df, selected_features, multicollinearity_threshold
        )
    
    # Calculate correlation with target for info
    correlations = df[selected_features].corrwith(df[target_col]).abs().sort_values(ascending=False)
    
    info = {
        'categories': category_info,
        'correlations': correlations,
        'n_features': len(selected_features),
    }
    
    print("=" * 60)
    print(f"Selected {len(selected_features)} features for LSTM")
    print("=" * 60)
    
    return selected_features, info


def feature_selection_pipeline(df: pd.DataFrame,
                               target_col: str = 'traffic_volume',
                               date_col: str = 'date_time',
                               multicollinearity_threshold: float = 0.95) -> Tuple[List[str], Dict]:
    """
    Complete feature selection pipeline for LSTM.
    
    This is a simplified pipeline that:
    1. Uses only LSTM-safe features (no leakage)
    2. Removes highly correlated features
    3. Returns features grouped by category
    """
    return select_features_for_lstm(
        df=df,
        target_col=target_col,
        date_col=date_col,
        include_categories=None,  # Include all safe categories
        multicollinearity_threshold=multicollinearity_threshold
    )
