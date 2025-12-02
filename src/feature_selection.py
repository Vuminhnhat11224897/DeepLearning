"""
Feature Selection Functions
"""
import pandas as pd
import numpy as np
from typing import List, Tuple
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression, RFE
from sklearn.preprocessing import StandardScaler

def correlation_with_target(df: pd.DataFrame, 
                            target_col: str = 'traffic_volume',
                            threshold: float = 0.05) -> Tuple[pd.Series, List[str]]:
    """Calculate correlation with target and filter by threshold"""
    # Get numerical columns only
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numerical_cols:
        numerical_cols.remove(target_col)
    
    correlations = df[numerical_cols].corrwith(df[target_col]).abs().sort_values(ascending=False)
    
    # Filter by threshold
    selected = correlations[correlations >= threshold].index.tolist()
    
    print(f"Features with |correlation| >= {threshold}: {len(selected)}")
    return correlations, selected

def remove_highly_correlated(df: pd.DataFrame,
                             features: List[str],
                             threshold: float = 0.95) -> List[str]:
    """Remove features that are highly correlated with each other"""
    corr_matrix = df[features].corr().abs()
    
    # Upper triangle
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Find features to drop
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    
    selected = [f for f in features if f not in to_drop]
    
    print(f"Removed {len(to_drop)} highly correlated features (>{threshold})")
    print(f"Dropped: {to_drop}")
    return selected

def random_forest_importance(df: pd.DataFrame,
                             features: List[str],
                             target_col: str = 'traffic_volume',
                             n_estimators: int = 100,
                             top_k: int = None) -> Tuple[pd.Series, List[str]]:
    """Calculate feature importance using Random Forest"""
    X = df[features].values
    y = df[target_col].values
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train Random Forest
    rf = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)
    rf.fit(X_scaled, y)
    
    # Get importance
    importance = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
    
    if top_k:
        selected = importance.head(top_k).index.tolist()
    else:
        selected = features
    
    print(f"Random Forest feature importance calculated")
    return importance, selected

def mutual_information_scores(df: pd.DataFrame,
                              features: List[str],
                              target_col: str = 'traffic_volume',
                              top_k: int = None) -> Tuple[pd.Series, List[str]]:
    """Calculate mutual information scores"""
    X = df[features].values
    y = df[target_col].values
    
    # Calculate MI
    mi_scores = mutual_info_regression(X, y, random_state=42)
    mi_series = pd.Series(mi_scores, index=features).sort_values(ascending=False)
    
    if top_k:
        selected = mi_series.head(top_k).index.tolist()
    else:
        selected = features
    
    print(f"Mutual Information scores calculated")
    return mi_series, selected

def recursive_feature_elimination(df: pd.DataFrame,
                                  features: List[str],
                                  target_col: str = 'traffic_volume',
                                  n_features: int = 20) -> List[str]:
    """Select features using Recursive Feature Elimination"""
    X = df[features].values
    y = df[target_col].values
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # RFE with Random Forest
    estimator = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    rfe = RFE(estimator, n_features_to_select=n_features, step=1)
    rfe.fit(X_scaled, y)
    
    # Get selected features
    selected = [features[i] for i in range(len(features)) if rfe.support_[i]]
    
    print(f"RFE selected {len(selected)} features")
    return selected

def feature_selection_pipeline(df: pd.DataFrame,
                               target_col: str = 'traffic_volume',
                               date_col: str = 'date_time',
                               exclude_cols: List[str] = None,
                               correlation_threshold: float = 0.05,
                               multicollinearity_threshold: float = 0.95,
                               top_k: int = 30) -> Tuple[List[str], dict]:
    """Complete feature selection pipeline"""
    print("=" * 50)
    print("Starting Feature Selection Pipeline")
    print("=" * 50)
    
    # Get all numerical features
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Exclude certain columns
    if exclude_cols is None:
        exclude_cols = [target_col]
    else:
        exclude_cols = exclude_cols + [target_col]
    
    features = [c for c in numerical_cols if c not in exclude_cols]
    print(f"Initial features: {len(features)}")
    
    results = {}
    
    # 1. Correlation with target
    corr_scores, corr_selected = correlation_with_target(df, target_col, correlation_threshold)
    results['correlation'] = corr_scores
    features = [f for f in features if f in corr_selected]
    print(f"After correlation filter: {len(features)}")
    
    # 2. Remove multicollinearity
    features = remove_highly_correlated(df, features, multicollinearity_threshold)
    print(f"After multicollinearity removal: {len(features)}")
    
    # 3. Random Forest importance
    rf_importance, _ = random_forest_importance(df, features, target_col)
    results['rf_importance'] = rf_importance
    
    # 4. Mutual Information
    mi_scores, _ = mutual_information_scores(df, features, target_col)
    results['mi_scores'] = mi_scores
    
    # 5. Final selection based on combined ranking
    # Average rank from RF and MI
    rf_ranks = rf_importance[features].rank(ascending=False)
    mi_ranks = mi_scores[features].rank(ascending=False)
    avg_ranks = (rf_ranks + mi_ranks) / 2
    
    final_features = avg_ranks.sort_values().head(top_k).index.tolist()
    results['final_ranking'] = avg_ranks.sort_values()
    
    print("=" * 50)
    print(f"Selected {len(final_features)} features")
    print("=" * 50)
    
    return final_features, results
