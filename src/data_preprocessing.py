"""
Data Preprocessing Functions
"""
import pandas as pd
import numpy as np
from typing import Tuple, Optional, List

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
    # Do NOT impute target labels; only interpolate feature columns
    feature_numeric_cols = [c for c in numerical_cols if c != target_col]
    if feature_numeric_cols:
        df[feature_numeric_cols] = df[feature_numeric_cols].interpolate(method='linear')
    
    categorical_cols = df.select_dtypes(include=['object']).columns
    df[categorical_cols] = df[categorical_cols].fillna(method='ffill').fillna(method='bfill')
    
    df = df.reset_index().rename(columns={'index': date_col})
    # Report missing target labels after resampling so they can be handled upstream
    missing_target = df[target_col].isna().sum() if target_col in df.columns else 0
    print(f"Resampled to hourly: {len(df)} rows")
    if missing_target:
        print(f"Warning: {missing_target} rows have missing target '{target_col}'. Drop these before training.")
    return df


def split_continuous_segments(df: pd.DataFrame,
                               date_col: str = 'date_time',
                               target_col: str = 'traffic_volume',
                               min_length: int = 48,
                               freq_hours: int = 1) -> List[pd.DataFrame]:
    """
    Tách DataFrame thành các segment THỰC SỰ liên tục:
    1. Không có NaN trong target
    2. Không có gap trong timestamps (mỗi row cách nhau đúng freq_hours)
    
    Dùng cho LSTM Seq2Seq để đảm bảo mỗi sequence có:
    - Nhãn thật (không interpolate)
    - Timestamps liên tục (không nhảy qua gaps)
    
    Args:
        df: DataFrame với cột datetime và target
        date_col: Tên cột datetime
        target_col: Tên cột target
        min_length: Độ dài tối thiểu của segment (loại bỏ segment ngắn)
        freq_hours: Khoảng cách giờ mong đợi giữa các rows (mặc định 1 giờ)
    
    Returns:
        List các DataFrame, mỗi cái là một segment liên tục
    """
    df = df.copy()
    df = df.sort_values(date_col).reset_index(drop=True)
    
    # Đảm bảo date_col là datetime
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col])
    
    # Tính time difference giữa các rows liên tiếp
    time_diff = df[date_col].diff()
    expected_diff = pd.Timedelta(hours=freq_hours)
    
    # Tạo mask cho:
    # 1. Target hợp lệ (không NaN)
    # 2. Không phải điểm có gap (time_diff > expected_diff)
    valid_target = df[target_col].notna()
    
    # Điểm bắt đầu segment mới: row đầu tiên HOẶC có gap trước đó
    # Gap = time_diff > expected_diff (cho phép tolerance nhỏ)
    tolerance = pd.Timedelta(minutes=5)  # Tolerance 5 phút
    is_gap = (time_diff > expected_diff + tolerance) | time_diff.isna()
    
    # Tạo segment ID: tăng mỗi khi gặp gap hoặc invalid target
    # Mỗi lần có gap hoặc NaN target → segment mới
    invalid_point = is_gap | ~valid_target
    segment_id = invalid_point.cumsum()
    
    # Tuy nhiên, row có invalid_point=True không thuộc segment nào
    # Chỉ giữ các rows có valid_target
    df['_segment_id'] = segment_id
    df['_valid'] = valid_target
    
    segments = []
    skipped_short = 0
    skipped_invalid = 0
    
    for seg_id in df['_segment_id'].unique():
        segment_df = df[df['_segment_id'] == seg_id].copy()
        
        # Bỏ qua segment có target NaN
        if not segment_df['_valid'].all():
            skipped_invalid += 1
            continue
        
        # Bỏ qua segment quá ngắn
        if len(segment_df) < min_length:
            skipped_short += 1
            continue
        
        # Verify lại segment thực sự liên tục
        seg_time_diff = segment_df[date_col].diff().dropna()
        if len(seg_time_diff) > 0:
            max_gap = seg_time_diff.max()
            if max_gap > expected_diff + tolerance:
                # Segment vẫn có gap bên trong → không nên xảy ra nhưng check để an toàn
                print(f"  Warning: Segment {seg_id} has internal gap of {max_gap}")
                continue
        
        # Clean up và thêm vào list
        segment_df = segment_df.drop(columns=['_segment_id', '_valid'])
        segment_df = segment_df.reset_index(drop=True)
        segments.append(segment_df)
    
    total_rows = sum(len(s) for s in segments)
    total_original = len(df)
    
    print(f"=" * 50)
    print(f"SEGMENT SPLITTING RESULTS")
    print(f"=" * 50)
    print(f"Original rows: {total_original:,}")
    print(f"Gaps detected: {is_gap.sum() - 1:,}")  # -1 vì row đầu luôn có NaN diff
    print(f"Segments created: {len(segments)}")
    print(f"Segments skipped (too short < {min_length}): {skipped_short}")
    print(f"Usable rows: {total_rows:,} / {total_original:,} ({100*total_rows/total_original:.1f}%)")
    print(f"=" * 50)
    
    # In thông tin top 10 segments lớn nhất
    if segments:
        seg_lengths = [(i+1, len(s)) for i, s in enumerate(segments)]
        seg_lengths.sort(key=lambda x: x[1], reverse=True)
        print(f"\nTop {min(10, len(segments))} largest segments:")
        for seg_num, seg_len in seg_lengths[:10]:
            print(f"  Segment {seg_num}: {seg_len:,} rows")
    
    return segments


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
