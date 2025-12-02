# 📊 Project Structure: Traffic Volume Forecasting với Encoder-Decoder

## 🎯 Mục tiêu
Dự báo `traffic_volume` cho **5 bước thời gian tiếp theo** sử dụng mô hình **Encoder-Decoder (Seq2Seq)**

---

## 📁 Cấu trúc thư mục

```
DeepLearning_final/
│
├── data/
│   ├── raw/
│   │   └── Metro_Interstate_Traffic_Volume.csv    # Dữ liệu gốc
│   ├── processed/
│   │   ├── cleaned_data.csv                       # Sau preprocessing
│   │   ├── featured_data.csv                      # Sau feature engineering
│   │   └── selected_features.csv                  # Sau feature selection
│   └── sequences/
│       ├── X_train.npy                            # Training sequences
│       ├── y_train.npy
│       ├── X_val.npy                              # Validation sequences
│       ├── y_val.npy
│       ├── X_test.npy                             # Test sequences
│       └── y_test.npy
│
├── notebooks/
│   ├── 01_EDA.ipynb                               # Khám phá & phân tích dữ liệu
│   ├── 02_Preprocessing.ipynb                     # Tiền xử lý dữ liệu
│   ├── 03_Feature_Engineering.ipynb               # Tạo features mới
│   ├── 04_Feature_Selection.ipynb                 # Chọn features quan trọng
│   ├── 05_Data_Preparation.ipynb                  # Chuẩn bị data cho model
│   ├── 06_Model_Training.ipynb                    # Huấn luyện Encoder-Decoder
│   └── 07_Evaluation.ipynb                        # Đánh giá & visualization
│
├── src/
│   ├── __init__.py
│   ├── config.py                                  # Hyperparameters & paths
│   ├── data_preprocessing.py                      # Functions xử lý dữ liệu
│   ├── feature_engineering.py                     # Functions tạo features
│   ├── feature_selection.py                       # Functions chọn features
│   ├── dataset.py                                 # PyTorch Dataset class
│   ├── model.py                                   # Encoder-Decoder architecture
│   ├── train.py                                   # Training loop & callbacks
│   ├── evaluate.py                                # Metrics: R², NSE, MAE, RMSE
│   └── utils.py                                   # Utility functions
│
├── models/
│   ├── checkpoints/                               # Model checkpoints mỗi epoch
│   ├── best_model.pth                             # Best model weights
│   └── scaler.pkl                                 # Saved scaler object
│
├── results/
│   ├── figures/
│   │   ├── eda/                                   # EDA plots
│   │   ├── training/                              # Learning curves
│   │   └── evaluation/                            # Prediction plots
│   ├── metrics.json                               # Kết quả đánh giá
│   └── predictions.csv                            # Predicted values
│
├── Deep/                                          # Virtual environment
├── requirements.txt                               # Dependencies
├── PROJECT_STRUCTURE.md                           # File này
└── README.md                                      # Hướng dẫn project
```

---

## 📓 Chi tiết từng Notebook

### **01_EDA.ipynb** - Exploratory Data Analysis
```
Input:  data/raw/Metro_Interstate_Traffic_Volume.csv
Output: results/figures/eda/*.png
        Hiểu biết về dữ liệu

Tasks:
├── Load & inspect data (shape, dtypes, head/tail)
├── Basic statistics (describe, info)
├── Missing values analysis
├── Duplicates check
├── Target distribution (traffic_volume)
├── Time series visualization
├── Correlation heatmap
├── Feature distributions
└── Insights & findings summary
```

### **02_Preprocessing.ipynb** - Data Preprocessing
```
Input:  data/raw/Metro_Interstate_Traffic_Volume.csv
Output: data/processed/cleaned_data.csv

Tasks:
├── Handle DateTime (convert, sort, set index)
├── Handle missing values
├── Handle duplicates
├── Handle outliers (IQR/Z-score)
├── Resample to ensure hourly continuity
├── Data validation
└── Save cleaned data
```

### **03_Feature_Engineering.ipynb** - Create New Features
```
Input:  data/processed/cleaned_data.csv
Output: data/processed/featured_data.csv

Tasks:
├── Temporal features (hour, day, month, year, is_weekend, is_rush_hour)
├── Cyclical encoding (sin/cos for hour, day, month)
├── Lag features (t-1, t-24, t-168)
├── Rolling statistics (mean, std, min, max)
├── Difference features (diff, pct_change)
├── Holiday features
├── Weather engineering (temp_celsius, is_rainy, etc.)
├── Interaction features
└── Save featured data
```

### **04_Feature_Selection.ipynb** - Select Best Features
```
Input:  data/processed/featured_data.csv
Output: data/processed/selected_features.csv
        Danh sách features được chọn

Tasks:
├── Correlation analysis với target
├── Remove highly correlated features (>0.95)
├── Feature importance (Random Forest)
├── Mutual Information
├── Recursive Feature Elimination (RFE)
├── Final feature selection
├── Document lý do chọn/loại
└── Save selected features data
```

### **05_Data_Preparation.ipynb** - Prepare for Seq2Seq
```
Input:  data/processed/selected_features.csv
Output: data/sequences/X_train.npy, y_train.npy, etc.
        models/scaler.pkl

Tasks:
├── Define INPUT_SEQ_LEN, OUTPUT_SEQ_LEN
├── Scaling (fit on train only)
├── Create sequences (sliding window)
├── Train/Val/Test split (time-based, 70/15/15)
├── Save sequences as numpy arrays
├── Save scaler for inverse transform
└── Verify data shapes
```

### **06_Model_Training.ipynb** - Train Encoder-Decoder
```
Input:  data/sequences/*.npy
        models/scaler.pkl
Output: models/best_model.pth
        models/checkpoints/
        results/figures/training/

Tasks:
├── Load sequences & create DataLoaders
├── Define Encoder-Decoder architecture
├── Setup: loss, optimizer, scheduler
├── Training loop with:
│   ├── Early stopping
│   ├── Gradient clipping
│   ├── Model checkpointing
│   └── Progress tracking (tqdm)
├── Plot learning curves
└── Save best model
```

### **07_Evaluation.ipynb** - Evaluate & Visualize Results
```
Input:  models/best_model.pth
        data/sequences/X_test.npy, y_test.npy
        models/scaler.pkl
Output: results/metrics.json
        results/predictions.csv
        results/figures/evaluation/

Tasks:
├── Load model & test data
├── Generate predictions
├── Inverse transform predictions
├── Calculate metrics per step:
│   ├── R² (Coefficient of Determination)
│   ├── NSE (Nash-Sutcliffe Efficiency)
│   ├── MAE (Mean Absolute Error)
│   └── RMSE (Root Mean Squared Error)
├── Calculate average metrics
├── Visualizations:
│   ├── Actual vs Predicted time series
│   ├── Scatter plots
│   ├── Residual analysis
│   ├── Error by forecast horizon
│   └── Metrics summary table
├── Save results
└── Final conclusions
```

---

## 🔄 Pipeline hoàn chỉnh

### **PHASE 1: DATA EXPLORATION & UNDERSTANDING**
```
┌─────────────────────────────────────────────────────────────┐
│  1.1 Load Data                                              │
│      - Đọc CSV, kiểm tra shape, dtypes                      │
│      - Xem mẫu dữ liệu đầu/cuối                            │
│                                                             │
│  1.2 Basic Statistics                                       │
│      - describe(), info()                                   │
│      - Kiểm tra missing values                              │
│      - Kiểm tra duplicates                                  │
│                                                             │
│  1.3 Visualize                                              │
│      - Distribution của traffic_volume                      │
│      - Time series plot                                     │
│      - Correlation heatmap                                  │
└─────────────────────────────────────────────────────────────┘
```

### **PHASE 2: DATA PREPROCESSING**
```
┌─────────────────────────────────────────────────────────────┐
│  2.1 Handle DateTime                                        │
│      - Convert 'date_time' → datetime object                │
│      - Set as index hoặc sort by time                       │
│      - Kiểm tra time continuity (missing timestamps)        │
│                                                             │
│  2.2 Handle Missing Values                                  │
│      - Numerical: mean/median/interpolation                 │
│      - Categorical: mode/forward fill                       │
│                                                             │
│  2.3 Handle Duplicates                                      │
│      - Remove duplicate timestamps                          │
│      - Keep first/last/mean                                 │
│                                                             │
│  2.4 Handle Outliers                                        │
│      - IQR method / Z-score                                 │
│      - Clip hoặc remove                                     │
│                                                             │
│  2.5 Resample (nếu cần)                                     │
│      - Đảm bảo dữ liệu đều theo giờ                         │
│      - Fill missing hours                                   │
└─────────────────────────────────────────────────────────────┘
```

### **PHASE 3: FEATURE ENGINEERING**
```
┌─────────────────────────────────────────────────────────────┐
│  3.1 Temporal Features (từ date_time)                       │
│      ├── hour (0-23)                                        │
│      ├── day_of_week (0-6, Monday=0)                        │
│      ├── day_of_month (1-31)                                │
│      ├── month (1-12)                                       │
│      ├── year                                               │
│      ├── is_weekend (0/1)                                   │
│      ├── is_rush_hour (0/1) - giờ cao điểm 7-9, 16-18       │
│      ├── quarter (1-4)                                      │
│      ├── week_of_year (1-52)                                │
│      └── season (Spring/Summer/Fall/Winter)                 │
│                                                             │
│  3.2 Cyclical Encoding (cho features tuần hoàn)             │
│      ├── hour_sin = sin(2π × hour/24)                       │
│      ├── hour_cos = cos(2π × hour/24)                       │
│      ├── day_sin = sin(2π × day_of_week/7)                  │
│      ├── day_cos = cos(2π × day_of_week/7)                  │
│      ├── month_sin = sin(2π × month/12)                     │
│      └── month_cos = cos(2π × month/12)                     │
│                                                             │
│  3.3 Lag Features (Historical values)                       │
│      ├── traffic_lag_1h (t-1)                               │
│      ├── traffic_lag_2h (t-2)                               │
│      ├── traffic_lag_3h (t-3)                               │
│      ├── traffic_lag_6h (t-6)                               │
│      ├── traffic_lag_12h (t-12)                             │
│      ├── traffic_lag_24h (t-24) - cùng giờ hôm qua          │
│      ├── traffic_lag_168h (t-168) - cùng giờ tuần trước     │
│      └── traffic_lag_720h (t-720) - cùng giờ tháng trước    │
│                                                             │
│  3.4 Rolling Statistics (Window-based)                      │
│      ├── rolling_mean_3h                                    │
│      ├── rolling_mean_6h                                    │
│      ├── rolling_mean_12h                                   │
│      ├── rolling_mean_24h                                   │
│      ├── rolling_std_3h                                     │
│      ├── rolling_std_6h                                     │
│      ├── rolling_std_24h                                    │
│      ├── rolling_min_24h                                    │
│      ├── rolling_max_24h                                    │
│      └── ewm_mean (Exponential Weighted Mean)               │
│                                                             │
│  3.5 Difference Features                                    │
│      ├── diff_1h = traffic(t) - traffic(t-1)                │
│      ├── diff_24h = traffic(t) - traffic(t-24)              │
│      ├── pct_change_1h = (t - t-1) / t-1                    │
│      └── pct_change_24h                                     │
│                                                             │
│  3.6 Holiday Features                                       │
│      ├── is_holiday (0/1)                                   │
│      ├── holiday_type (encoded)                             │
│      ├── days_to_holiday                                    │
│      └── days_after_holiday                                 │
│                                                             │
│  3.7 Weather Feature Engineering                            │
│      ├── temp_celsius = temp - 273.15 (convert from Kelvin) │
│      ├── temp_category (cold/mild/warm/hot)                 │
│      ├── weather_encoded (Label/One-hot encoding)           │
│      ├── is_rainy (rain_1h > 0)                             │
│      ├── is_snowy (snow_1h > 0)                             │
│      ├── cloud_category (clear/partly/cloudy/overcast)      │
│      └── weather_severity_score                             │
│                                                             │
│  3.8 Interaction Features                                   │
│      ├── hour × is_weekend                                  │
│      ├── hour × is_holiday                                  │
│      ├── temp × is_rush_hour                                │
│      ├── rain × is_rush_hour                                │
│      └── weather × day_of_week                              │
└─────────────────────────────────────────────────────────────┘
```

### **PHASE 4: FEATURE SELECTION**
```
┌─────────────────────────────────────────────────────────────┐
│  4.1 Correlation Analysis                                   │
│      - Pearson correlation với target                       │
│      - Remove highly correlated features (>0.95)            │
│                                                             │
│  4.2 Feature Importance                                     │
│      - Random Forest importance                             │
│      - XGBoost importance                                   │
│      - Permutation importance                               │
│                                                             │
│  4.3 Statistical Tests                                      │
│      - Mutual Information                                   │
│      - ANOVA F-test                                         │
│                                                             │
│  4.4 Wrapper Methods                                        │
│      - Recursive Feature Elimination (RFE)                  │
│      - Sequential Feature Selection                         │
│                                                             │
│  4.5 Final Feature Set                                      │
│      - Select top K features                                │
│      - Document selected features và lý do                  │
└─────────────────────────────────────────────────────────────┘
```

### **PHASE 5: DATA PREPARATION FOR SEQ2SEQ**
```
┌─────────────────────────────────────────────────────────────┐
│  5.1 Scaling/Normalization                                  │
│      - MinMaxScaler (0-1) hoặc StandardScaler               │
│      - Fit on TRAIN only, transform on val/test             │
│      - Lưu scaler để inverse transform khi predict          │
│                                                             │
│  5.2 Create Sequences                                       │
│      - Input sequence length: N (e.g., 24, 48, 168)         │
│      - Output sequence length: 5 (predict 5 steps)          │
│      - Sliding window approach                              │
│                                                             │
│      Example:                                               │
│      X: [t-N, t-N+1, ..., t-1, t] → shape: (N, num_features)│
│      Y: [t+1, t+2, t+3, t+4, t+5] → shape: (5,)             │
│                                                             │
│  5.3 Train/Validation/Test Split                            │
│      - Time-based split (KHÔNG random shuffle!)             │
│      - Train: 70% (đầu tiên)                                │
│      - Validation: 15% (giữa)                               │
│      - Test: 15% (cuối cùng)                                │
│                                                             │
│  5.4 Create DataLoaders                                     │
│      - Batch size: 32, 64, 128                              │
│      - Shuffle=True cho train, False cho val/test           │
│      - num_workers cho parallel loading                     │
└─────────────────────────────────────────────────────────────┘
```

### **PHASE 6: MODEL ARCHITECTURE (ENCODER-DECODER)**
```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                    ENCODER                          │    │
│  │  ┌─────────────────────────────────────────────┐   │    │
│  │  │  Input: (batch, seq_len, input_features)    │   │    │
│  │  │              ↓                              │   │    │
│  │  │  LSTM/GRU Layers (stacked, bidirectional)   │   │    │
│  │  │              ↓                              │   │    │
│  │  │  Hidden State: (num_layers, batch, hidden)  │   │    │
│  │  │  Cell State: (num_layers, batch, hidden)    │   │    │
│  │  └─────────────────────────────────────────────┘   │    │
│  └─────────────────────────────────────────────────────┘    │
│                          ↓                                  │
│                    Context Vector                           │
│                          ↓                                  │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                    DECODER                          │    │
│  │  ┌─────────────────────────────────────────────┐   │    │
│  │  │  Input: Previous prediction + Context       │   │    │
│  │  │              ↓                              │   │    │
│  │  │  LSTM/GRU Layers                            │   │    │
│  │  │              ↓                              │   │    │
│  │  │  Fully Connected Layer                      │   │    │
│  │  │              ↓                              │   │    │
│  │  │  Output: (batch, output_seq_len, 1)         │   │    │
│  │  │         [t+1, t+2, t+3, t+4, t+5]           │   │    │
│  │  └─────────────────────────────────────────────┘   │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                             │
│  Optional Enhancements:                                     │
│  - Attention Mechanism (Bahdanau/Luong)                     │
│  - Teacher Forcing (training technique)                     │
│  - Dropout for regularization                               │
│  - Batch Normalization                                      │
└─────────────────────────────────────────────────────────────┘
```

### **PHASE 7: TRAINING**
```
┌─────────────────────────────────────────────────────────────┐
│  7.1 Loss Function                                          │
│      - MSELoss (Mean Squared Error)                         │
│      - MAELoss (L1Loss)                                     │
│      - HuberLoss (robust to outliers)                       │
│                                                             │
│  7.2 Optimizer                                              │
│      - Adam (lr=0.001)                                      │
│      - AdamW (with weight decay)                            │
│                                                             │
│  7.3 Learning Rate Scheduler                                │
│      - ReduceLROnPlateau                                    │
│      - CosineAnnealingLR                                    │
│      - StepLR                                               │
│                                                             │
│  7.4 Training Loop                                          │
│      - Epochs: 50-200                                       │
│      - Early Stopping (patience=10-20)                      │
│      - Gradient Clipping                                    │
│      - Model Checkpointing (save best model)                │
│                                                             │
│  7.5 Monitoring                                             │
│      - Training loss per epoch                              │
│      - Validation loss per epoch                            │
│      - Learning curves visualization                        │
└─────────────────────────────────────────────────────────────┘
```

### **PHASE 8: EVALUATION METRICS**
```
┌─────────────────────────────────────────────────────────────┐
│  8.1 R² (Coefficient of Determination)                      │
│      R² = 1 - (SS_res / SS_tot)                             │
│      SS_res = Σ(y_true - y_pred)²                           │
│      SS_tot = Σ(y_true - y_mean)²                           │
│      Range: (-∞, 1], Best = 1                               │
│                                                             │
│  8.2 NSE (Nash-Sutcliffe Efficiency)                        │
│      NSE = 1 - [Σ(y_true - y_pred)² / Σ(y_true - y_mean)²]  │
│      Tương tự R² nhưng dùng trong hydrology                 │
│      Range: (-∞, 1], Best = 1                               │
│      NSE > 0.5: acceptable, > 0.65: good, > 0.75: very good │
│                                                             │
│  8.3 MAE (Mean Absolute Error)                              │
│      MAE = (1/n) × Σ|y_true - y_pred|                       │
│      Range: [0, ∞), Best = 0                                │
│      Interpretable in original scale                        │
│                                                             │
│  8.4 RMSE (Root Mean Squared Error)                         │
│      RMSE = √[(1/n) × Σ(y_true - y_pred)²]                  │
│      Range: [0, ∞), Best = 0                                │
│      Penalizes large errors more                            │
│                                                             │
│  8.5 Metrics for Seq2Seq (Multi-step)                       │
│      - Calculate metrics for EACH step (t+1, t+2,...,t+5)   │
│      - Calculate AVERAGE metrics across all steps           │
│      - Analyze error degradation over horizon               │
└─────────────────────────────────────────────────────────────┘
```

### **PHASE 9: RESULTS VISUALIZATION**
```
┌─────────────────────────────────────────────────────────────┐
│  9.1 Training Curves                                        │
│      - Loss vs Epochs (train & validation)                  │
│      - Learning rate changes                                │
│                                                             │
│  9.2 Prediction Plots                                       │
│      - Actual vs Predicted time series                      │
│      - Scatter plot (Actual vs Predicted)                   │
│      - Residual plots                                       │
│                                                             │
│  9.3 Error Analysis                                         │
│      - Error distribution histogram                         │
│      - Error by time step (t+1 to t+5)                      │
│      - Error by hour/day/month                              │
│                                                             │
│  9.4 Metrics Summary Table                                  │
│      ┌──────────┬───────┬───────┬───────┬───────┬───────┐   │
│      │ Step     │ t+1   │ t+2   │ t+3   │ t+4   │ t+5   │   │
│      ├──────────┼───────┼───────┼───────┼───────┼───────┤   │
│      │ R²       │       │       │       │       │       │   │
│      │ NSE      │       │       │       │       │       │   │
│      │ MAE      │       │       │       │       │       │   │
│      │ RMSE     │       │       │       │       │       │   │
│      └──────────┴───────┴───────┴───────┴───────┴───────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## 📝 Features từ Dataset gốc

| Column | Type | Description |
|--------|------|-------------|
| `holiday` | Categorical | Tên ngày lễ hoặc None |
| `temp` | Numerical | Nhiệt độ (Kelvin) |
| `rain_1h` | Numerical | Lượng mưa trong 1 giờ (mm) |
| `snow_1h` | Numerical | Lượng tuyết trong 1 giờ (mm) |
| `clouds_all` | Numerical | % mây che phủ |
| `weather_main` | Categorical | Thời tiết chính (Clear, Clouds, Rain...) |
| `weather_description` | Categorical | Mô tả chi tiết thời tiết |
| `date_time` | DateTime | Timestamp |
| **`traffic_volume`** | **Numerical** | **TARGET - Lưu lượng giao thông** |

---

## 🚀 Quick Start Code Template

```python
# 1. Import libraries
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# 2. Load data
df = pd.read_csv('data/Metro_Interstate_Traffic_Volume.csv')

# 3. Preprocess
# ... (xem chi tiết trong notebook)

# 4. Feature Engineering
# ... (xem chi tiết trong notebook)

# 5. Create sequences
# INPUT_SEQ_LEN = 24  # Use 24 hours of history
# OUTPUT_SEQ_LEN = 5  # Predict next 5 hours

# 6. Build model
# class Encoder(nn.Module): ...
# class Decoder(nn.Module): ...
# class Seq2Seq(nn.Module): ...

# 7. Train
# ... 

# 8. Evaluate
# R², NSE, MAE, RMSE for each prediction step
```

---

## 📚 References

1. **Seq2Seq Papers:**
   - Sutskever et al. (2014) - Sequence to Sequence Learning
   - Bahdanau et al. (2015) - Attention Mechanism

2. **Time Series Forecasting:**
   - Multi-step forecasting strategies
   - Feature engineering for time series

3. **Metrics:**
   - NSE: Nash & Sutcliffe (1970)
   - Standard regression metrics

---

## ✅ Checklist

- [ ] EDA completed
- [ ] Missing values handled
- [ ] Duplicates removed
- [ ] Outliers handled
- [ ] Temporal features created
- [ ] Lag features created
- [ ] Rolling statistics created
- [ ] Features scaled
- [ ] Sequences created
- [ ] Data split (time-based)
- [ ] Model architecture defined
- [ ] Training completed
- [ ] Metrics calculated (R², NSE, MAE, RMSE)
- [ ] Results visualized
- [ ] Model saved
