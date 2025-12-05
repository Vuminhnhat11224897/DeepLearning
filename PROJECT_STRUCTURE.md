# 📊 Project Structure: Traffic Volume Forecasting với LSTM Encoder-Decoder

## 🎯 Mục tiêu
Dự báo `traffic_volume` cho **5 bước thời gian tiếp theo** sử dụng mô hình **LSTM Encoder-Decoder (Seq2Seq)**

---

## 📁 Cấu trúc thư mục

```
DeepLearning_final/
│
├── data/
│   ├── raw/
│   │   └── Metro_Interstate_Traffic_Volume.csv    # Dữ liệu gốc (~48,204 rows)
│   ├── processed/
│   │   ├── cleaned_data.csv                       # Sau preprocessing (40,575 rows)
│   │   ├── featured_data.csv                      # Sau feature engineering
│   │   ├── selected_features.csv                  # Sau feature selection (22 features)
│   │   └── selected_features_info.json            # Thông tin features đã chọn
│   └── sequences/
│       ├── X_train.npy, y_train.npy               # Training sequences
│       ├── X_val.npy, y_val.npy                   # Validation sequences
│       ├── X_test.npy, y_test.npy                 # Test sequences
│       └── metadata.json                          # Sequence metadata
│
├── notebooks/
│   ├── 01_EDA.ipynb                               # Exploratory Data Analysis
│   ├── 02_Preprocessing.ipynb                     # Data cleaning & preprocessing
│   ├── 03_Feature_Engineering.ipynb               # Feature creation (LSTM-safe)
│   ├── 04_Feature_Selection.ipynb                 # Feature selection (no leakage)
│   ├── 05_Data_Preparation.ipynb                  # Segment splitting & sequences
│   ├── 06_Model_Training.ipynb                    # LSTM Encoder-Decoder training
│   ├── 06a_Optuna_Optimization.ipynb              # Hyperparameter tuning
│   └── 07_Evaluation.ipynb                        # Evaluation & visualization
│
├── src/
│   ├── __init__.py
│   ├── config.py                                  # Hyperparameters & paths
│   ├── data_preprocessing.py                      # Preprocessing + segment splitting
│   ├── feature_engineering.py                     # LSTM-safe feature functions
│   ├── feature_selection.py                       # Feature selection (no leakage)
│   ├── dataset.py                                 # PyTorch Dataset & sequences
│   ├── model.py                                   # LSTM Encoder-Decoder architecture
│   ├── train.py                                   # Training loop
│   ├── evaluate.py                                # Metrics: R², NSE, MAE, RMSE
│   ├── optuna_optimization.py                     # Hyperparameter tuning
│   ├── logger.py                                  # Logging utilities
│   └── utils.py                                   # Helper functions
│
├── models/
│   ├── best_model.pth                             # Best model weights
│   ├── scaler.pkl                                 # Saved MinMaxScaler
│   └── checkpoints/                               # Epoch checkpoints
│
├── results/
│   ├── metrics.json                               # Evaluation metrics
│   ├── predictions.csv                            # Test predictions
│   └── figures/
│       ├── eda/                                   # EDA plots
│       ├── training/                              # Learning curves
│       └── evaluation/                            # Prediction plots
│
├── logs/
│   ├── best_params.json                           # Best hyperparameters
│   └── optuna_study.db                            # Optuna database
│
├── Deep/                                          # Virtual environment
├── requirements.txt                               # Dependencies
├── PROJECT_STRUCTURE.md                           # This file
└── README.md                                      # Project overview
```

---

## 📓 Chi tiết từng Notebook

### **01_EDA.ipynb** - Exploratory Data Analysis
```
Input:  data/raw/Metro_Interstate_Traffic_Volume.csv
Output: Insights về dữ liệu

Tasks:
├── Load & inspect data (48,204 rows, 9 columns)
├── Check duplicates (7,629 duplicates found)
├── Check missing timestamps (11,976 gaps)
├── Target distribution analysis
├── Time series visualization
├── Correlation analysis
└── Initial findings
```

### **02_Preprocessing.ipynb** - Data Preprocessing
```
Input:  data/raw/Metro_Interstate_Traffic_Volume.csv
Output: data/processed/cleaned_data.csv (40,575 rows)

Tasks:
├── Convert datetime & sort
├── Remove duplicates → 40,575 rows
├── Handle missing values
├── Handle outliers (IQR clipping)
├── Verify data quality
└── Save cleaned data
```

### **03_Feature_Engineering.ipynb** - LSTM-Safe Features
```
Input:  data/processed/cleaned_data.csv
Output: data/processed/featured_data.csv

⚠️ IMPORTANT: No lag/rolling/diff features for LSTM!

Tasks:
├── Temporal features (hour, day_of_week, month, season, etc.)
├── Cyclical encoding (sin/cos for hour, day, month)
├── Binary features (is_weekend, is_rush_hour)
├── Weather features (temp_celsius, is_rainy, is_snowy)
├── Interaction features (rush_rain)
└── Save featured data

❌ NOT included (by design):
├── Lag features (traffic_lag_1h, etc.)
├── Rolling statistics (rolling_mean, etc.)
└── Difference features (diff_1h, etc.)
→ LSTM learns these patterns from sequence input!
```

### **04_Feature_Selection.ipynb** - No-Leakage Selection
```
Input:  data/processed/featured_data.csv
Output: data/processed/selected_features.csv (22 features)
        data/processed/selected_features_info.json

Tasks:
├── Define LSTM-safe feature categories
├── Check for leakage features
├── Remove highly correlated features (>0.95)
├── Verify correlation with target
└── Save selected features

Selected Categories:
├── cyclical: hour_sin/cos, day_sin/cos, month_sin/cos
├── temporal: hour, day_of_week, month, season, etc.
├── context: is_weekend, is_rush_hour
├── weather: temp, temp_celsius, clouds_all, rain_1h, etc.
└── interaction: rush_rain
```

### **05_Data_Preparation.ipynb** - Segment-Based Workflow
```
Input:  data/processed/selected_features.csv
Output: data/sequences/*.npy
        models/scaler.pkl

⚠️ KEY INNOVATION: Segment-based splitting!

Tasks:
├── Load data (40,575 rows)
├── Split into continuous segments:
│   ├── Detect 2,588 timestamp gaps
│   ├── Create 113 valid segments
│   ├── Skip 2,476 short segments (<48 rows)
│   └── Usable: 30,871 rows (76.1%)
├── Scale data (fit on training portion only)
├── Create sequences from each segment independently
├── Train/Val/Test split (70/15/15, time-based)
└── Save sequences & metadata

Sequence Format:
├── X: (n_samples, 24, 22) - 24 hours × 22 features
└── y: (n_samples, 5) - next 5 hours traffic volume
```

### **06_Model_Training.ipynb** - LSTM Encoder-Decoder
```
Input:  data/sequences/*.npy
Output: models/best_model.pth
        results/figures/training/

Architecture:
┌─────────────────────────────────────────────────────────┐
│  Input (batch, 24, 22)                                  │
│         ↓                                               │
│  ┌─────────────────────────────────────────────────┐    │
│  │  ENCODER (Bidirectional LSTM)                   │    │
│  │  - hidden_size: 128                             │    │
│  │  - num_layers: 2                                │    │
│  │  - dropout: 0.2                                 │    │
│  └─────────────────────────────────────────────────┘    │
│         ↓                                               │
│  Context Vector (hidden_state, cell_state)              │
│         ↓                                               │
│  ┌─────────────────────────────────────────────────┐    │
│  │  DECODER (LSTM)                                 │    │
│  │  - Teacher Forcing during training              │    │
│  │  - Autoregressive during inference              │    │
│  └─────────────────────────────────────────────────┘    │
│         ↓                                               │
│  Output (batch, 5) - traffic volume for t+1...t+5       │
└─────────────────────────────────────────────────────────┘

Training:
├── Loss: MSELoss
├── Optimizer: Adam (lr=0.001)
├── Scheduler: ReduceLROnPlateau
├── Early Stopping (patience=15)
├── Gradient Clipping (max_norm=1.0)
└── Teacher Forcing Ratio: 0.5
```

### **07_Evaluation.ipynb** - Metrics & Visualization
```
Input:  models/best_model.pth
        data/sequences/X_test.npy, y_test.npy
Output: results/metrics.json
        results/predictions.csv
        results/figures/evaluation/

Metrics (per step t+1...t+5 and average):
├── R² (Coefficient of Determination)
├── NSE (Nash-Sutcliffe Efficiency)
├── MAE (Mean Absolute Error)
└── RMSE (Root Mean Squared Error)

Visualizations:
├── Actual vs Predicted time series
├── Scatter plots
├── Residual analysis
├── Error by forecast horizon
└── Metrics summary table
```

---

## 🔄 Data Flow Pipeline

```
Raw Data (48,204 rows)
    ↓
[02_Preprocessing] Remove duplicates, handle outliers
    ↓
Cleaned Data (40,575 rows, 2,588 gaps)
    ↓
[03_Feature_Engineering] Add temporal, cyclical, weather features
    ↓
Featured Data (40,575 rows, ~30 features)
    ↓
[04_Feature_Selection] Select LSTM-safe features (no leakage)
    ↓
Selected Data (40,575 rows, 22 features)
    ↓
[05_Data_Preparation] Split into 113 continuous segments
    ↓
Usable Data (30,871 rows from 113 segments)
    ↓
Create Sequences (sliding window within each segment)
    ↓
Sequences: X(n, 24, 22), y(n, 5)
    ↓
Train/Val/Test Split (70/15/15, time-based)
    ↓
[06_Model_Training] LSTM Encoder-Decoder
    ↓
[07_Evaluation] Metrics & Predictions
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
| `weather_main` | Categorical | Thời tiết chính |
| `weather_description` | Categorical | Mô tả chi tiết |
| `date_time` | DateTime | Timestamp |
| **`traffic_volume`** | **Numerical** | **TARGET** |

---

## 🔑 Key Design Decisions

### 1. No Lag/Rolling/Diff Features
```
❌ Không dùng: traffic_lag_1h, rolling_mean_24h, diff_1h, etc.
✅ Lý do: 
   - LSTM học temporal patterns từ sequence input
   - Thêm lag features thủ công gây data leakage
   - Model nhận 24 giờ history → tự "thấy" lag information
```

### 2. Segment-Based Workflow
```
❌ Không dùng: Tạo sequences từ toàn bộ data (có gaps)
✅ Lý do:
   - Data có 2,588 timestamp gaps
   - Sequence chồng qua gap → LSTM học sai pattern
   - Tách segments → mỗi sequence đảm bảo liên tục
```

### 3. Time-Based Split
```
❌ Không dùng: Random shuffle
✅ Lý do:
   - Time series cần giữ thứ tự thời gian
   - Train trên past → predict future
   - Shuffle gây data leakage từ future
```

---

## ✅ Checklist

- [x] EDA completed
- [x] Duplicates removed (7,629)
- [x] Missing values handled
- [x] Outliers clipped
- [x] Temporal features created
- [x] Cyclical encoding added
- [x] Weather features engineered
- [x] Features selected (no leakage)
- [x] Segments split (113 continuous)
- [x] Sequences created (24→5)
- [x] Data scaled (fit on train only)
- [x] Train/Val/Test split (time-based)
- [ ] Model trained
- [ ] Metrics calculated
- [ ] Results visualized
- [ ] Model saved

---

## 📚 References

1. **Seq2Seq**: Sutskever et al. (2014) - Sequence to Sequence Learning
2. **LSTM**: Hochreiter & Schmidhuber (1997) - Long Short-Term Memory
3. **Dataset**: Metro Interstate Traffic Volume - UCI ML Repository
4. **Metrics**: Nash & Sutcliffe (1970) - NSE for model evaluation
