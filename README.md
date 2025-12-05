# Traffic Volume Forecasting — LSTM Encoder-Decoder (Seq2Seq)

Dự đoán lưu lượng giao thông (traffic_volume) sử dụng mô hình **LSTM Encoder-Decoder**.

## 📊 Tổng quan

| Thông tin | Chi tiết |
|-----------|----------|
| **Dataset** | Metro Interstate Traffic Volume (~48,204 bản ghi, 2012-2018) |
| **Input** | 24 giờ lịch sử (24 timesteps × 22 features) |
| **Output** | 5 giờ tương lai (traffic volume) |
| **Model** | LSTM Encoder-Decoder với Teacher Forcing |
| **Metrics** | RMSE, MAE, R², NSE cho từng horizon (t+1...t+5) |

## 🔑 Key Features

### Feature Engineering (LSTM-optimized)
- ✅ **Temporal features**: hour, day_of_week, month, season, is_weekend, is_rush_hour
- ✅ **Cyclical encoding**: sin/cos cho hour, day, month
- ✅ **Weather features**: temp, temp_celsius, clouds_all, rain_1h, snow_1h, is_rainy, is_snowy
- ✅ **Interaction**: rush_rain

### Segment-based Data Preparation
- Dữ liệu có ~2,588 gaps (timestamps bị thiếu)
- Tách thành 113 segments liên tục
- Mỗi sequence đảm bảo timestamps liên tục
- **30,871 rows usable** (76.1% của 40,575)
- Tất cả labels là dữ liệu thật (không interpolate)

## 🚀 Quick Start

```bash
# 1. Cài PyTorch với CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 2. Cài dependencies
pip install -r requirements.txt

# 3. Chạy notebooks theo thứ tự
```

## 📓 Notebooks

| # | Notebook | Mô tả |
|---|----------|-------|
| 01 | `01_EDA.ipynb` | Exploratory Data Analysis |
| 02 | `02_Preprocessing.ipynb` | Xử lý missing, duplicates, outliers |
| 03 | `03_Feature_Engineering.ipynb` | Tạo temporal, cyclical, weather features |
| 04 | `04_Feature_Selection.ipynb` | Chọn features cho LSTM (no leakage) |
| 05 | `05_Data_Preparation.ipynb` | Segment splitting, scaling, create sequences |
| 06 | `06_Model_Training.ipynb` | Train LSTM Encoder-Decoder |
| 06a | `06a_Optuna_Optimization.ipynb` | Hyperparameter tuning (optional) |
| 07 | `07_Evaluation.ipynb` | Đánh giá & visualization |

## 📁 Project Structure

```
DeepLearning_final/
├── data/
│   ├── raw/                    # Raw dataset
│   ├── processed/              # Cleaned, featured, selected data
│   └── sequences/              # Train/val/test sequences (.npy)
├── notebooks/                  # Jupyter notebooks
├── src/                        # Source code modules
├── models/                     # Saved models & checkpoints
├── results/                    # Metrics, predictions, figures
└── logs/                       # Training & Optuna logs
```

## 📈 Model Architecture

```
Input (24, 22) → Encoder (Bidirectional LSTM) → Context → Decoder (LSTM) → Output (5,)
                     ↓                                        ↑
              Hidden State ─────────────────────────────→ Initial State
```

## 📝 Notes

- **Scaling**: Fit scaler trên training data only → apply cho val/test
- **Split**: Time-based (70/15/15), không shuffle
- **Segment workflow**: Đảm bảo mỗi sequence có timestamps liên tục
- **Teacher Forcing**: Sử dụng trong training để tăng tốc học

## 📚 References

- Sutskever et al. (2014) - Sequence to Sequence Learning
- Metro Interstate Traffic Volume Dataset - UCI ML Repository

