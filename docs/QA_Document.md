# 📚 TÀI LIỆU HỎI ĐÁP - TRAFFIC VOLUME FORECASTING PROJECT

## PHẦN 1: CÂU HỎI LIÊN QUAN ĐẾN CODE

---

### 1.1 HIỂU DỮ LIỆU ĐƯỢC SỬ DỤNG TRONG PROJECT

#### 1.1.1 Giới thiệu Dataset

**Dataset:** Metro Interstate Traffic Volume  
**Nguồn:** UCI Machine Learning Repository  
**Mục tiêu:** Dự báo lưu lượng giao thông (`traffic_volume`) cho 5 giờ tiếp theo

#### 1.1.2 Các thuộc tính trong dữ liệu gốc

| STT | Thuộc tính | Kiểu dữ liệu | Mô tả | Vai trò |
|-----|------------|--------------|-------|---------|
| 1 | `date_time` | DateTime | Thời điểm ghi nhận (theo giờ) | Index thời gian |
| 2 | `holiday` | Categorical | Tên ngày lễ hoặc None | Feature |
| 3 | `temp` | Numerical | Nhiệt độ (Kelvin) | Feature |
| 4 | `rain_1h` | Numerical | Lượng mưa trong 1 giờ (mm) | Feature |
| 5 | `snow_1h` | Numerical | Lượng tuyết trong 1 giờ (mm) | Feature |
| 6 | `clouds_all` | Numerical | Phần trăm mây che phủ (%) | Feature |
| 7 | `weather_main` | Categorical | Thời tiết chính (Clear, Rain, Clouds...) | Feature |
| 8 | `weather_description` | Categorical | Mô tả chi tiết thời tiết | Feature |
| 9 | **`traffic_volume`** | **Numerical** | **Lưu lượng xe/giờ** | **TARGET** |

#### 1.1.3 Features được tạo thêm (Feature Engineering)

Sau quá trình Feature Engineering, các features mới được tạo:

**A. Temporal Features (từ date_time):**
| Feature | Mô tả | Ví dụ |
|---------|-------|-------|
| `hour` | Giờ trong ngày | 0-23 |
| `day_of_week` | Ngày trong tuần | 0=Monday, 6=Sunday |
| `day_of_month` | Ngày trong tháng | 1-31 |
| `month` | Tháng | 1-12 |
| `year` | Năm | 2012-2018 |
| `week_of_year` | Tuần trong năm | 1-52 |
| `quarter` | Quý | 1-4 |
| `season` | Mùa | 0=Spring, 1=Summer, 2=Fall, 3=Winter |
| `is_weekend` | Cuối tuần? | 0 hoặc 1 |
| `is_rush_hour` | Giờ cao điểm? (7-9h, 16-18h) | 0 hoặc 1 |

**B. Cyclical Features (mã hóa tuần hoàn):**
| Feature | Công thức | Mục đích |
|---------|-----------|----------|
| `hour_sin` | sin(2π × hour/24) | Mã hóa giờ dạng vòng tròn |
| `hour_cos` | cos(2π × hour/24) | Giờ 23 gần giờ 0 |
| `day_sin` | sin(2π × day_of_week/7) | Mã hóa ngày trong tuần |
| `day_cos` | cos(2π × day_of_week/7) | Chủ nhật gần thứ 2 |
| `month_sin` | sin(2π × month/12) | Mã hóa tháng |
| `month_cos` | cos(2π × month/12) | Tháng 12 gần tháng 1 |

**C. Weather Features:**
| Feature | Mô tả |
|---------|-------|
| `temp_celsius` | Nhiệt độ (°C) = temp - 273.15 |
| `is_rainy` | Có mưa? (rain_1h > 0) |
| `is_snowy` | Có tuyết? (snow_1h > 0) |

**D. Interaction Features:**
| Feature | Mô tả |
|---------|-------|
| `rush_rain` | is_rush_hour × is_rainy |

#### 1.1.4 Thuộc tính được sử dụng cho X_train và y_train

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           INPUT SEQUENCE (X)                            │
│                        Shape: (n_samples, 24, 22)                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  22 Features (tại mỗi timestep):                                │   │
│  │                                                                  │   │
│  │  1. traffic_volume (target cũng là feature cho input)           │   │
│  │  2. temp, temp_celsius                                          │   │
│  │  3. clouds_all, rain_1h, snow_1h                                │   │
│  │  4. hour, day_of_week, day_of_month, month, year                │   │
│  │  5. week_of_year, quarter, season                               │   │
│  │  6. is_weekend, is_rush_hour                                    │   │
│  │  7. hour_sin, hour_cos, day_sin, day_cos, month_sin, month_cos  │   │
│  │  8. is_rainy, is_snowy                                          │   │
│  │  9. rush_rain                                                   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Sequence length: 24 timesteps (24 giờ lịch sử)                        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          OUTPUT SEQUENCE (y)                            │
│                          Shape: (n_samples, 5)                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Chỉ có 1 thuộc tính: traffic_volume                                   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  y[0] = traffic_volume tại t+1                                  │   │
│  │  y[1] = traffic_volume tại t+2                                  │   │
│  │  y[2] = traffic_volume tại t+3                                  │   │
│  │  y[3] = traffic_volume tại t+4                                  │   │
│  │  y[4] = traffic_volume tại t+5                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Dự báo 5 giờ tiếp theo                                                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Tóm tắt:**
- **X_train:** Sử dụng **22 features** (bao gồm cả traffic_volume) trong **24 timesteps**
- **y_train:** Chỉ sử dụng **traffic_volume** cho **5 timesteps tiếp theo**

#### 1.1.5 Lý do không dùng Lag/Rolling/Diff Features

```
❌ KHÔNG SỬ DỤNG:
   - traffic_lag_1h, traffic_lag_24h (lag features)
   - rolling_mean_24h, rolling_std_6h (rolling statistics)
   - diff_1h, pct_change (difference features)

✅ LÝ DO:
   1. LSTM tự học temporal patterns từ sequence input
   2. Input đã có 24 giờ lịch sử → model "thấy" được lag information
   3. Thêm lag features thủ công → data leakage khi tạo sequences
```

---

### 1.2 HIỂU CÁCH TIỀN XỬ LÝ DỮ LIỆU

#### 1.2.1 Pipeline Tiền Xử Lý

```
Raw Data (48,204 rows)
        │
        ▼
┌───────────────────────────────────────┐
│  STEP 1: Convert DateTime             │
│  - Parse date_time → datetime object  │
│  - Sort by timestamp                  │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│  STEP 2: Handle Duplicates            │
│  - 7,629 duplicate timestamps found   │
│  - Aggregation strategy:              │
│    • temp, clouds_all → mean          │
│    • rain_1h, snow_1h → max           │
│    • traffic_volume → mean            │
│    • weather_main → combine unique    │
│    • holiday → any non-null           │
│  - Result: 40,575 rows                │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│  STEP 3: Handle Missing Values        │
│  - Numerical: interpolate (linear)    │
│  - Categorical: forward fill          │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│  STEP 4: Handle Outliers (IQR)        │
│  - Column: traffic_volume             │
│  - Method: IQR với factor = 1.5       │
│  - Q1 = 25th percentile               │
│  - Q3 = 75th percentile               │
│  - IQR = Q3 - Q1                      │
│  - Lower = Q1 - 1.5 × IQR             │
│  - Upper = Q3 + 1.5 × IQR             │
│  - Action: Clip values to [Lower,Upper]│
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│  STEP 5: Check Time Continuity        │
│  - KHÔNG resample/interpolate target  │
│  - Giữ nguyên data thật               │
│  - Xử lý gaps trong Data Preparation  │
└───────────────────────────────────────┘
        │
        ▼
Cleaned Data (40,575 rows)
```

#### 1.2.2 Code Xử Lý Duplicates (Chi tiết)

```python
# Thay vì chỉ drop duplicates, aggregation để giữ thông tin
df = df.groupby("date_time", as_index=False).agg({
    "temp": "mean",           # Trung bình nhiệt độ
    "rain_1h": "max",         # Nếu có mưa ở bất kỳ row nào → giữ
    "snow_1h": "max",         # Tương tự cho tuyết
    "clouds_all": "mean",     # Trung bình % mây
    "traffic_volume": "mean", # Trung bình lưu lượng
    "weather_main": lambda x: ",".join(sorted(set(x))),  # Gộp unique
    "holiday": lambda x: 0 if x.isna().all() else 1      # Binary
})
```

#### 1.2.3 Code Xử Lý Outliers (IQR Method)

```python
def handle_outliers_iqr(df, column, factor=1.5, method='clip'):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower = Q1 - factor * IQR
    upper = Q3 + factor * IQR
    
    # Clip values ngoài khoảng [lower, upper]
    df[column] = df[column].clip(lower=lower, upper=upper)
    
    return df
```

#### 1.2.4 Segment-Based Data Preparation (QUAN TRỌNG!)

**Vấn đề:** Dữ liệu có 2,588 timestamp gaps (không liên tục)

```
Timeline thực tế:
... 10:00 → 11:00 → 12:00 → [GAP] → 15:00 → 16:00 ...
                              ↑
                    Thiếu 13:00, 14:00
```

**Giải pháp:** Tách data thành các segments liên tục

```python
def split_continuous_segments(df, date_col, target_col, min_length=48, freq_hours=1):
    """
    Tách DataFrame thành các segment THỰC SỰ liên tục:
    1. Không có NaN trong target
    2. Không có gap trong timestamps (mỗi row cách nhau đúng freq_hours)
    """
    # Tính time difference giữa các rows liên tiếp
    time_diff = df[date_col].diff()
    expected_diff = pd.Timedelta(hours=freq_hours)
    
    # Phát hiện gap: time_diff > expected_diff
    is_gap = (time_diff > expected_diff + tolerance) | time_diff.isna()
    
    # Tạo segment ID: tăng mỗi khi gặp gap
    segment_id = is_gap.cumsum()
    
    # Chỉ giữ segments đủ dài (>= min_length)
    # ...
```

**Kết quả:**
```
Original rows:     40,575
Gaps detected:      2,588
Segments created:     113
Usable rows:       30,871 (76.1%)
```

#### 1.2.5 Scaling Data

```python
from sklearn.preprocessing import MinMaxScaler

# QUAN TRỌNG: Chỉ fit scaler trên TRAINING data
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit + Transform
X_val_scaled = scaler.transform(X_val)          # Chỉ Transform
X_test_scaled = scaler.transform(X_test)        # Chỉ Transform

# Lý do: Tránh data leakage từ validation/test vào training
```

#### 1.2.6 Train/Validation/Test Split

```
┌─────────────────────────────────────────────────────────────────┐
│                    TIMELINE-BASED SPLIT                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ◄─────── TRAIN (70%) ───────►◄─ VAL (15%) ─►◄─ TEST (15%) ─►  │
│                                                                  │
│  [Past data]                                  [Future data]      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

❌ KHÔNG shuffle vì:
   - Time series phải giữ thứ tự thời gian
   - Train trên quá khứ, predict tương lai
   - Shuffle → data leakage từ future
```

---

### 1.3 HIỂU THIẾT KẾ KIẾN TRÚC CỦA CÁC MẠNG ĐƯỢC SỬ DỤNG

#### 1.3.1 Tổng quan kiến trúc Encoder-Decoder (Seq2Seq)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     ENCODER-DECODER ARCHITECTURE                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  INPUT: (batch_size, 24, 22)                                           │
│  - batch_size: số samples                                              │
│  - 24: sequence length (24 giờ lịch sử)                                │
│  - 22: số features                                                     │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────┐     │
│  │                      ENCODER                                   │     │
│  │  ┌─────────────────────────────────────────────────────────┐  │     │
│  │  │  Bidirectional LSTM                                      │  │     │
│  │  │  - input_size: 22 (số features)                          │  │     │
│  │  │  - hidden_size: 64/128/256                               │  │     │
│  │  │  - num_layers: 2                                         │  │     │
│  │  │  - bidirectional: True                                   │  │     │
│  │  │  - dropout: 0.2-0.3                                      │  │     │
│  │  └─────────────────────────────────────────────────────────┘  │     │
│  │                           │                                    │     │
│  │                           ▼                                    │     │
│  │  ┌─────────────────────────────────────────────────────────┐  │     │
│  │  │  Linear Projection (if bidirectional)                   │  │     │
│  │  │  - hidden_size * 2 → hidden_size                        │  │     │
│  │  └─────────────────────────────────────────────────────────┘  │     │
│  │                           │                                    │     │
│  │                           ▼                                    │     │
│  │  Output: (hidden_state, cell_state)                           │     │
│  │  - hidden: (num_layers, batch, hidden_size)                   │     │
│  │  - cell: (num_layers, batch, hidden_size)                     │     │
│  └───────────────────────────────────────────────────────────────┘     │
│                              │                                          │
│                              │ Context Vector                           │
│                              ▼                                          │
│  ┌───────────────────────────────────────────────────────────────┐     │
│  │                      DECODER                                   │     │
│  │                                                                │     │
│  │  for t in range(5):  # 5 output steps                         │     │
│  │      ┌─────────────────────────────────────────────────────┐  │     │
│  │      │  LSTM Cell                                           │  │     │
│  │      │  - input: previous_output (or teacher forcing)       │  │     │
│  │      │  - hidden_state, cell_state from encoder/prev step   │  │     │
│  │      └─────────────────────────────────────────────────────┘  │     │
│  │                           │                                    │     │
│  │                           ▼                                    │     │
│  │      ┌─────────────────────────────────────────────────────┐  │     │
│  │      │  Fully Connected Layer                               │  │     │
│  │      │  - hidden_size → 1                                   │  │     │
│  │      └─────────────────────────────────────────────────────┘  │     │
│  │                           │                                    │     │
│  │                           ▼                                    │     │
│  │      prediction[t] = output                                   │     │
│  │                                                                │     │
│  └───────────────────────────────────────────────────────────────┘     │
│                                                                         │
│  OUTPUT: (batch_size, 5)                                               │
│  - 5 giá trị traffic_volume dự báo cho t+1, t+2, t+3, t+4, t+5        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

#### 1.3.2 Chi tiết Encoder

```python
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, 
                 dropout=0.2, bidirectional=True):
        super(Encoder, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,      # 22 features
            hidden_size=hidden_size,    # 128
            num_layers=num_layers,      # 2 layers stacked
            batch_first=True,           # Input: (batch, seq, features)
            dropout=dropout,            # Dropout between layers
            bidirectional=bidirectional # Đọc cả 2 chiều
        )
        
        # Project bidirectional output back to hidden_size
        if bidirectional:
            self.fc_hidden = nn.Linear(hidden_size * 2, hidden_size)
            self.fc_cell = nn.Linear(hidden_size * 2, hidden_size)
    
    def forward(self, x):
        # x: (batch, 24, 22)
        outputs, (hidden, cell) = self.lstm(x)
        # outputs: (batch, 24, hidden_size * 2) if bidirectional
        # hidden: (num_layers * 2, batch, hidden_size)
        
        if self.bidirectional:
            # Combine forward and backward states
            hidden = self.fc_hidden(...)  # → (num_layers, batch, hidden_size)
            cell = self.fc_cell(...)
        
        return outputs, (hidden, cell)
```

#### 1.3.3 Chi tiết Decoder

```python
class Decoder(nn.Module):
    def __init__(self, output_size=1, hidden_size=128, num_layers=2, dropout=0.2):
        super(Decoder, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=output_size,     # 1 (previous prediction)
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        
        self.fc = nn.Linear(hidden_size, output_size)  # → 1
    
    def forward(self, x, hidden, cell):
        # x: (batch, 1, 1) - previous output
        output, (hidden, cell) = self.lstm(x, (hidden, cell))
        prediction = self.fc(output)  # (batch, 1, 1)
        
        return prediction, (hidden, cell)
```

#### 1.3.4 Teacher Forcing

```python
def forward(self, x, target=None, teacher_forcing_ratio=0.5):
    # Encode
    _, (hidden, cell) = self.encoder(x)
    
    # Initial decoder input: last known traffic_volume
    decoder_input = x[:, -1, 0].unsqueeze(1).unsqueeze(2)  # (batch, 1, 1)
    
    outputs = []
    for t in range(5):  # 5 prediction steps
        prediction, (hidden, cell) = self.decoder(decoder_input, hidden, cell)
        outputs.append(prediction)
        
        # Teacher Forcing: dùng ground truth với xác suất teacher_forcing_ratio
        if target is not None and random.random() < teacher_forcing_ratio:
            decoder_input = target[:, t].unsqueeze(1).unsqueeze(2)  # Use true value
        else:
            decoder_input = prediction  # Use predicted value
    
    return torch.cat(outputs, dim=1)  # (batch, 5)
```

**Tác dụng của Teacher Forcing:**
- Trong training: Đôi khi dùng ground truth thay vì prediction làm input cho step tiếp theo
- Giúp model học nhanh hơn, ổn định hơn
- Tỷ lệ giảm dần theo epoch: `current_tf_ratio = tf_ratio * (1 - epoch/num_epochs)`

#### 1.3.5 Tại sao dùng Bidirectional Encoder?

```
Forward LSTM:  t=0 → t=1 → t=2 → ... → t=23
                                         ↓
                              forward_hidden_state
                              
Backward LSTM: t=0 ← t=1 ← t=2 ← ... ← t=23
                ↓
     backward_hidden_state

Combined: concat(forward, backward) → richer representation
```

**Lợi ích:**
- Encoder "nhìn" được cả quá khứ và tương lai trong input sequence
- Capture patterns từ cả 2 chiều (VD: traffic tăng trước giờ cao điểm, giảm sau)

---

### 1.4 HIỂU PHƯƠNG PHÁP ĐÁNH GIÁ CHẤT LƯỢNG CÁC MODEL

#### 1.4.1 Các Metrics được sử dụng

**1. R² (Coefficient of Determination)**

$$R^2 = 1 - \frac{SS_{res}}{SS_{tot}} = 1 - \frac{\sum_{i}(y_i - \hat{y}_i)^2}{\sum_{i}(y_i - \bar{y})^2}$$

| Giá trị | Ý nghĩa |
|---------|---------|
| R² = 1 | Dự báo hoàn hảo |
| R² = 0 | Model = dự báo bằng mean |
| R² < 0 | Model tệ hơn dự báo bằng mean |

**Đánh giá:**
- R² > 0.9: Excellent
- R² > 0.7: Good
- R² > 0.5: Moderate
- R² < 0.5: Poor

---

**2. NSE (Nash-Sutcliffe Efficiency)**

$$NSE = 1 - \frac{\sum_{i}(y_i - \hat{y}_i)^2}{\sum_{i}(y_i - \bar{y})^2}$$

| Giá trị | Ý nghĩa |
|---------|---------|
| NSE > 0.75 | Very Good |
| NSE > 0.65 | Good |
| NSE > 0.50 | Satisfactory |
| NSE < 0.50 | Unsatisfactory |

*Note: NSE và R² có công thức tương tự, nhưng NSE thường dùng trong hydrology và time series.*

---

**3. MAE (Mean Absolute Error)**

$$MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$$

- Đơn vị: Giống với target (vehicles/hour)
- Dễ hiểu: "Trung bình, model sai bao nhiêu?"
- Không phạt nặng outliers

---

**4. RMSE (Root Mean Squared Error)**

$$RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$$

- Đơn vị: Giống với target (vehicles/hour)
- Phạt nặng các errors lớn (do bình phương)
- RMSE ≥ MAE (bằng khi tất cả errors bằng nhau)

---

#### 1.4.2 Đánh giá Per-Step (Multi-Step Forecasting)

```python
def calculate_metrics_per_step(y_true, y_pred):
    """
    y_true, y_pred: shape (n_samples, 5)
    
    Tính metrics cho từng step: t+1, t+2, t+3, t+4, t+5
    """
    results = []
    for i in range(5):  # 5 steps
        metrics = {
            'Step': f't+{i+1}',
            'R2': r2_score(y_true[:, i], y_pred[:, i]),
            'NSE': calculate_nse(y_true[:, i], y_pred[:, i]),
            'MAE': mean_absolute_error(y_true[:, i], y_pred[:, i]),
            'RMSE': calculate_rmse(y_true[:, i], y_pred[:, i])
        }
        results.append(metrics)
    
    # Average across all steps
    avg_metrics = calculate_metrics(y_true.flatten(), y_pred.flatten())
    results.append({'Step': 'Average', **avg_metrics})
    
    return results
```

#### 1.4.3 Kết quả thực tế của Model

| Step | R² | NSE | MAE | RMSE |
|------|-----|-----|-----|------|
| t+1 | 0.9841 | 0.9841 | 178.9 | 249.6 |
| t+2 | 0.9783 | 0.9783 | 200.4 | 291.7 |
| t+3 | 0.9720 | 0.9720 | 211.8 | 331.0 |
| t+4 | 0.9665 | 0.9665 | 220.7 | 362.3 |
| t+5 | 0.9615 | 0.9615 | 233.7 | 388.8 |
| **Average** | **0.9725** | **0.9725** | **209.1** | **328.4** |

**Nhận xét:**
- R² > 0.96 cho tất cả steps → Excellent
- Error tăng dần theo horizon (t+1 chính xác nhất, t+5 kém nhất) → Expected behavior
- Model dự báo tốt cả 5 steps

#### 1.4.4 Inverse Transform trước khi đánh giá

```python
# Predictions đang ở scaled space [0, 1]
# Cần inverse transform về original space để metrics có ý nghĩa

def inverse_transform_predictions(y_scaled, scaler, target_idx):
    """
    y_scaled: (n_samples, 5) - scaled predictions
    scaler: fitted MinMaxScaler
    target_idx: index của traffic_volume trong feature list
    """
    n_samples, n_steps = y_scaled.shape
    n_features = scaler.n_features_in_
    
    y_original = np.zeros_like(y_scaled)
    
    for i in range(n_steps):
        # Tạo dummy array với đúng số features
        dummy = np.zeros((n_samples, n_features))
        dummy[:, target_idx] = y_scaled[:, i]
        
        # Inverse transform và lấy cột target
        y_original[:, i] = scaler.inverse_transform(dummy)[:, target_idx]
    
    return y_original
```

---

## PHẦN 2: CÂU HỎI LÝ THUYẾT

---

### Câu 1: Dữ liệu đầu vào để tính trạng thái ẩn $h_t$ trong RNN

**Trong mạng nơ-ron hồi tiếp (RNN), dữ liệu đầu vào để tính trạng thái ẩn $h_t$ tại node thứ $t$ gồm:**

1. **Input hiện tại $x_t$:** Vector đặc trưng tại thời điểm $t$
2. **Trạng thái ẩn trước đó $h_{t-1}$:** Thông tin từ các timesteps trước

**Công thức:**

$$h_t = \tanh(W_{xh} \cdot x_t + W_{hh} \cdot h_{t-1} + b_h)$$

Trong đó:
- $W_{xh}$: Ma trận trọng số từ input đến hidden
- $W_{hh}$: Ma trận trọng số từ hidden đến hidden (recurrent)
- $b_h$: Bias
- $\tanh$: Hàm kích hoạt

```
        ┌─────────┐
x_t ───►│         │
        │  CELL   ├───► h_t
h_{t-1}►│         │
        └─────────┘
```

---

### Câu 2: Các cổng trong mạng GRU

**Mạng GRU (Gated Recurrent Unit) có 2 cổng:**

#### 1. Update Gate ($z_t$) - Cổng cập nhật

$$z_t = \sigma(W_z \cdot [h_{t-1}, x_t] + b_z)$$

**Tác dụng:**
- Quyết định **bao nhiêu thông tin từ quá khứ** ($h_{t-1}$) được giữ lại
- Quyết định **bao nhiêu thông tin mới** ($\tilde{h}_t$) được thêm vào
- $z_t$ gần 1: Giữ nhiều thông tin cũ (long-term memory)
- $z_t$ gần 0: Cập nhật nhiều thông tin mới

#### 2. Reset Gate ($r_t$) - Cổng reset

$$r_t = \sigma(W_r \cdot [h_{t-1}, x_t] + b_r)$$

**Tác dụng:**
- Quyết định **bao nhiêu thông tin quá khứ cần "quên"** khi tính candidate hidden state
- $r_t$ gần 0: "Quên" nhiều thông tin cũ
- $r_t$ gần 1: Giữ thông tin cũ để tính state mới

#### Công thức tính hidden state:

$$\tilde{h}_t = \tanh(W_h \cdot [r_t \odot h_{t-1}, x_t] + b_h)$$

$$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$

```
┌─────────────────────────────────────────────────────────────┐
│                         GRU CELL                             │
│                                                              │
│    ┌─────────┐                    ┌─────────┐               │
│    │ Reset   │                    │ Update  │               │
│    │ Gate r_t│                    │ Gate z_t│               │
│    └────┬────┘                    └────┬────┘               │
│         │                              │                     │
│         ▼                              │                     │
│    ┌─────────┐                         │                     │
│    │Candidate│                         │                     │
│    │  h̃_t    │                         │                     │
│    └────┬────┘                         │                     │
│         │                              │                     │
│         └──────────►(×)◄───────────────┘                    │
│                      │                                       │
│                      ▼                                       │
│                    h_t                                       │
└─────────────────────────────────────────────────────────────┘
```

---

### Câu 3: Các cổng trong mạng LSTM

**Mạng LSTM (Long Short-Term Memory) có 3 cổng:**

#### 1. Forget Gate ($f_t$) - Cổng quên

$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

**Tác dụng:**
- Quyết định **thông tin nào từ cell state cũ $C_{t-1}$ cần được loại bỏ**
- $f_t$ gần 0: Quên thông tin
- $f_t$ gần 1: Giữ thông tin

#### 2. Input Gate ($i_t$) - Cổng đầu vào

$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$

$$\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$$

**Tác dụng:**
- $i_t$: Quyết định **thông tin mới nào sẽ được lưu vào cell state**
- $\tilde{C}_t$: Candidate values có thể thêm vào cell state
- Kết hợp: $i_t \odot \tilde{C}_t$ = thông tin mới thực sự được thêm

#### 3. Output Gate ($o_t$) - Cổng đầu ra

$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$

**Tác dụng:**
- Quyết định **phần nào của cell state được xuất ra** làm hidden state
- Lọc thông tin từ cell state để tạo output

#### Công thức cập nhật Cell State và Hidden State:

$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$

$$h_t = o_t \odot \tanh(C_t)$$

```
┌─────────────────────────────────────────────────────────────────────┐
│                            LSTM CELL                                 │
│                                                                      │
│  C_{t-1} ─────────►(×)────────────►(+)────────────────► C_t         │
│                     ▲               ▲                                │
│                     │               │                                │
│                 ┌───┴───┐       ┌───┴───┐                           │
│                 │Forget │       │ Input │                            │
│                 │Gate f │       │Gate i │                            │
│                 │  t    │       │  t    │                            │
│                 └───────┘       └───┬───┘                           │
│                                     │                                │
│                                 ┌───┴───┐                           │
│                                 │ C̃_t   │                           │
│                                 │(tanh) │                           │
│                                 └───────┘                           │
│                                                                      │
│  h_{t-1} ─────────────────────────────────────────┬────► h_t        │
│                                                    │                 │
│                                                ┌───┴───┐            │
│                                                │Output │            │
│                                                │Gate o │            │
│                                                │  t    │            │
│                                                └───────┘            │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

### Câu 4: Tính trạng thái ẩn $h_t$ trong Bidirectional RNN

**Trong mạng Bidirectional RNN, trạng thái ẩn $h_t$ được tính từ 2 hướng:**

#### Forward Direction (Thuận):
Xử lý sequence từ $t=1$ đến $t=T$

$$\overrightarrow{h_t} = f(\overrightarrow{W_{xh}} \cdot x_t + \overrightarrow{W_{hh}} \cdot \overrightarrow{h_{t-1}} + \overrightarrow{b_h})$$

#### Backward Direction (Nghịch):
Xử lý sequence từ $t=T$ đến $t=1$

$$\overleftarrow{h_t} = f(\overleftarrow{W_{xh}} \cdot x_t + \overleftarrow{W_{hh}} \cdot \overleftarrow{h_{t+1}} + \overleftarrow{b_h})$$

#### Kết hợp (Concatenation):

$$h_t = [\overrightarrow{h_t}; \overleftarrow{h_t}]$$

hoặc:

$$h_t = \overrightarrow{h_t} + \overleftarrow{h_t} \quad \text{(sum)}$$

$$h_t = (\overrightarrow{h_t} + \overleftarrow{h_t}) / 2 \quad \text{(average)}$$

```
Forward:    x_1 ──► h_1 ──► h_2 ──► h_3 ──► ... ──► h_T
                    ↓       ↓       ↓               ↓
Combine:           [;]     [;]     [;]             [;]
                    ↑       ↑       ↑               ↑
Backward:   x_1 ◄── h_1 ◄── h_2 ◄── h_3 ◄── ... ◄── h_T
```

**Ưu điểm:**
- Hidden state tại $t$ chứa thông tin từ **cả quá khứ và tương lai**
- Hữu ích cho các task cần context đầy đủ (NER, POS tagging, machine translation encoder)

---

### Câu 5: Vai trò của Encoder và Decoder trong Seq2Seq

#### ENCODER

**Vai trò:**
1. **Đọc và hiểu** toàn bộ input sequence
2. **Nén thông tin** thành một vector ngữ cảnh cố định (context vector)
3. **Trích xuất đặc trưng** quan trọng từ input

**Cách hoạt động:**
- Nhận input sequence: $X = (x_1, x_2, ..., x_T)$
- Xử lý tuần tự qua các RNN/LSTM cells
- Output: Hidden state cuối cùng (hoặc tất cả hidden states nếu dùng Attention)

```
INPUT: "I love machine learning"

        x_1      x_2       x_3          x_4
         │        │         │            │
         ▼        ▼         ▼            ▼
      ┌─────┐  ┌─────┐   ┌─────┐     ┌─────┐
      │ h_1 ├─►│ h_2 ├──►│ h_3 ├────►│ h_4 ├──► Context Vector
      └─────┘  └─────┘   └─────┘     └─────┘
        "I"    "love"  "machine"   "learning"
```

#### DECODER

**Vai trò:**
1. **Nhận context vector** từ Encoder
2. **Sinh ra output sequence** từng phần tử một
3. **Dịch/Chuyển đổi** thông tin từ context thành output mong muốn

**Cách hoạt động:**
- Khởi tạo hidden state từ context vector của Encoder
- Sinh output từng bước: $y_1, y_2, ..., y_{T'}$
- Mỗi bước: Input = output của bước trước (hoặc ground truth nếu teacher forcing)

```
Context Vector
      │
      ▼
   ┌─────┐     ┌─────┐     ┌─────┐     ┌─────┐
   │ s_1 ├────►│ s_2 ├────►│ s_3 ├────►│ s_4 │
   └──┬──┘     └──┬──┘     └──┬──┘     └──┬──┘
      │           │           │           │
      ▼           ▼           ▼           ▼
    "Tôi"      "yêu"       "học"       "máy"

OUTPUT: "Tôi yêu học máy"
```

**Tổng kết:**

| Component | Input | Output | Vai trò |
|-----------|-------|--------|---------|
| **Encoder** | Source sequence $(x_1, ..., x_T)$ | Context vector | Nén thông tin input |
| **Decoder** | Context + previous outputs | Target sequence $(y_1, ..., y_{T'})$ | Sinh output tuần tự |

---

### Câu 6: Key, Value, Query trong Attention (Encoder-Decoder)

**Trong cơ chế Attention giữa Encoder-Decoder:**

#### Query (Q) - Truy vấn
- **Nguồn:** Hidden state của **Decoder** tại bước hiện tại: $Q = s_t$
- **Ý nghĩa:** "Tôi (decoder) đang ở trạng thái này, cần thông tin gì từ encoder?"
- **Vai trò:** Đại diện cho "câu hỏi" cần tìm thông tin liên quan

#### Key (K) - Khóa
- **Nguồn:** Tất cả hidden states của **Encoder**: $K = (h_1, h_2, ..., h_T)$
- **Ý nghĩa:** "Đây là các keys để so sánh với query"
- **Vai trò:** Dùng để tính độ tương đồng với Query

#### Value (V) - Giá trị
- **Nguồn:** Tất cả hidden states của **Encoder**: $V = (h_1, h_2, ..., h_T)$
- **Ý nghĩa:** "Đây là thông tin thực sự sẽ được lấy"
- **Vai trò:** Thông tin được trích xuất dựa trên attention weights

**Lưu ý:** Trong Encoder-Decoder Attention cơ bản, $K = V$ (đều là encoder hidden states)

#### Công thức tính Attention:

**1. Tính attention scores (alignment):**

$$e_{t,i} = score(s_t, h_i)$$

Các hàm score phổ biến:
- **Dot product:** $e_{t,i} = s_t^T \cdot h_i$
- **General:** $e_{t,i} = s_t^T \cdot W_a \cdot h_i$
- **Concat (Bahdanau):** $e_{t,i} = v_a^T \cdot \tanh(W_a \cdot [s_t; h_i])$

**2. Softmax để được attention weights:**

$$\alpha_{t,i} = \frac{\exp(e_{t,i})}{\sum_{j=1}^{T}\exp(e_{t,j})}$$

**3. Tính context vector:**

$$c_t = \sum_{i=1}^{T} \alpha_{t,i} \cdot h_i$$

```
                    Encoder Hidden States (K, V)
                 h_1      h_2      h_3      h_4
                  │        │        │        │
                  │        │        │        │
   Query ────────►┼────────┼────────┼────────┤
   (s_t)          │        │        │        │
                  ▼        ▼        ▼        ▼
              ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐
              │α_{t,1}│ │α_{t,2}│ │α_{t,3}│ │α_{t,4}│  Attention Weights
              └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘
                 │        │        │        │
                 ▼        ▼        ▼        ▼
              h_1×α    h_2×α    h_3×α    h_4×α
                 │        │        │        │
                 └────────┴────────┴────────┘
                              │
                              ▼
                    Context Vector (c_t)
```

---

### Câu 7: Key, Value, Query trong Self-Attention

**Trong Self-Attention, Q, K, V đều được tạo từ CÙNG MỘT input:**

Cho input sequence $X = (x_1, x_2, ..., x_n)$

#### Query, Key, Value được tính:

$$Q = X \cdot W^Q$$
$$K = X \cdot W^K$$
$$V = X \cdot W^V$$

Trong đó:
- $W^Q, W^K, W^V$ là các ma trận trọng số học được
- $X$: Input embeddings, shape $(n, d_{model})$
- $Q, K, V$: shape $(n, d_k)$ hoặc $(n, d_v)$

**Điểm khác biệt với Encoder-Decoder Attention:**

| | Self-Attention | Encoder-Decoder Attention |
|--|---------------|---------------------------|
| **Q nguồn** | Input sequence | Decoder hidden state |
| **K nguồn** | Input sequence (same as Q) | Encoder hidden states |
| **V nguồn** | Input sequence (same as Q, K) | Encoder hidden states |
| **Mục đích** | Học mối quan hệ giữa các phần tử trong cùng sequence | Học alignment giữa input và output |

#### Công thức Self-Attention (Scaled Dot-Product):

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

```
Input: "The cat sat on the mat"
        ↓
   ┌────┴────┐
   │Embedding│
   └────┬────┘
        │
        ├────────────┬────────────┐
        ▼            ▼            ▼
   ┌────────┐   ┌────────┐   ┌────────┐
   │   W^Q  │   │   W^K  │   │   W^V  │
   └────┬───┘   └────┬───┘   └────┬───┘
        │            │            │
        ▼            ▼            ▼
        Q            K            V
        │            │            │
        │     ┌──────┴            │
        ▼     ▼                   │
      Q × K^T                     │
        │                         │
        ▼                         │
   ÷ √d_k                         │
        │                         │
        ▼                         │
    Softmax                       │
        │                         │
        └─────────────────────────┼───► × V
                                  │
                                  ▼
                             Output (Attention)
```

---

### Câu 8: Ý tưởng của Multi-Head Attention

**Ý tưởng chính:**

Thay vì thực hiện **một phép attention** với $d_{model}$ dimensions, chia thành **nhiều "heads"** thực hiện attention song song trên các không gian con khác nhau.

#### Tại sao cần Multi-Head?

1. **Capture nhiều loại relationships:**
   - Head 1: Có thể học syntactic relationships
   - Head 2: Có thể học semantic relationships
   - Head 3: Có thể học positional relationships
   - ...

2. **Attention ở nhiều positions khác nhau:**
   - Một head có thể focus vào từ gần
   - Head khác có thể focus vào từ xa

3. **Richer representation:**
   - Kết hợp nhiều perspectives

#### Công thức Multi-Head Attention:

**1. Tạo multiple heads:**

Với mỗi head $i$ (từ 1 đến $h$):

$$Q_i = Q \cdot W_i^Q, \quad K_i = K \cdot W_i^K, \quad V_i = V \cdot W_i^V$$

Trong đó:
- $W_i^Q \in \mathbb{R}^{d_{model} \times d_k}$
- $W_i^K \in \mathbb{R}^{d_{model} \times d_k}$
- $W_i^V \in \mathbb{R}^{d_{model} \times d_v}$
- Thường: $d_k = d_v = d_{model} / h$

**2. Tính attention cho mỗi head:**

$$\text{head}_i = \text{Attention}(Q_i, K_i, V_i) = \text{softmax}\left(\frac{Q_i K_i^T}{\sqrt{d_k}}\right) V_i$$

**3. Concatenate tất cả heads:**

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h) \cdot W^O$$

Trong đó $W^O \in \mathbb{R}^{hd_v \times d_{model}}$

```
                    Input (Q, K, V)
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
     ┌─────────┐    ┌─────────┐     ┌─────────┐
     │ Head 1  │    │ Head 2  │ ... │ Head h  │
     │ W_1^Q   │    │ W_2^Q   │     │ W_h^Q   │
     │ W_1^K   │    │ W_2^K   │     │ W_h^K   │
     │ W_1^V   │    │ W_2^V   │     │ W_h^V   │
     └────┬────┘    └────┬────┘     └────┬────┘
          │              │               │
          ▼              ▼               ▼
     ┌─────────┐    ┌─────────┐     ┌─────────┐
     │Attention│    │Attention│     │Attention│
     │ head_1  │    │ head_2  │ ... │ head_h  │
     └────┬────┘    └────┬────┘     └────┬────┘
          │              │               │
          └──────────────┼───────────────┘
                         │
                         ▼
                  ┌────────────┐
                  │ Concat     │
                  └──────┬─────┘
                         │
                         ▼
                  ┌────────────┐
                  │   W^O      │
                  └──────┬─────┘
                         │
                         ▼
                  MultiHead Output
```

#### Ví dụ với Transformer (h=8 heads):

| Parameter | Giá trị |
|-----------|---------|
| $d_{model}$ | 512 |
| $h$ (số heads) | 8 |
| $d_k = d_v$ | 512/8 = 64 |

- Mỗi head có dimension 64
- 8 heads chạy song song
- Concat: 8 × 64 = 512
- Project qua $W^O$: 512 → 512

---

## TÓM TẮT

### Phần Code:
1. **Dữ liệu:** 9 thuộc tính gốc + 22 features sau engineering → X_train (24 timesteps × 22 features), y_train (5 timesteps × 1 target)
2. **Tiền xử lý:** Handle duplicates (aggregation), outliers (IQR), segment splitting (2,588 gaps → 113 segments)
3. **Kiến trúc:** Bidirectional LSTM Encoder + LSTM Decoder với Teacher Forcing
4. **Đánh giá:** R², NSE, MAE, RMSE per step và average

### Phần Lý thuyết:
1. **RNN:** $h_t = f(x_t, h_{t-1})$
2. **GRU:** 2 cổng (Update, Reset)
3. **LSTM:** 3 cổng (Forget, Input, Output) + Cell State
4. **BiRNN:** Forward + Backward hidden states
5. **Encoder-Decoder:** Encoder nén input → Context → Decoder sinh output
6. **Attention (Enc-Dec):** Q=decoder state, K=V=encoder states
7. **Self-Attention:** Q, K, V từ cùng input với learned projections
8. **Multi-Head:** h heads parallel attention → concat → project

---

*Document created for Traffic Volume Forecasting Project - Deep Learning Course*
