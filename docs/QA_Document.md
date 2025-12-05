# 📚 TÀI LIỆU HỎI ĐÁP - TRAFFIC VOLUME FORECASTING PROJECT

## MỤC LỤC

- [PHẦN 1: CÂU HỎI LIÊN QUAN ĐẾN CODE](#phần-1-câu-hỏi-liên-quan-đến-code)
  - [1.1 Hiểu dữ liệu](#11-hiểu-dữ-liệu-được-sử-dụng-trong-project)
  - [1.2 Tiền xử lý dữ liệu](#12-hiểu-cách-tiền-xử-lý-dữ-liệu)
  - [1.3 Kiến trúc Encoder-Decoder](#13-hiểu-thiết-kế-kiến-trúc-của-các-mạng-được-sử-dụng)
  - [1.4 Optuna Hyperparameter Optimization](#14-optuna-hyperparameter-optimization)
  - [1.5 Phương pháp đánh giá Model](#15-hiểu-phương-pháp-đánh-giá-chất-lượng-các-model)
- [PHẦN 2: CÂU HỎI LÝ THUYẾT](#phần-2-câu-hỏi-lý-thuyết)

---

## PHẦN 1: CÂU HỎI LIÊN QUAN ĐẾN CODE

---

### 1.1 HIỂU DỮ LIỆU ĐƯỢC SỬ DỤNG TRONG PROJECT

#### 1.1.1 Giới thiệu Dataset

**Dataset:** Metro Interstate Traffic Volume  
**Nguồn:** UCI Machine Learning Repository  
**Mục tiêu:** Dự báo lưu lượng giao thông (`traffic_volume`) cho 5 giờ tiếp theo

**Bài toán:** Đây là bài toán **Multi-step Time Series Forecasting** (Dự báo chuỗi thời gian nhiều bước):
- **Input:** 24 giờ dữ liệu lịch sử (24 timesteps × 22 features)
- **Output:** 5 giá trị traffic_volume cho 5 giờ tiếp theo (t+1, t+2, t+3, t+4, t+5)
- **Mô hình:** LSTM Encoder-Decoder (Seq2Seq)

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

#### 1.3.1 Tại sao dùng Encoder-Decoder cho bài toán này?

**Bài toán:** Dự báo chuỗi thời gian nhiều bước (Multi-step Forecasting)
- **Input:** Sequence có độ dài 24 (24 giờ lịch sử)
- **Output:** Sequence có độ dài 5 (5 giờ dự báo)

**Vấn đề với các phương pháp truyền thống:**

| Phương pháp | Mô tả | Nhược điểm |
|-------------|-------|------------|
| **Direct Multi-Output** | 1 model → 5 outputs cùng lúc | Không capture dependencies giữa các outputs |
| **Recursive (Iterated)** | Predict t+1, dùng làm input predict t+2,... | Error accumulation nghiêm trọng |
| **Direct (5 models)** | 5 models riêng biệt | Không share knowledge, tốn resources |

**Giải pháp: Encoder-Decoder (Seq2Seq)**

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    TẠI SAO ENCODER-DECODER?                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. ENCODER:                                                            │
│     - Đọc TOÀN BỘ 24 giờ lịch sử                                       │
│     - Nén thông tin thành "Context Vector"                             │
│     - Context chứa đủ thông tin để dự báo                              │
│                                                                         │
│  2. DECODER:                                                            │
│     - Sinh output TUẦN TỰ: t+1 → t+2 → t+3 → t+4 → t+5                │
│     - Mỗi step nhận: context + output của step trước                   │
│     - Capture được dependencies giữa các predictions                   │
│                                                                         │
│  3. LỢI ÍCH:                                                            │
│     - Flexible input/output length (24 → 5)                            │
│     - Sequential generation (như language model)                        │
│     - Teacher forcing giúp train ổn định                                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

#### 1.3.2 Kiến trúc tổng quan

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     ENCODER-DECODER ARCHITECTURE                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  INPUT: (batch_size, 24, 22)                                           │
│  - batch_size: số samples trong 1 batch                                │
│  - 24: sequence length (24 giờ lịch sử)                                │
│  - 22: số features (traffic_volume + các features khác)                │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────┐     │
│  │                      ENCODER                                   │     │
│  │  ┌─────────────────────────────────────────────────────────┐  │     │
│  │  │  Bidirectional LSTM                                      │  │     │
│  │  │  - input_size: 22 (số features)                          │  │     │
│  │  │  - hidden_size: 64 (optimized by Optuna)                 │  │     │
│  │  │  - num_layers: 2 (stacked LSTM)                          │  │     │
│  │  │  - bidirectional: True (forward + backward)              │  │     │
│  │  │  - dropout: 0.248                                        │  │     │
│  │  └─────────────────────────────────────────────────────────┘  │     │
│  │                           │                                    │     │
│  │                           ▼                                    │     │
│  │  ┌─────────────────────────────────────────────────────────┐  │     │
│  │  │  Linear Projection (vì bidirectional)                   │  │     │
│  │  │  - 64*2 = 128 → 64                                       │  │     │
│  │  └─────────────────────────────────────────────────────────┘  │     │
│  │                           │                                    │     │
│  │                           ▼                                    │     │
│  │  Output: (hidden_state, cell_state) = "Context Vector"        │     │
│  │  - hidden: (2, batch, 64) → 2 layers, 64 dimensions          │     │
│  │  - cell: (2, batch, 64) → memory của LSTM                    │     │
│  └───────────────────────────────────────────────────────────────┘     │
│                              │                                          │
│                              │ Context Vector truyền sang Decoder       │
│                              ▼                                          │
│  ┌───────────────────────────────────────────────────────────────┐     │
│  │                      DECODER                                   │     │
│  │                                                                │     │
│  │  Lặp 5 lần (5 output steps):                                  │     │
│  │      ┌─────────────────────────────────────────────────────┐  │     │
│  │      │  LSTM Cell                                           │  │     │
│  │      │  - input: previous_output (hoặc ground truth)        │  │     │
│  │      │  - hidden: từ encoder hoặc step trước                │  │     │
│  │      │  - cell: từ encoder hoặc step trước                  │  │     │
│  │      └─────────────────────────────────────────────────────┘  │     │
│  │                           │                                    │     │
│  │                           ▼                                    │     │
│  │      ┌─────────────────────────────────────────────────────┐  │     │
│  │      │  Fully Connected: hidden_size → 1                    │  │     │
│  │      │  Output: 1 giá trị traffic_volume                    │  │     │
│  │      └─────────────────────────────────────────────────────┘  │     │
│  │                           │                                    │     │
│  │                           ▼                                    │     │
│  │      predictions[t] = output value                            │     │
│  │                                                                │     │
│  └───────────────────────────────────────────────────────────────┘     │
│                                                                         │
│  OUTPUT: (batch_size, 5)                                               │
│  - 5 giá trị traffic_volume cho t+1, t+2, t+3, t+4, t+5               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

#### 1.3.3 ENCODER - Chi tiết hoạt động

##### A. LSTM Cell - Đơn vị cơ bản

**LSTM (Long Short-Term Memory)** giải quyết vấn đề vanishing gradient của RNN thông thường bằng cách sử dụng **3 cổng** và **cell state**:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         LSTM CELL CHI TIẾT                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Input tại timestep t:                                                  │
│  • x_t: input vector (22 features)                                     │
│  • h_{t-1}: hidden state từ timestep trước                             │
│  • C_{t-1}: cell state từ timestep trước                               │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  FORGET GATE (Cổng quên): Quyết định quên thông tin nào        │   │
│  │                                                                  │   │
│  │  f_t = σ(W_f · [h_{t-1}, x_t] + b_f)                            │   │
│  │                                                                  │   │
│  │  • f_t ≈ 0: Quên thông tin từ cell state cũ                    │   │
│  │  • f_t ≈ 1: Giữ thông tin từ cell state cũ                     │   │
│  │                                                                  │   │
│  │  Ví dụ Traffic: Khi chuyển từ ngày thường → ngày lễ,            │   │
│  │  forget gate "quên" patterns ngày thường                        │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  INPUT GATE (Cổng đầu vào): Quyết định thêm thông tin mới nào  │   │
│  │                                                                  │   │
│  │  i_t = σ(W_i · [h_{t-1}, x_t] + b_i)     ← Lọc thông tin        │   │
│  │  C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)  ← Thông tin mới        │   │
│  │                                                                  │   │
│  │  • i_t quyết định bao nhiêu của C̃_t được thêm vào              │   │
│  │                                                                  │   │
│  │  Ví dụ Traffic: Khi có mưa lớn đột ngột,                        │   │
│  │  input gate thêm thông tin "mưa ảnh hưởng traffic"              │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  CELL STATE UPDATE: Cập nhật bộ nhớ dài hạn                    │   │
│  │                                                                  │   │
│  │  C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t                               │   │
│  │        ↑                   ↑                                     │   │
│  │        Quên bớt           Thêm mới                               │   │
│  │                                                                  │   │
│  │  Cell state là "highway" cho gradient flow                       │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  OUTPUT GATE (Cổng đầu ra): Quyết định output gì                │   │
│  │                                                                  │   │
│  │  o_t = σ(W_o · [h_{t-1}, x_t] + b_o)                            │   │
│  │  h_t = o_t ⊙ tanh(C_t)                                          │   │
│  │                                                                  │   │
│  │  • Lọc thông tin từ cell state thành hidden state               │   │
│  │  • h_t được dùng cho prediction và truyền sang timestep tiếp    │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Output:                                                                │
│  • h_t: hidden state (cho layer tiếp theo hoặc output)                 │
│  • C_t: cell state (truyền sang timestep tiếp theo)                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

##### B. Bidirectional LSTM

Encoder sử dụng **Bidirectional LSTM** - xử lý sequence theo 2 hướng:

```
Input Sequence: [x_1, x_2, x_3, ..., x_24]

Forward LSTM (→):
   x_1 ──► h_1^f ──► h_2^f ──► h_3^f ──► ... ──► h_24^f
           │         │         │                  │
   Học patterns từ quá khứ đến hiện tại

Backward LSTM (←):
   x_1 ◄── h_1^b ◄── h_2^b ◄── h_3^b ◄── ... ◄── h_24^b
           │         │         │                  │
   Học patterns từ tương lai về quá khứ

Combined output tại mỗi timestep t:
   h_t = [h_t^f ; h_t^b]  (concatenate)
   
   → Mỗi h_t chứa thông tin từ CẢ 2 HƯỚNG
```

**Tại sao cần Bidirectional?**

```
Ví dụ: Traffic tại 8:00 (giờ cao điểm sáng)

Forward only:  Chỉ thấy 7:00, 6:00, 5:00... → biết đang đến rush hour
Backward only: Chỉ thấy 9:00, 10:00... → biết rush hour sắp kết thúc
Bidirectional: Thấy CẢ HAI → hiểu đầy đủ context của rush hour
```

##### C. Code Encoder

```python
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, 
                 dropout=0.2, bidirectional=True):
        super(Encoder, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,      # 22 features
            hidden_size=hidden_size,    # 64
            num_layers=num_layers,      # 2 stacked layers
            batch_first=True,           # (batch, seq, features)
            dropout=dropout,            # Dropout giữa layers
            bidirectional=bidirectional
        )
        
        # Projection layers: 64*2 → 64 (để match với Decoder)
        if bidirectional:
            self.fc_hidden = nn.Linear(hidden_size * 2, hidden_size)
            self.fc_cell = nn.Linear(hidden_size * 2, hidden_size)
    
    def forward(self, x):
        """
        x: (batch, 24, 22) - 24 timesteps, 22 features
        
        Returns:
        - outputs: (batch, 24, 64*2) - all hidden states
        - (hidden, cell): final states, each (2, batch, 64)
        """
        # LSTM forward pass
        outputs, (hidden, cell) = self.lstm(x)
        # outputs: (batch, 24, 128) nếu bidirectional
        # hidden: (4, batch, 64) = 2 layers × 2 directions
        
        if self.bidirectional:
            # Reshape: (num_layers * 2, batch, hidden) → (num_layers, batch, hidden * 2)
            batch_size = hidden.shape[1]
            
            # Tách forward và backward cho mỗi layer
            hidden = hidden.view(self.num_layers, 2, batch_size, self.hidden_size)
            cell = cell.view(self.num_layers, 2, batch_size, self.hidden_size)
            
            # Concatenate forward và backward
            hidden = torch.cat([hidden[:, 0, :, :], hidden[:, 1, :, :]], dim=2)
            cell = torch.cat([cell[:, 0, :, :], cell[:, 1, :, :]], dim=2)
            
            # Project 128 → 64
            hidden = self.fc_hidden(hidden)  # (2, batch, 64)
            cell = self.fc_cell(cell)        # (2, batch, 64)
        
        return outputs, (hidden, cell)
```

#### 1.3.4 DECODER - Chi tiết hoạt động

##### A. Cách Decoder sinh output

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    DECODER STEP-BY-STEP                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Nhận từ Encoder:                                                       │
│  • hidden_0 = encoder_hidden (context về toàn bộ input)                │
│  • cell_0 = encoder_cell                                               │
│                                                                         │
│  STEP t=0 (predict t+1):                                               │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  input_0 = last_traffic_volume từ input sequence                 │  │
│  │           (giá trị traffic cuối cùng mà model "biết")            │  │
│  │                          ↓                                        │  │
│  │  ┌──────────────────────────────────────────────────────────┐    │  │
│  │  │         LSTM Cell                                        │    │  │
│  │  │  input: input_0, hidden_0, cell_0                        │    │  │
│  │  │  output: hidden_1, cell_1                                │    │  │
│  │  └──────────────────────────────────────────────────────────┘    │  │
│  │                          ↓                                        │  │
│  │  ┌──────────────────────────────────────────────────────────┐    │  │
│  │  │  Fully Connected: hidden_1 → prediction_0                │    │  │
│  │  │  prediction_0 = traffic_volume tại t+1                   │    │  │
│  │  └──────────────────────────────────────────────────────────┘    │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  STEP t=1 (predict t+2):                                               │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  if Teacher Forcing (random < 0.65):                              │  │
│  │      input_1 = ground_truth[t+1]  ← Dùng đáp án đúng             │  │
│  │  else:                                                            │  │
│  │      input_1 = prediction_0       ← Dùng prediction vừa sinh     │  │
│  │                          ↓                                        │  │
│  │  LSTM Cell(input_1, hidden_1, cell_1) → hidden_2, cell_2          │  │
│  │                          ↓                                        │  │
│  │  FC(hidden_2) → prediction_1 = traffic_volume tại t+2            │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ... Lặp tương tự cho t=2,3,4 ...                                      │
│                                                                         │
│  Final Output: [prediction_0, prediction_1, ..., prediction_4]         │
│               = [traffic_t+1, traffic_t+2, ..., traffic_t+5]           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

##### B. Teacher Forcing - Kỹ thuật huấn luyện

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    TEACHER FORCING EXPLAINED                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  VẤN ĐỀ: Autoregressive Decoding                                       │
│  ────────────────────────────────                                       │
│  • Decoder dùng output của step trước làm input cho step tiếp          │
│  • Nếu step đầu sai → error lan truyền (error accumulation)            │
│                                                                         │
│  Không có Teacher Forcing:                                              │
│  ─────────────────────────                                              │
│  True:    [1000, 1100, 1200, 1150, 1050]                               │
│  Step 0:  Predict 1000 → OK                                            │
│  Step 1:  Input=1000, Predict 1080 (sai 20)                            │
│  Step 2:  Input=1080 (đã sai), Predict 1250 (sai 50)                   │
│  Step 3:  Input=1250 (sai hơn), Predict 1400 (sai 250!)                │
│  → Error tích lũy ngày càng lớn!                                       │
│                                                                         │
│  VỚI Teacher Forcing (ratio = 0.65):                                   │
│  ───────────────────────────────────                                   │
│  True:    [1000, 1100, 1200, 1150, 1050]                               │
│  Step 0:  Predict 1000                                                 │
│  Step 1:  Random < 0.65 → Input = TRUE 1100, Predict 1095              │
│           (Model học từ đáp án đúng, không bị ảnh hưởng bởi error)     │
│  Step 2:  Random > 0.65 → Input = predicted 1095, Predict 1185         │
│           (Model cũng học xử lý khi input không hoàn hảo)              │
│                                                                         │
│  → Cân bằng giữa:                                                       │
│    • Học nhanh (dùng ground truth)                                     │
│    • Robust (quen với imperfect input)                                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Decay Teacher Forcing Ratio:**

```python
# Trong training, giảm dần teacher forcing theo epoch
current_tf_ratio = teacher_forcing_ratio * (1 - epoch / num_epochs)

# Epoch 1:  tf_ratio = 0.65 * (1 - 0/100) = 0.65 (dùng nhiều ground truth)
# Epoch 50: tf_ratio = 0.65 * (1 - 50/100) = 0.325
# Epoch 99: tf_ratio = 0.65 * (1 - 99/100) ≈ 0.007 (gần như autoregressive)

# → Model dần quen với việc dùng chính predictions của mình
```

##### C. Code Decoder

```python
class Decoder(nn.Module):
    def __init__(self, output_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        super(Decoder, self).__init__()
        
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM: input là 1 giá trị (previous traffic_volume)
        self.lstm = nn.LSTM(
            input_size=output_size,     # 1
            hidden_size=hidden_size,    # 64
            num_layers=num_layers,      # 2
            batch_first=True,
            dropout=dropout
        )
        
        # Fully Connected: hidden → 1 output
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, hidden, cell):
        """
        x: (batch, 1, 1) - previous output/ground truth
        hidden: (2, batch, 64) - từ encoder hoặc step trước
        cell: (2, batch, 64)
        
        Returns:
        - prediction: (batch, 1, 1)
        - (hidden, cell): updated states
        """
        output, (hidden, cell) = self.lstm(x, (hidden, cell))
        # output: (batch, 1, 64)
        
        prediction = self.fc(output)  # (batch, 1, 1)
        
        return prediction, (hidden, cell)
```

#### 1.3.5 Seq2Seq - Kết hợp Encoder và Decoder

```python
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, output_seq_len=5, device=None):
        super(Seq2Seq, self).__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.output_seq_len = output_seq_len  # 5 steps
        self.device = device
    
    def forward(self, x, target=None, teacher_forcing_ratio=0.5):
        """
        x: (batch, 24, 22) - input sequence
        target: (batch, 5) - ground truth (for teacher forcing)
        teacher_forcing_ratio: probability of using ground truth
        
        Returns: (batch, 5) - 5 predictions
        """
        batch_size = x.shape[0]
        
        # 1. ENCODE: Nén 24 timesteps thành context vector
        _, (hidden, cell) = self.encoder(x)
        # hidden, cell: (2, batch, 64) - context từ toàn bộ input
        
        # 2. DECODE: Sinh 5 outputs tuần tự
        
        # Initial input: last traffic_volume từ input sequence
        # x[:, -1, 0] = traffic_volume tại timestep cuối cùng
        decoder_input = x[:, -1, 0].unsqueeze(1).unsqueeze(2)  # (batch, 1, 1)
        
        outputs = []
        
        for t in range(self.output_seq_len):  # 5 steps
            # Decode 1 step
            prediction, (hidden, cell) = self.decoder(decoder_input, hidden, cell)
            outputs.append(prediction.squeeze(2))  # (batch, 1)
            
            # Chọn input cho step tiếp theo
            if target is not None and torch.rand(1).item() < teacher_forcing_ratio:
                # Teacher Forcing: dùng ground truth
                decoder_input = target[:, t].unsqueeze(1).unsqueeze(2)
            else:
                # Autoregressive: dùng prediction vừa sinh
                decoder_input = prediction
        
        # Concatenate: [(batch, 1), ...] → (batch, 5)
        outputs = torch.cat(outputs, dim=1)
        
        return outputs
```

#### 1.3.6 Flow hoàn chỉnh một Forward Pass

```
┌─────────────────────────────────────────────────────────────────────────┐
│                 COMPLETE FORWARD PASS EXAMPLE                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  INPUT: batch_size=32, seq_len=24, features=22                         │
│  X shape: (32, 24, 22)                                                 │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│  ENCODER                                                                │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
│  1. Bidirectional LSTM:                                                │
│     X (32, 24, 22) ──► LSTM ──► outputs (32, 24, 128)                  │
│                              └─► hidden (4, 32, 64)                    │
│                              └─► cell (4, 32, 64)                      │
│                                                                         │
│  2. Reshape & Project:                                                 │
│     hidden (4, 32, 64) ──► (2, 32, 128) ──► fc ──► (2, 32, 64)        │
│     cell   (4, 32, 64) ──► (2, 32, 128) ──► fc ──► (2, 32, 64)        │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│  DECODER (5 iterations)                                                 │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
│  Initial: decoder_input = X[:, -1, 0] = (32, 1, 1)                     │
│                                                                         │
│  t=0: input(32,1,1) + hidden(2,32,64) ──► LSTM ──► hidden'(2,32,64)   │
│       └─► FC ──► pred_0 (32, 1)                                        │
│                                                                         │
│  t=1: input(32,1,1) + hidden'(2,32,64) ──► LSTM ──► hidden''          │
│       └─► FC ──► pred_1 (32, 1)                                        │
│                                                                         │
│  ... (repeat for t=2, 3, 4) ...                                        │
│                                                                         │
│  3. Concatenate:                                                        │
│     [pred_0, pred_1, pred_2, pred_3, pred_4] ──► (32, 5)               │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│  OUTPUT: (32, 5) = 32 samples × 5 predictions                          │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

### 1.4 OPTUNA HYPERPARAMETER OPTIMIZATION

#### 1.4.1 Optuna là gì?

**Optuna** là một framework tự động tối ưu hóa hyperparameters (AutoML) với các đặc điểm:

1. **Define-by-Run API:** Định nghĩa search space linh hoạt trong code
2. **Efficient Sampling:** Sử dụng thuật toán TPE (Tree-structured Parzen Estimator)
3. **Pruning:** Dừng sớm các trials không hứa hẹn để tiết kiệm thời gian
4. **Visualization:** Cung cấp các công cụ visualize kết quả

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    OPTUNA OPTIMIZATION FLOW                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────┐                                                       │
│  │ Search Space │  Định nghĩa khoảng giá trị cho mỗi hyperparameter    │
│  └──────┬───────┘                                                       │
│         │                                                               │
│         ▼                                                               │
│  ┌──────────────┐                                                       │
│  │   Sampler    │  TPE chọn giá trị hyperparameters thông minh         │
│  │    (TPE)     │  dựa trên kết quả các trials trước                    │
│  └──────┬───────┘                                                       │
│         │                                                               │
│         ▼                                                               │
│  ┌──────────────┐                                                       │
│  │    Trial     │  Huấn luyện model với hyperparameters được chọn      │
│  │  (Training)  │                                                       │
│  └──────┬───────┘                                                       │
│         │                                                               │
│         ▼                                                               │
│  ┌──────────────┐                                                       │
│  │   Pruner     │  Dừng sớm nếu trial không tốt (MedianPruner)         │
│  └──────┬───────┘                                                       │
│         │                                                               │
│         ▼                                                               │
│  ┌──────────────┐                                                       │
│  │  Objective   │  Trả về validation loss để Optuna đánh giá           │
│  │    Value     │                                                       │
│  └──────┬───────┘                                                       │
│         │                                                               │
│         └──────────► Lặp lại cho n_trials                              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

#### 1.4.2 Các Hyperparameters được tối ưu

Trong project này, Optuna tối ưu **7 hyperparameters**:

| Hyperparameter | Search Space | Ý nghĩa | Tác dụng |
|----------------|--------------|---------|----------|
| **hidden_size** | [64, 128, 256] | Số neurons trong hidden layer của LSTM | Càng lớn → model càng phức tạp, học được patterns phức tạp hơn, nhưng dễ overfit |
| **num_layers** | [1, 2, 3] | Số LSTM layers xếp chồng | Nhiều layers → học hierarchical features, nhưng khó train hơn |
| **dropout** | (0.1, 0.5) | Tỷ lệ dropout giữa các layers | Regularization, giảm overfitting |
| **learning_rate** | (1e-4, 1e-2) | Tốc độ học của optimizer | Quá cao → không hội tụ, quá thấp → train chậm |
| **batch_size** | [32, 64, 128] | Số samples trong 1 batch | Ảnh hưởng đến gradient estimation và memory |
| **weight_decay** | (1e-6, 1e-3) | L2 regularization strength | Giảm overfitting bằng cách penalize large weights |
| **teacher_forcing_ratio** | (0.3, 0.7) | Tỷ lệ dùng ground truth khi train decoder | Cân bằng giữa học nhanh và exposure bias |

#### 1.4.3 Chi tiết từng Hyperparameter

##### A. Hidden Size (64, 128, 256)

```
hidden_size = 64:        hidden_size = 256:
┌────────────────┐       ┌────────────────────────────────────┐
│  64 neurons    │       │         256 neurons                │
│  Ít parameters │       │         Nhiều parameters           │
│  Nhanh train   │       │         Chậm train                 │
│  Có thể underfitｾ      │         Có thể overfit             │
└────────────────┘       └────────────────────────────────────┘
```

**Tác dụng:**
- Quyết định **capacity** (khả năng học) của model
- hidden_size = 64 được chọn: Đủ để capture patterns mà không overfit

##### B. Number of Layers (1, 2, 3)

```
1 Layer:           2 Layers:              3 Layers:
┌─────┐           ┌─────┐                ┌─────┐
│LSTM │           │LSTM │ ← Layer 2      │LSTM │ ← Layer 3
└─────┘           └─────┘                └─────┘
                  ┌─────┐                ┌─────┐
                  │LSTM │ ← Layer 1      │LSTM │ ← Layer 2
                  └─────┘                └─────┘
                                         ┌─────┐
                                         │LSTM │ ← Layer 1
                                         └─────┘
```

**Tác dụng:**
- Nhiều layers = học **hierarchical representations**
- Layer 1: Low-level patterns (trends ngắn hạn)
- Layer 2+: High-level patterns (trends dài hạn)
- num_layers = 2 được chọn: Balance giữa capacity và training difficulty

##### C. Dropout (0.1 - 0.5)

```python
# Dropout hoạt động trong training:
# Randomly "tắt" một số neurons với xác suất p

Input:  [0.5, 0.3, 0.8, 0.2, 0.6]
              ↓ dropout=0.3
Output: [0.5, 0.0, 0.8, 0.0, 0.6]  # 30% neurons bị tắt
              ↓ scale by 1/(1-p)
Output: [0.71, 0.0, 1.14, 0.0, 0.86]  # Scale để giữ expected value
```

**Tác dụng:**
- **Regularization:** Giảm overfitting bằng cách buộc model không phụ thuộc vào bất kỳ neuron nào
- **Ensemble effect:** Như train nhiều sub-networks rồi average
- dropout ≈ 0.25 được chọn: Vừa đủ regularization

##### D. Learning Rate (1e-4 - 1e-2)

```
Learning Rate quá cao (0.01):     Learning Rate quá thấp (0.0001):
        Loss                              Loss
          │                                 │
          │  ╱╲  ╱╲                         │ ────────────────
          │ ╱  ╲╱  ╲                        │          ╲
          │╱        ╲                       │           ╲
          └──────────►                      └──────────────►
            Epochs                              Epochs
      Oscillating, không hội tụ           Hội tụ chậm, tốn thời gian

Learning Rate phù hợp (~0.001):
        Loss
          │
          │╲
          │ ╲
          │  ╲────────
          └──────────────►
              Epochs
      Hội tụ nhanh và ổn định
```

**Tác dụng:**
- Quyết định **step size** khi update weights
- lr ≈ 0.0015 được chọn: Hội tụ nhanh mà ổn định

##### E. Batch Size (32, 64, 128)

| Batch Size | Ưu điểm | Nhược điểm |
|------------|---------|------------|
| 32 (nhỏ) | Gradient noisy → better generalization | Chậm (nhiều updates/epoch) |
| 128 (lớn) | Nhanh, stable gradients | Có thể converge đến sharp minima |
| 64 (vừa) | Balance giữa speed và generalization | - |

**Tác dụng:**
- batch_size = 64 được chọn: Cân bằng tốc độ và chất lượng

##### F. Weight Decay (1e-6 - 1e-3)

```python
# L2 Regularization: Thêm penalty vào loss
# Loss_total = Loss_original + weight_decay * Σ(w²)

# weight_decay = 0.0004:
# Penalize các weights lớn → model đơn giản hơn → giảm overfit
```

**Tác dụng:**
- **L2 regularization** trong optimizer
- Giữ weights nhỏ → model ít overfit

##### G. Teacher Forcing Ratio (0.3 - 0.7)

```
Teacher Forcing Ratio = 0.5:

Training Step t=1:
  Input: last_known_value
  Output: prediction_1
  
Training Step t=2:
  if random() < 0.5:   # 50% chance
      Input = ground_truth[1]     # Teacher Forcing: dùng đáp án đúng
  else:
      Input = prediction_1        # Autoregressive: dùng prediction
  Output: prediction_2
```

**Tác dụng:**
- **Teacher Forcing = 1.0:** Luôn dùng ground truth → train nhanh nhưng **exposure bias** (inference khác training)
- **Teacher Forcing = 0.0:** Luôn dùng prediction → train chậm, dễ error accumulation
- **Teacher Forcing ≈ 0.65:** Cân bằng giữa train nhanh và giảm exposure bias

#### 1.4.4 Kết quả Optuna trong Project

**Best Hyperparameters tìm được:**

```json
{
    "hidden_size": 64,
    "num_layers": 2,
    "dropout": 0.248,
    "learning_rate": 0.00155,
    "batch_size": 64,
    "weight_decay": 0.000379,
    "teacher_forcing_ratio": 0.648
}
```

**Phân tích kết quả:**

| Hyperparameter | Giá trị | Nhận xét |
|----------------|---------|----------|
| hidden_size = 64 | Nhỏ nhất | Data không quá phức tạp, 64 đủ capacity |
| num_layers = 2 | Trung bình | Cần hierarchy nhưng không quá sâu |
| dropout ≈ 0.25 | Trung bình | Regularization vừa phải |
| lr ≈ 0.0015 | Cao hơn default | Model có thể học nhanh |
| batch_size = 64 | Trung bình | Balance speed và generalization |
| weight_decay ≈ 0.0004 | Nhỏ | Không cần quá nhiều L2 reg |
| teacher_forcing ≈ 0.65 | Cao | Ưu tiên learning speed |

#### 1.4.5 Code Optuna trong Project

```python
class Seq2SeqObjective:
    """Objective function cho Optuna optimization"""
    
    def __call__(self, trial: optuna.Trial) -> float:
        # 1. Sample hyperparameters từ search space
        hidden_size = trial.suggest_categorical('hidden_size', [64, 128, 256])
        num_layers = trial.suggest_categorical('num_layers', [1, 2, 3])
        dropout = trial.suggest_float('dropout', 0.1, 0.5)
        lr = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
        teacher_forcing = trial.suggest_float('teacher_forcing_ratio', 0.3, 0.7)
        
        # 2. Build model với hyperparameters được chọn
        model = build_model(
            input_size=self.input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            ...
        )
        
        # 3. Training loop với early stopping
        for epoch in range(self.n_epochs):
            train_loss = train_one_epoch(...)
            val_loss = validate(...)
            
            # 4. Report để Optuna có thể prune
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        # 5. Return validation loss (minimize)
        return best_val_loss


# Run optimization
study = optuna.create_study(
    direction="minimize",  # Minimize validation loss
    pruner=optuna.pruners.MedianPruner()  # Prune trials tệ hơn median
)
study.optimize(objective, n_trials=50, timeout=3600)
```

#### 1.4.6 Pruning - Dừng sớm trials không tốt

```
Trial 1: val_loss = [0.5, 0.4, 0.3, 0.25, 0.22] → Complete ✓
Trial 2: val_loss = [0.6, 0.55, 0.52, ...     ] → Pruned ✗ (worse than median)
Trial 3: val_loss = [0.45, 0.35, 0.28, 0.23, 0.20] → Complete ✓ (Best!)
Trial 4: val_loss = [0.7, 0.65, ...           ] → Pruned ✗
...

MedianPruner: Nếu val_loss tại epoch t > median của các trials khác tại epoch t → Prune
```

**Tác dụng:**
- Tiết kiệm thời gian bằng cách dừng sớm các trials không hứa hẹn
- Tập trung resources vào các trials có tiềm năng

---

### 1.5 HIỂU PHƯƠNG PHÁP ĐÁNH GIÁ CHẤT LƯỢNG CÁC MODEL

#### 1.5.1 Các Metrics được sử dụng

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

#### 1.5.2 Đánh giá Per-Step (Multi-Step Forecasting)

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

#### 1.5.3 Kết quả thực tế của Model

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

#### 1.5.4 Inverse Transform trước khi đánh giá

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
