# CHANGELOG - lstm.py 对齐 itransformer_model.py

## 修改日期
2026-03-09

## 修改概述
本次修改为 LSTM 添加高级特征集，替代原始 OHLCV 数据：
1. **高级特征**：使用价格动量、波动率、成交量等工程特征
2. **横截面特征**：添加排名特征（rank within each time step）
3. **外部数据**：整合 trades, financial, options 数据

### 详细修改

#### 1. 新增 load_all_data_for_lstm() 函数

**文件位置**: lines 378-417

**功能**: 加载完整的特征数据集，包括：
- 价格动量特征 (7个): stk_ret_1/2/3/5/10/20, stk_logret_1
- 日内价格结构特征 (4个): stk_hl_spread, stk_oc_spread, stk_close_to_ma_5/20
- 波动率特征 (3个): stk_vol_5/10/20
- 成交量特征 (4个): stk_volume_chg_1, stk_volume_to_ma_5/10/20
- 横截面特征 (3个): stk_ret_1_rank, stk_vol_20_rank, stk_volume_chg_1_rank
- 期权特征 (1个): iv_avg
- 交易流向特征 (4个): trd_individual/foreigners/securitiescos/investmenttrusts
- 财务特征 (3个): stk_mcap, stk_sector, stk_market_segment

#### 2. 新增 add_cross_sectional_features() 函数

**文件位置**: lines 419-435

**功能**: 添加横截面特征
- 在每个时间点（Date），计算每只股票的排名
- 捕捉市场相对位置信息

#### 3. 新增 LSTM_FEATURES 常量

**文件位置**: lines 438-467

**定义**: 高级特征列表

#### 4. 修改 load_dataset() 函数

**文件位置**: lines 469-495

**修改**: 使用高级特征替代原始 OHLCV

**对比**:
| 数据类型 | 特征数 | 特征示例 |
|---------|-------|---------|
| 原始 OHLCV | 5 | Open, High, Low, Close, Volume |
| **高级特征** | 26+ | 动量、波动率、成交量、横截面排名等 |

---

## 修改日期
2026-03-09

## 修改概述
本次修改将 LSTM 的标准化方式改为 Cross-sectional Normalization：
1. **标准化方式**：从全局标准化改为横截面标准化
2. **新函数**：添加 cross_sectional_normalize() 函数
3. **效果**：移除市场整体影响，让模型专注于学习股票间的相对差异

### 详细修改

#### 1. 添加 cross_sectional_normalize() 函数

**文件位置**: lines 472-530

**新增函数**:
```python
def cross_sectional_normalize(df, feature_cols, eps=1e-8):
    """
    Cross-sectional normalization: normalize across stocks at each time step

    For each time step (Date), normalize features across all stocks.
    This removes market-wide effects and focuses on relative differences between stocks.
    """
    log("Applying cross-sectional normalization across stocks at each time step...")

    df = df.sort_values(["Date", "SecuritiesCode"]).reset_index(drop=True)

    # For each time step, normalize across all stocks
    for col in tqdm(feature_cols, desc="Cross-sectional normalizing"):
        # Group by date and compute mean/std across all stocks for each date
        date_stats = df.groupby("Date")[col].agg(['mean', 'std']).reset_index()
        date_stats.columns = ['Date', f'{col}_mean', f'{col}_std']

        # Merge stats back to original dataframe
        df = df.merge(date_stats, on='Date', how='left')

        # Fill missing std with 1 to avoid division by zero
        df[f'{col}_std'] = df[f'{col}_std'].fillna(1.0)
        df[f'{col}_std'] = df[f'{col}_std'].replace(0, 1.0)
        df[f'{col}_std'] = df[f'{col}_std'].clip(lower=eps)

        # Normalize: (x - mean) / std
        df[col] = (df[col] - df[f'{col}_mean']) / df[f'{col}_std']
        df[col] = df[col].clip(-10, 10)

        # Drop temporary columns
        df = df.drop(columns=[f'{col}_mean', f'{col}_std'])

    log("Cross-sectional normalization completed.")
    return df
```

**设计理念**:
- 在每个时间点，对所有股票的特征进行 z-score 归一化
- 移除市场整体影响（大盘涨跌）
- 让模型专注于学习股票间的相对差异
- 与 iTransformer 的横截面归一化方法一致

#### 2. 修改 predict_with_lstm 函数

**文件位置**: lines 790-802

**修改前**:
```python
# Step 1: Normalize using full training period (2017-2020) stats
log("Step 1: Computing normalization statistics...")
train_stats_df = df[(df["Year"] >= 2017) & (df["Year"] < 2021)].copy()
train_stats = {}
for col in feature_cols:
    train_stats[col] = {
        'mean': train_stats_df[col].mean(),
        'std': train_stats_df[col].std()
    }

# Normalize ALL data using these stats
log("Step 2: Normalizing all data...")
df_norm = df.copy()
for col in feature_cols:
    # ... global normalization
```

**修改后**:
```python
# Step 1: Apply cross-sectional normalization across stocks at each time step
# This removes market-wide effects and focuses on relative differences between stocks
log("Step 1: Applying cross-sectional normalization (normalize across stocks at each time step)...")
df_norm = cross_sectional_normalize(df, feature_cols)

# Step 2: Create sequences from ALL normalized data
log("Step 2: Creating sequences from all data...")
```

**对比**:
| 标准化方式 | 作用 | 适用场景 |
|-----------|------|---------|
| 全局标准化 | 对所有数据统一归一化 | 保留市场整体信息 |
| **横截面标准化** | 每个时间点对所有股票归一化 | 移除市场影响，关注相对差异 |

---

## 修改日期
2026-03-08

## 修改概述
本次修改将 lstm.py 的输入数据改为原始 OHLCV 数据：
1. **数据格式**：使用原始 5 个 OHLCV 特征，而非 41 个工程特征
2. **设计理念**：LSTM 应从原始数据学习高维特征（returns, trends, volatility），而非使用人工计算的工程特征
3. **接口保留**：保持 load_dataset() 函数接口，与 transformer.py 一致

### 详细修改

#### 1. load_dataset() 函数 - 使用原始 OHLCV 数据

**文件位置**: lines 378-399

**修改后**:
```python
def load_dataset():
    """
    Load raw OHLCV data for LSTM.
    LSTM will learn returns, trends, volatility and other high-dimensional features from raw data.
    This is different from tree-based models which use engineered features.
    """
    log("Loading raw OHLCV data...")

    # Load stock prices
    stock_prices = pd.read_csv("train_files/stock_prices.csv")
    stock_prices = to_num(stock_prices, ["Open", "High", "Low", "Close", "Volume"])
    stock_prices["Date"] = pd.to_datetime(stock_prices["Date"])

    # Keep only needed columns
    df = stock_prices[["Date", "SecuritiesCode", "Open", "High", "Low", "Close", "Volume"]].copy()

    # Sort by stock and date
    df = df.sort_values(["SecuritiesCode", "Date"]).reset_index(drop=True)

    # Fill missing values: forward fill then backward fill per stock
    for col in RAW_FEATURES:
        df[col] = df.groupby("SecuritiesCode")[col].ffill()
        df[col] = df.groupby("SecuritiesCode")[col].bfill()
        df[col] = df[col].fillna(0)

    # Build 30-day forward return labels
    df = build_30d_labels_raw(df)

    log(f"Loaded: {len(df)} rows of raw OHLCV data")
    log(f"Features: {RAW_FEATURES}")
    log(f"Target: target_30d")

    return df, RAW_FEATURES, "target_30d"
```

**设计理念**:
- LSTM 作为深度学习模型，能够从原始 OHLCV 数据中自动学习高维特征
- 模型可以学习：收益率、趋势、波动率、成交量模式等技术指标
- 不需要人工计算特征，与树状模型（如 LightGBM）的设计理念不同
- 树状模型更适合使用人工工程特征

**对比**:
| 模型类型 | 特征来源 | 原因 |
|---------|---------|------|
| LSTM/RNN | 原始 OHLCV (5维) | 自动学习高维表示 |
| LightGBM/XGBoost | 工程特征 (41维) | 人工特征更适合树状结构 |

#### 2. main() 函数保持不变

**保持调用方式**:
```python
# Load dataset using load_dataset() - consistent with transformer.py
log("Loading dataset with engineered features...")
data, feature_cols, target_col = load_dataset()

# Run LSTM prediction
pred = predict_with_lstm(data, feature_cols, target_col)
```

---

## 修改日期
2026-03-08

## 修改概述
本次修改将 lstm.py 与 transformer.py 对齐，使 load_dataset() 函数接口一致：
1. **接口统一**：重写 load_dataset() 函数，返回 (data, feature_cols, target_col) 三元组
2. **main() 简化**：使用 load_dataset() 替代直接调用 load_all_data()

### 详细修改

#### 1. load_dataset() 函数重写

**文件位置**: lines 378-387

**修改前**:
```python
def load_dataset():
    """
    Load raw OHLCV data for LSTM.
    Instead of manual feature engineering, LSTM will learn features from raw data.
    """
    log("Loading raw OHLCV data...")

    # Load stock prices
    stock_prices = pd.read_csv("train_files/stock_prices.csv")
    stock_prices = to_num(stock_prices, ["Open", "High", "Low", "Close", "Volume"])
    stock_prices["Date"] = pd.to_datetime(stock_prices["Date"])

    # Keep only needed columns
    df = stock_prices[["Date", "SecuritiesCode", "Open", "High", "Low", "Close", "Volume"]].copy()

    # Sort by stock and date
    df = df.sort_values(["SecuritiesCode", "Date"]).reset_index(drop=True)

    # Fill missing values: forward fill then backward fill per stock
    for col in RAW_FEATURES:
        df[col] = df.groupby("SecuritiesCode")[col].ffill()
        df[col] = df.groupby("SecuritiesCode")[col].bfill()
        df[col] = df[col].fillna(0)

    # Build 30-day forward return labels
    df = build_30d_labels_raw(df)

    log(f"Loaded: {len(df)} rows of raw OHLCV data")
    log(f"Features: {RAW_FEATURES}")
    log(f"Target: target_30d")

    return df, RAW_FEATURES, "target_30d"
```

**修改后**:
```python
def load_dataset():
    """
    Load dataset - same as transformer.py.
    Returns clean interface: (data, feature_cols, target_col)
    """
    full_df, feature_cols = load_all_data()
    target_col = "target_30d"
    data = full_df[["Date", "SecuritiesCode"] + feature_cols + [target_col]].copy()
    log(f"Features: {len(feature_cols)}")
    log(f"Target: {target_col}")
    return data, feature_cols, target_col
```

**修改原因**:
- 原来的 `load_dataset()` 加载原始 OHLCV 数据（5个特征）
- 与 transformer.py 的接口不一致
- 新版本使用 `load_all_data()` 生成 41 个工程特征，接口与 transformer.py 一致

#### 2. main() 函数修改

**文件位置**: lines 1021-1035

**修改前**:
```python
# Load data with all features
log("Loading all data sources with engineered features...")
full_df, all_feature_cols = load_all_data()

# Filter to use only the specified features
available_features = [f for f in USE_FEATURES if f in full_df.columns]
missing_features = [f for f in USE_FEATURES if f not in full_df.columns]

if missing_features:
    log(f"Warning: {len(missing_features)} features not found: {missing_features[:5]}...")

log(f"Using {len(available_features)} features out of {len(USE_FEATURES)} requested")

# Run LSTM prediction
pred = predict_with_lstm(full_df, available_features, "target_30d")
```

**修改后**:
```python
# Load dataset using load_dataset() - consistent with transformer.py
log("Loading dataset with engineered features...")
data, feature_cols, target_col = load_dataset()

# Run LSTM prediction
pred = predict_with_lstm(data, feature_cols, target_col)
```

**修改原因**:
- 直接使用 `load_dataset()` 替代复杂的 `load_all_data()` + 特征过滤逻辑
- 与 transformer.py 的调用方式一致
- 代码更简洁、更易维护

### 对比 transformer.py

| 函数 | transformer.py | lstm.py (修改前) | lstm.py (修改后) |
|------|----------------|-----------------|-----------------|
| load_dataset() | ✅ 返回三元组 | ❌ 加载原始数据 | ✅ 返回三元组 |
| main() 调用 | load_dataset() | load_all_data() | load_dataset() |

---

## 修改日期
2026-03-07

## 修改概述
本次修改将 lstm.py 与 itransformer_model.py 对齐，主要包括：
1. **特征扩展**：从 5 个原始 OHLCV 特征扩展到 41 个工程特征
2. **训练策略完善**：从 2 轮训练扩展到 4 轮训练（expanding window）
3. **评价标准修改**：从月度换手改为每日换手
4. **Sharpe 年化修正**：从 sqrt(4) 修正为 sqrt(252)
5. **文档更新**：更新顶部文档说明以反映新方法

---

## 详细修改列表

### 1. 特征扩展（5 → 41 个特征）

**文件位置**: lines 38-90

**修改前**:
```python
# Raw OHLCV columns for LSTM input
RAW_FEATURES = ["Open", "High", "Low", "Close", "Volume"]
```

**修改后**:
```python
# Feature columns for LSTM input - aligned with itransformer_model.py (41 features)
USE_FEATURES = [
    # 收益率特征（7个）
    "stk_ret_1", "stk_ret_2", "stk_ret_3", "stk_ret_5", "stk_ret_10", "stk_ret_20",
    "stk_logret_1",

    # 波动率特征（3个）
    "stk_vol_5", "stk_vol_10", "stk_vol_20",

    # 价差特征（2个）
    "stk_hl_spread", "stk_oc_spread",

    # 成交量特征（4个）
    "stk_volume_chg_1",
    "stk_volume_to_ma_5", "stk_volume_to_ma_10", "stk_volume_to_ma_20",

    # 滚动均值收益（3个）
    "stk_ret_mean_5", "stk_ret_mean_10", "stk_ret_mean_20",

    # 移动平均比率（3个）
    "stk_close_to_ma_5", "stk_close_to_ma_10", "stk_close_to_ma_20",

    # 分布特征（1个）
    "stk_skew_20",

    # 时间特征（2个）
    "stk_dayofweek", "stk_month",

    # 公司行为（2个）
    "stk_expected_dividend", "stk_supervision_flag",

    # 基本面特征（3个）
    "stk_mcap", "stk_sector", "stk_market_segment",

    # 期权特征（1个）
    "iv_avg",

    # 交易特征（4个）
    "trd_individual", "trd_foreigners", "trd_securitiescos", "trd_investmenttrusts",

    # 财务特征（6个）
    "fin_netsales", "fin_operatingprofit", "fin_ordinaryprofit",
    "fin_profit", "fin_totalassets", "fin_equity",
]

# Legacy OHLCV features (no longer used)
RAW_FEATURES = ["Open", "High", "Low", "Close", "Volume"]
```

**修改原因**:
- 原始 OHLCV 特征信息量有限，只有 5 个特征
- 工程特征包含更多市场信息：动量、波动率、成交量、基本面、期权、交易流向等
- 与 itransformer_model.py 保持一致，使用相同的 41 个特征

**影响**:
- LSTM 输入维度：从 (batch, 60, 5) 变为 (batch, 60, 41)
- 模型参数量增加：input_size 从 5 增加到 41
- 预期模型性能提升（更多信息输入）

---

### 2. 数据加载修改

**文件位置**: lines 907-930 (main 函数)

**修改前**:
```python
def main():
    import time
    start_time = time.time()

    log("=" * 60)
    log("JPX 30-Day Horizon - LSTM Model (Raw OHLCV Data)")
    log(f"Configuration: seq_length={SEQ_LENGTH}, horizon={TARGET_HORIZON}")
    log(f"Input: 60-day window of OHLCV (5 features)")
    log("=" * 60)

    # Load data
    data, feature_cols, target_col = load_dataset()

    # Run LSTM prediction
    pred = predict_with_lstm(data, feature_cols, target_col)
```

**修改后**:
```python
def main():
    import time
    start_time = time.time()

    log("=" * 60)
    log("JPX 30-Day Horizon - LSTM Model (41 Engineered Features)")
    log(f"Configuration: seq_length={SEQ_LENGTH}, horizon={TARGET_HORIZON}")
    log(f"Input: 60-day window of {len(USE_FEATURES)} features")
    log("=" * 60)

    # Load data with all features
    log("Loading all data sources with engineered features...")
    full_df, all_feature_cols = load_all_data()

    # Filter to use only the specified features
    available_features = [f for f in USE_FEATURES if f in full_df.columns]
    missing_features = [f for f in USE_FEATURES if f not in full_df.columns]

    if missing_features:
        log(f"Warning: {len(missing_features)} features not found: {missing_features[:5]}...")

    log(f"Using {len(available_features)} features out of {len(USE_FEATURES)} requested")

    # Run LSTM prediction
    pred = predict_with_lstm(full_df, available_features, "target_30d")
```

**修改原因**:
- 原来的 `load_dataset()` 只加载 stock_prices.csv，只有 OHLCV 数据
- 新的 `load_all_data()` 加载所有 6 个数据源（stock_prices, stock_list, secondary_stock_prices, options, trades, financials）
- 新方法生成所有 41 个工程特征

**影响**:
- 数据加载时间增加（需要加载和处理更多数据源）
- 内存使用增加（更多特征列）
- 特征质量提升（包含市场上下文信息）

---

### 3. 训练策略完善（2 轮 → 4 轮）

**文件位置**: lines 784-903 (predict_with_lstm 函数)

**修改前**:
```python
pred_parts = []

# ======== 2020 Prediction (Validation) ========
log("Training for 2020 prediction...")
train_mask = years < 2020
test_mask = years == 2020
# ... (training code)

# ======== 2021 Prediction (Test) ========
log("Training for 2021 prediction...")
train_mask = years < 2021
test_mask = years == 2021
# ... (training code)
```

**修改后**:
```python
pred_parts = []

# ======== 2018 Prediction (Validation) ========
log("Training for 2018 prediction...")
train_mask = years < 2018
test_mask = years == 2018
# ... (training code)

# ======== 2019 Prediction (Validation) ========
log("Training for 2019 prediction...")
train_mask = years < 2019
test_mask = years == 2019
# ... (training code)

# ======== 2020 Prediction (Validation) ========
log("Training for 2020 prediction...")
train_mask = years < 2020
test_mask = years == 2020
# ... (training code)

# ======== 2021 Prediction (Test) ========
log("Training for 2021 prediction...")
train_mask = years < 2021
test_mask = years == 2021
# ... (training code)
```

**修改原因**:
- 原来只有 2 轮训练，缺少 2018 和 2019 年的验证
- 新方法使用 expanding window 策略，逐年扩展训练集
- 与 itransformer_model.py 保持一致的训练策略

**训练策略对比**:

| 轮次 | 训练数据 | 测试数据 | 原方法 | 新方法 |
|------|---------|---------|--------|--------|
| Round 1 | 2017 | 2018 | ❌ | ✅ |
| Round 2 | 2017-2018 | 2019 | ❌ | ✅ |
| Round 3 | 2017-2019 | 2020 | ✅ | ✅ |
| Round 4 | 2017-2020 | 2021 | ✅ | ✅ |

**影响**:
- 训练时间增加约 2 倍（4 轮 vs 2 轮）
- 获得更多验证年份的预测结果（2018, 2019）
- 可以更全面地评估模型性能
- 2021 年仍然只用于测试，从不用于训练

---

### 4. 评价标准修改（月度换手 → 每日换手）

**文件位置**: lines 648-709 (evaluate_portfolio 函数)

**修改前**:
```python
def evaluate_portfolio(pred_df):
    """Evaluate portfolio performance - same as train.py."""
    if pred_df.empty:
        return {"num_days": 0, "sharpe": np.nan, "hit_ratio": np.nan, "spread": np.nan}

    pred_df = pred_df.sort_values("Date").reset_index(drop=True)
    dates = sorted(pred_df["Date"].unique())

    # Monthly rebalancing dates (first trading day of each month)
    monthly_dates = []
    current_year_month = None
    for d in dates:
        dt = pd.to_datetime(d)
        year_month = (dt.year, dt.month)
        if year_month != current_year_month:
            monthly_dates.append(d)
            current_year_month = year_month

    daily_results = []
    prev_top = set()
    prev_bottom = set()

    for rebal_date in monthly_dates:
        # ... (evaluation code)
```

**修改后**:
```python
def evaluate_portfolio(pred_df):
    """Evaluate portfolio performance with daily rebalancing - aligned with itransformer_model.py."""
    if pred_df.empty:
        return {"num_days": 0, "sharpe": np.nan, "hit_ratio": np.nan, "spread": np.nan}

    pred_df = pred_df.sort_values("Date").reset_index(drop=True)
    dates = sorted(pred_df["Date"].unique())

    # Daily rebalancing (use all dates)
    daily_dates = dates

    daily_results = []
    prev_top = set()
    prev_bottom = set()

    for rebal_date in daily_dates:
        # ... (evaluation code)
```

**修改原因**:
- 原来只在每月第一个交易日评价，大量预测数据未被使用
- 新方法在每个交易日都进行评价，更准确反映模型性能
- 与 itransformer_model.py 保持一致的评价方式

**评价频率对比**:

| 评价方式 | 评价次数/年 | 4 年总评价次数 | 数据利用率 |
|---------|------------|---------------|-----------|
| 月度换手 | ~12 次 | ~48 次 | ~5% |
| 每日换手 | ~252 次 | ~1000 次 | 100% |

**影响**:
- 评价更准确（使用所有预测数据）
- 评价时间增加约 20 倍（但仍可接受）
- Sharpe ratio 更可靠（基于更多样本）

---

### 5. Sharpe 年化因子修正

**文件位置**: line 704 (evaluate_portfolio 函数)

**修改前**:
```python
sharpe = (avg_spread / std_spread * np.sqrt(4)) if std_spread > 0 else np.nan
```

**修改后**:
```python
# Annualize with sqrt(252) for daily rebalancing
sharpe = (avg_spread / std_spread * np.sqrt(252)) if std_spread > 0 else np.nan
```

**修改原因**:
- 原来使用 sqrt(4) 是针对季度数据（每年 4 个季度）
- 但实际评价是月度换手（每年 12 个月），sqrt(4) 是错误的
- 新方法改为每日换手，应该使用 sqrt(252)（每年约 252 个交易日）

**年化因子对比**:

| 换手频率 | 正确年化因子 | 原方法 | 新方法 |
|---------|------------|--------|--------|
| 每日 | sqrt(252) ≈ 15.87 | sqrt(4) = 2.0 ❌ | sqrt(252) ✅ |
| 每月 | sqrt(12) ≈ 3.46 | sqrt(4) = 2.0 ❌ | - |
| 每季度 | sqrt(4) = 2.0 | sqrt(4) = 2.0 ✅ | - |

**影响**:
- Sharpe ratio 数值会显著增加（约 7.9 倍）
- 但这是正确的年化方式，使得不同频率的策略可以比较
- 与 itransformer_model.py 的 Sharpe 计算方式一致

---

### 6. 文档更新

**文件位置**: lines 1-32 (顶部文档字符串)

**修改前**:
```python
"""
JPX Stock Prediction - LSTM Model with Raw OHLCV Data

Training Strategy:
- Use raw OHLCV data (no manual feature engineering)
- Use 60-day historical window to predict 30-day forward returns
- LSTM learns features automatically from raw price/volume data
- Expanding window: train on past data, predict next year
- 2017-2019 train -> predict 2020
- 2017-2020 train -> predict 2021

Model: LSTM with proper time-series handling
Evaluation: Same as train.py (Sharpe, Spearman, Hit Ratio)
"""
```

**修改后**:
```python
"""
JPX Stock Prediction - LSTM Model with 41 Engineered Features

Training Strategy:
- Use 41 engineered features (aligned with itransformer_model.py)
- Use 60-day historical window to predict 30-day forward returns
- Expanding window: train on past data, predict next year
  - Round 1: 2017 train -> predict 2018
  - Round 2: 2017-2018 train -> predict 2019
  - Round 3: 2017-2019 train -> predict 2020
  - Round 4: 2017-2020 train -> predict 2021

Features (41 total):
- Returns (7): stk_ret_1/2/3/5/10/20, stk_logret_1
- Volatility (3): stk_vol_5/10/20
- Spreads (2): stk_hl_spread, stk_oc_spread
- Volume (4): stk_volume_chg_1, stk_volume_to_ma_5/10/20
- Rolling mean returns (3): stk_ret_mean_5/10/20
- MA ratios (3): stk_close_to_ma_5/10/20
- Distribution (1): stk_skew_20
- Time (2): stk_dayofweek, stk_month
- Company (2): stk_expected_dividend, stk_supervision_flag
- Fundamentals (3): stk_mcap, stk_sector, stk_market_segment
- Options (1): iv_avg
- Trades (4): trd_individual/foreigners/securitiescos/investmenttrusts
- Financials (6): fin_netsales/operatingprofit/ordinaryprofit/profit/totalassets/equity

Model: LSTM with proper time-series handling
Evaluation: Daily rebalancing with sqrt(252) Sharpe annualization
"""
```

**修改原因**:
- 更新文档以反映新的特征使用方式
- 明确列出所有 41 个特征类别
- 说明新的训练策略（4 轮）
- 说明新的评价方式（每日换手）

---

## 配置变更

无配置参数变更。以下参数保持不变：
- `SEQ_LENGTH = 60`（60 天回看窗口）
- `TARGET_HORIZON = 30`（30 天预测窗口）
- `TOP_K = 200`（买入前 200 只股票）
- `BOTTOM_K = 200`（卖空后 200 只股票）
- `LSTM_HIDDEN_SIZE = 64`
- `LSTM_NUM_LAYERS = 2`
- `LSTM_DROPOUT = 0.2`
- `LSTM_EPOCHS = 10`
- `LSTM_BATCH_SIZE = 1024`
- `LSTM_LEARNING_RATE = 0.001`

---

## 影响分析

### 性能影响

| 方面 | 原方法 | 新方法 | 变化 |
|------|--------|--------|------|
| **特征数量** | 5 | 41 | +720% |
| **训练轮次** | 2 | 4 | +100% |
| **评价次数** | ~48 | ~1000 | +1983% |
| **训练时间** | 基准 | ~2× | +100% |
| **评价时间** | 基准 | ~20× | +1900% |
| **总运行时间** | 基准 | ~2-3× | +100-200% |

### 内存影响

| 方面 | 原方法 | 新方法 | 变化 |
|------|--------|--------|------|
| **输入维度** | (batch, 60, 5) | (batch, 60, 41) | +720% |
| **LSTM 参数** | ~50K | ~200K | +300% |
| **数据加载** | 1 个文件 | 6 个文件 | +500% |

### 预期效果

**优点**:
1. **更丰富的信息**：41 个特征 vs 5 个特征，包含更多市场信息
2. **更全面的验证**：4 年验证 vs 2 年验证，更可靠的性能评估
3. **更准确的评价**：每日评价 vs 月度评价，使用所有预测数据
4. **正确的 Sharpe**：sqrt(252) vs sqrt(4)，可与其他策略比较
5. **与 itransformer 对齐**：相同的特征和评价方式，便于模型比较

**缺点**:
1. **训练时间增加**：约 2 倍（4 轮 vs 2 轮）
2. **内存需求增加**：约 3-4 倍（更多特征和数据源）
3. **复杂度增加**：更多特征可能引入噪声

---

## 验证方法

### 1. 语法检查
```bash
cd d:/code/Competition/jpx-tokyo-stock-exchange-prediction
python -m py_compile lstm.py
```

### 2. 运行测试
```bash
cd d:/code/Competition/jpx-tokyo-stock-exchange-prediction
python lstm.py
```

### 3. 检查日志输出

**特征使用验证**:
```
[INFO] Loading all data sources with engineered features...
[INFO] Using 41 features out of 41 requested
[INFO] Input: 60-day window of 41 features
```

**训练轮次验证**:
```
[INFO] Training for 2018 prediction...
[INFO]   2018: Train XXX, Test XXX
[INFO] Training for 2019 prediction...
[INFO]   2019: Train XXX, Test XXX
[INFO] Training for 2020 prediction...
[INFO]   2020: Train XXX, Test XXX
[INFO] Training for 2021 prediction...
[INFO]   2021: Train XXX, Test XXX
```

**评价方式验证**:
```
[INFO] PORTFOLIO METRICS
[INFO] Rebalance Days: ~1000  # 应该是 ~1000 而不是 ~48
[INFO] Sharpe: X.XXXX  # 数值应该比原来大约 7.9 倍
```

### 4. 对比指标

**预期改进**:
- 特征数量：5 → 41 ✅
- 训练轮次：2 → 4 ✅
- 评价天数：~48 → ~1000 ✅
- Sharpe 年化：sqrt(4) → sqrt(252) ✅
- 模型性能：可能提升（更多特征）

---

## 风险和注意事项

### 1. 特征质量
- 并非所有 41 个特征都有用，可能引入噪声
- 建议：先运行，观察结果，如果性能下降可以做特征选择

### 2. 数据缺失
- 某些特征（如财务数据）可能有大量缺失值
- 当前处理：`fillna(0)` (line 279 in build_feature_table)
- 可能需要更好的缺失值处理策略

### 3. 类别特征
- `stk_sector`, `stk_market_segment` 是类别编码
- 当前处理：直接使用数值编码
- 可能需要 embedding 或 one-hot 编码（但会增加维度）

### 4. 时间特征
- `stk_dayofweek`, `stk_month` 是周期性特征
- 可能需要 sin/cos 变换来保持周期性

### 5. 内存和 GPU
- 41 个特征需要更多内存和 GPU 显存
- 如果遇到 OOM，可以：
  - 减少 `LSTM_BATCH_SIZE`（从 1024 减少到 512 或 256）
  - 减少 `max_train_samples`（从 300000 减少到 200000）
  - 减少特征数量（选择最重要的 20-30 个特征）

### 6. 过拟合风险
- 更多特征可能导致过拟合
- 建议：监控训练集和验证集的性能差异
- 如果过拟合，可以：
  - 增加 `LSTM_DROPOUT`（从 0.2 增加到 0.3 或 0.4）
  - 减少 `LSTM_EPOCHS`（从 10 减少到 5）
  - 使用特征选择

---

## 总结

本次修改成功将 lstm.py 与 itransformer_model.py 对齐，主要改进包括：

1. ✅ **特征扩展**：从 5 个 OHLCV 特征扩展到 41 个工程特征
2. ✅ **训练完善**：从 2 轮训练扩展到 4 轮训练（expanding window）
3. ✅ **评价改进**：从月度换手改为每日换手
4. ✅ **Sharpe 修正**：从 sqrt(4) 修正为 sqrt(252)
5. ✅ **文档更新**：更新顶部文档说明

这些修改使得 lstm.py 和 itransformer_model.py 使用相同的特征、训练策略和评价方式，便于公平比较两种模型的性能。

**下一步建议**:
1. 运行修改后的 lstm.py，验证所有修改是否正常工作
2. 比较 lstm.py 和 itransformer_model.py 的性能指标
3. 如果性能不理想，考虑特征选择或超参数调优
4. 监控内存和 GPU 使用情况，必要时调整配置

---

## 紧急 Bug 修复（2026-03-07）

### 问题描述

在运行修改后的 lstm.py 时，发现了严重问题：
- 加载了 34 个特征（7 个缺失：stk_mcap, stk_sector, stk_market_segment, trd_individual, trd_foreigners）
- **创建了 0 个序列**：`X=(0,), y=(0,)`
- 无法生成任何预测

### 根本原因分析

通过深入分析，发现了 **两个关键 Bug**：

#### Bug 1: 目标索引错误（主要问题）

**位置**: `create_sequences()` 函数，line 507

**问题**:
```python
# Line 503: 循环在 len(stock_data) - TARGET_HORIZON 处停止（减去 30）
for i in range(seq_length, len(stock_data) - TARGET_HORIZON):
    seq_features = feature_values[i - seq_length:i]

    # Line 507: 然后尝试访问 i + TARGET_HORIZON（加上 30）
    target = target_values[i + TARGET_HORIZON]  # ❌ 错误！
```

**根本原因**:
- `target_30d` 已经是 30 天前向收益率（line 353: `shift(-30)`）
- 这意味着每只股票的最后 30 行已经有 NaN 目标值
- 循环正确地提前 30 行停止：`range(seq_length, len(stock_data) - TARGET_HORIZON)`
- 但 line 507 又尝试访问 `target_values[i + TARGET_HORIZON]`，这相当于向前 60 天
- 结果：所有访问的目标值都是 NaN！

**修复**:
```python
# Line 507: 修改为直接使用 i
target = target_values[i]  # ✅ 正确：target_30d 已经是前向收益率
```

#### Bug 2: 归一化中的 NaN 传播（次要问题）

**位置**: `predict_with_lstm()` 函数，lines 765-772

**问题**:
```python
for col in feature_cols:
    mean_val = train_stats[col]['mean']
    std_val = train_stats[col]['std']
    df_norm[col] = (df[col] - mean_val) / (std_val + 1e-8)  # 如果 mean_val 或 std_val 是 NaN，结果是 NaN
    df_norm[col] = df_norm[col].clip(-10, 10)  # clip 不会修复 NaN 值
```

**根本原因**:
- 如果某个特征的 mean 或 std 是 NaN（例如，训练期间所有值都是 0，或全是 NaN）
- 归一化会产生整列 NaN
- 然后所有序列都会因为 NaN 检查（line 510）而被跳过

**修复**:
```python
for col in feature_cols:
    mean_val = train_stats[col]['mean']
    std_val = train_stats[col]['std']

    # Handle NaN mean/std to prevent NaN propagation
    if pd.isna(mean_val):
        mean_val = 0.0
    if pd.isna(std_val) or std_val < 1e-8:
        std_val = 1.0

    df_norm[col] = (df[col] - mean_val) / (std_val + 1e-8)
    df_norm[col] = df_norm[col].fillna(0)  # Fill any remaining NaN with 0
    df_norm[col] = df_norm[col].clip(-10, 10)
```

### 修改详情

#### 修改 1: 修复目标索引（Bug 1）

**文件**: `lstm.py`
**位置**: Line 507 in `create_sequences()` function

**修改前**:
```python
target = target_values[i + TARGET_HORIZON]
```

**修改后**:
```python
target = target_values[i]  # target_30d is already a forward return
```

**影响**:
- 将能够创建序列（不再全是 NaN 目标）
- 预期创建数十万个序列（与 itransformer 类似）

#### 修改 2: 修复归一化 NaN 处理（Bug 2）

**文件**: `lstm.py`
**位置**: Lines 765-772 in `predict_with_lstm()` function

**修改前**:
```python
for col in feature_cols:
    mean_val = train_stats[col]['mean']
    std_val = train_stats[col]['std']
    df_norm[col] = (df[col] - mean_val) / (std_val + 1e-8)
    df_norm[col] = df_norm[col].clip(-10, 10)
```

**修改后**:
```python
for col in feature_cols:
    mean_val = train_stats[col]['mean']
    std_val = train_stats[col]['std']

    # Handle NaN mean/std to prevent NaN propagation
    if pd.isna(mean_val):
        mean_val = 0.0
    if pd.isna(std_val) or std_val < 1e-8:
        std_val = 1.0

    df_norm[col] = (df[col] - mean_val) / (std_val + 1e-8)
    df_norm[col] = df_norm[col].fillna(0)  # Fill any remaining NaN with 0
    df_norm[col] = df_norm[col].clip(-10, 10)
```

**影响**:
- 防止 NaN 在归一化过程中传播
- 使代码更健壮，能够处理边缘情况

### 验证结果

#### 语法检查
```bash
python -m py_compile lstm.py
```
✅ 通过

#### 预期运行结果

**修复前（错误）**:
```
[INFO] Created sequences: X=(0,), y=(0,)
[INFO] No sequences created!
```

**修复后（预期）**:
```
[INFO] Created sequences: X=(500000, 60, 34), y=(500000,)
[INFO]   2018: Train 100,000, Test 50,000
[INFO]   2019: Train 200,000, Test 50,000
[INFO]   2020: Train 300,000, Test 50,000
[INFO]   2021: Train 400,000, Test 50,000
[INFO] Total predictions: 200,000
```

### 对比 itransformer_model.py

这两个 Bug 在 itransformer_model.py 中都被正确处理：
1. **目标索引**: itransformer 使用 `target_values[i]` 直接访问（line 447）
2. **NaN 处理**: itransformer 使用 `np.nanmean()` 和 `np.nanstd()` 优雅处理 NaN（lines 787-788），并设置 `x_std[x_std < 1e-8] = 1.0` 避免除零问题（line 789）

### 总结

这两个 Bug 修复后，lstm.py 应该能够：
1. ✅ 成功创建序列（不再是 0 个）
2. ✅ 生成预测结果
3. ✅ 与 itransformer_model.py 使用相同的逻辑处理目标和归一化

**优先级**: 紧急修复 - Bug 1 必须立即修复，否则 lstm.py 完全无法工作
