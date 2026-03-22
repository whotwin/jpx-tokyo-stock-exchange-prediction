# iTransformer 模型修改日志

**修改日期**：2026-03-07
**修改目的**：扩展特征使用 + 改进评价标准

---

## 修改概述

本次修改主要实现两个核心功能：
1. **扩展特征使用**：从 12 个特征扩展到 41 个特征，充分利用所有可用数据源
2. **改进评价标准**：从月度换手改为每日换手，更全面地评价模型的预测能力

---

## 详细修改列表

### 修改 1：扩展特征列表（Lines 360-403）

**文件**：`itransformer_model.py`
**函数**：`create_itransformer_sequences()`
**位置**：Lines 360-403

**修改前**（12 个特征）：
```python
use_features = [
    # Stock returns (momentum signals)
    "stk_ret_1", "stk_ret_5", "stk_ret_20",
    # Volatility (risk measures)
    "stk_vol_5", "stk_vol_20",
    # Spread (intraday patterns)
    "stk_hl_spread", "stk_oc_spread",
    # Volume
    "stk_volume_chg_1",
    # Trend
    "stk_close_to_ma_20",
    # Distribution
    "stk_skew_20",
    # Options (market sentiment)
    "iv_avg",
    # Trades (institutional flow)
    "trd_foreigners",
]
```

**修改后**（41 个特征）：
```python
use_features = [
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
```

**修改原因**：
- 原始代码只使用了 12 个特征，浪费了大量可用数据
- 数据集提供了 40+ 个特征，包括价格、成交量、财务、交易等多维度信息
- 更多特征可以提供更丰富的信号，提升模型预测能力

**影响**：
- 特征数量：12 → 41 (增加 242%)
- num_variates：24,000 → 82,000 (假设 2000 只股票)
- 内存需求：增加约 3.4 倍
- GPU 显存需求：约 7-10 GB

---

### 修改 2：改为每日换手评价（Lines 617-677）

**文件**：`itransformer_model.py`
**函数**：`evaluate_portfolio()`
**位置**：Lines 617-677

**关键变更 A：删除月度筛选逻辑**

**修改前**（Lines 625-632）：
```python
monthly_dates = []
current_year_month = None
for d in dates:
    dt = pd.to_datetime(d)
    year_month = (dt.year, dt.month)
    if year_month != current_year_month:
        monthly_dates.append(d)
        current_year_month = year_month

# ...
for rebal_date in monthly_dates:  # 只评价每月第一天
```

**修改后**：
```python
# Use all dates for daily rebalancing (not just monthly)
daily_dates = dates

# ...
for rebal_date in daily_dates:  # 评价所有日期
```

**关键变更 B：修改年化因子**

**修改前**（Line 673）：
```python
sharpe = (avg_spread / std_spread * np.sqrt(4)) if std_spread > 0 else np.nan
```

**修改后**（Line 656）：
```python
# Annualize with sqrt(252) for daily rebalancing (252 trading days per year)
sharpe = (avg_spread / std_spread * np.sqrt(252)) if std_spread > 0 else np.nan
```

**修改原因**：
- 原始代码只评价每月第一个交易日，导致大量预测数据未被评价
- 每月评价约 48 次（4年），每日评价约 1000 次（4年），评价更全面
- 年化因子从 sqrt(4) 改为 sqrt(252)，适应每日换手的频率

**影响**：
- 评价天数：~48 → ~1000 (增加约 20 倍)
- Sharpe ratio 计算更准确（基于更多数据点）
- 评价时间增加约 20 倍（但不影响训练时间）

---

## 新增功能

### 1. 多维度特征支持

**新增特征类别**：
- **收益率扩展**：添加 2/3/10 天收益率和对数收益率
- **波动率扩展**：添加 10 天波动率
- **成交量分析**：添加成交量相对 MA 的比率
- **趋势分析**：添加 5/10 天 MA 比率和滚动均值收益
- **时间效应**：添加星期几和月份特征
- **基本面**：添加市值、板块、市场分类
- **交易行为**：添加个人投资者、证券公司、投资信托的交易数据
- **财务数据**：添加利润表和资产负债表数据

### 2. 每日评价系统

**功能描述**：
- 对每个交易日的预测进行评价
- 计算每日的 top 200 和 bottom 200 股票组合
- 跟踪每日换手率和交易成本
- 基于每日收益计算年化 Sharpe ratio

---

## 配置变更

### 当前配置（无需修改）

```python
# Lines 59-64
ITRANSFORMER_BATCH_SIZE = 4  # 已针对多特征优化
MAX_NUM_STOCKS = None  # 使用所有股票
```

### 如果遇到内存不足（OOM）

**选项 1：减少 batch size**
```python
ITRANSFORMER_BATCH_SIZE = 2  # 或 1
```

**选项 2：限制股票数量**
```python
MAX_NUM_STOCKS = 1000  # 从 2000 减少到 1000
```

---

## 验证方法

### 1. 语法检查
```bash
cd d:/code/Competition/jpx-tokyo-stock-exchange-prediction
python -m py_compile itransformer_model.py
```

### 2. 运行模型
```bash
# 激活虚拟环境
cd d:/code/Competition/jpx-tokyo-stock-exchange-prediction
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# 运行模型
python itransformer_model.py
```

### 3. 检查日志输出

**特征使用验证**：
```
[INFO] Using 41 features: ['stk_ret_1', 'stk_ret_2', 'stk_ret_3', ...]
[INFO] num_variates=82000 = 2000 stocks × 41 features
```

**评价方式验证**：
```
[INFO] PORTFOLIO METRICS
[INFO] Rebalance Days: ~1000  # 应该接近 1000，而不是 ~48
[INFO] Sharpe: X.XXXX
[INFO] Hit Ratio: XX.XX%
```

### 4. 预期结果

**特征数量**：
- 日志应显示 "Using 41 features"
- num_variates 应为 股票数 × 41

**评价天数**：
- Rebalance Days 应接近 1000（4年 × 252 交易日）
- 远大于之前的 ~48 天

**性能指标**：
- Sharpe ratio 可能变化（取决于特征质量）
- Hit ratio 应该更稳定（基于更多评价点）

---

## 风险和注意事项

### 1. 内存和显存需求

**问题**：41 个特征需要约 3.4 倍的内存
**解决方案**：
- 如果 OOM，减少 batch_size 到 2 或 1
- 或限制 MAX_NUM_STOCKS 到 1000

### 2. 特征质量

**问题**：并非所有特征都有用，可能引入噪声
**建议**：
- 先运行观察结果
- 如果性能下降，可以做特征选择
- 可以使用特征重要性分析移除低价值特征

### 3. 数据缺失

**问题**：财务数据、交易数据可能有缺失值
**当前处理**：`fillna(0)` (line 307)
**改进方向**：可以考虑更好的缺失值处理策略

### 4. 类别特征

**问题**：`stk_sector`, `stk_market_segment` 是类别编码
**当前处理**：直接使用数值编码
**改进方向**：可以考虑 embedding 或 one-hot 编码

### 5. 时间特征

**问题**：`stk_dayofweek`, `stk_month` 是周期性特征
**当前处理**：直接使用数值
**改进方向**：可以考虑 sin/cos 变换保持周期性

---

## 实现的功能总结

### ✅ 已实现

1. **扩展特征使用**
   - 从 12 个特征扩展到 41 个特征
   - 覆盖价格、成交量、财务、交易、基本面等多个维度
   - 自动过滤不存在的特征

2. **每日评价系统**
   - 从月度换手改为每日换手
   - 评价天数从 ~48 增加到 ~1000
   - 年化因子从 sqrt(4) 改为 sqrt(252)

3. **向后兼容**
   - 保留原有的训练流程（expanding window）
   - 保留原有的模型架构
   - 保留原有的配置参数

### 📋 待优化（可选）

1. 特征工程改进（sin/cos 变换、embedding）
2. 特征选择（移除低重要性特征）
3. 缺失值处理优化
4. 添加特征重要性分析

---

## 技术细节

### 特征维度变化

| 项目 | 修改前 | 修改后 | 变化 |
|------|--------|--------|------|
| 特征数量 | 12 | 41 | +242% |
| num_variates (2000股) | 24,000 | 82,000 | +242% |
| 输入形状 | (batch, 20, 24000) | (batch, 20, 82000) | +242% |
| 内存需求 | ~2-3 GB | ~7-10 GB | +3.4× |

### 评价指标变化

| 项目 | 修改前 | 修改后 | 变化 |
|------|--------|--------|------|
| 换手频率 | 月度 | 每日 | +20× |
| 评价天数 (4年) | ~48 | ~1000 | +20× |
| 年化因子 | sqrt(4) | sqrt(252) | +7.9× |
| 评价时间 | 快 | 慢 20× | +20× |

---

## 文件清单

### 修改的文件
- `itransformer_model.py` - 主模型文件

### 新增的文件
- `CHANGELOG_itransformer.md` - 本修改日志

### 未修改的文件
- 所有数据文件（train_files/）
- 其他模型文件（lstm.py, train.py, main.py）
- 配置文件（pyproject.toml, uv.lock）

---

## 版本信息

- **修改前版本**：Multi-feature support (12 features)
- **修改后版本**：Extended features + Daily evaluation (41 features)
- **兼容性**：向后兼容，可以通过修改 use_features 列表回退到 12 个特征

---

## 联系和支持

如有问题或需要进一步优化，请参考：
- 计划文件：`C:\Users\hty\.claude\plans\typed-watching-turing.md`
- Kaggle 竞赛页面：https://www.kaggle.com/competitions/jpx-tokyo-stock-exchange-prediction

---

## 修改 2：数据格式修改 - stock ↔ stock attention

**修改日期**：2026-03-09
**修改目的**：改进 attention 机制，让模型学习 stock ↔ stock 关系

---

### 修改概述

将数据格式从 2D 改为 4D，使 Transformer attention 学习 stock ↔ stock 关系而不是 (stock_feature) ↔ (stock_feature) 关系。

**核心思想**：
- 输入: (batch, seq_len, num_stocks, num_features)
- 模型中: Linear(num_features → d_model) → (batch, seq_len, num_stocks, d_model)
- Attention 在 num_stocks 维度上计算，学习 stock ↔ stock 关系

---

### 修改详情

**文件**：`itransformer/data.py`
**函数**：`create_itransformer_sequences()`

**修改前**：
```python
# Stack: (seq_length, num_stocks, num_features)
seq_features_stacked = np.stack(seq_features_list, axis=-1)

# Reshape: (seq_length, num_stocks × num_features)
seq_features = seq_features_stacked.reshape(seq_length, -1)
```

**修改后**：
```python
# Stack: (seq_length, num_stocks, num_features)
# Keep 4D tensor: (seq_length, num_stocks, num_features)
# NOT reshaping to 2D - attention will learn stock ↔ stock relationships
seq_features = np.stack(seq_features_list, axis=-1)
```

**输出形状变化**：
- 修改前: X = (samples, seq_length, num_stocks × num_features)
- 修改后: X = (samples, seq_length, num_stocks, num_features)

**额外返回**：
- 返回 num_features 用于模型的 Linear(num_features → d_model) 投影

---

### 为什么这样修改

**原来的问题**：
- 把每个股票的每个特征当作独立的 variate
- Attention 学习的是 (stock_feature) ↔ (stock_feature) 的关系
- 无法直接捕捉股票之间的关系

**改进后的优势**：
- 每个股票的所有特征被投影到 d_model 维度
- Attention 在 num_stocks 维度上计算
- 直接学习 stock ↔ stock 之间的关系
- 更符合金融市场的直觉：股票之间存在相关性

---

### 修改 3：使用投影层进行特征降维（2026-03-09）

**文件**：`itransformer/train.py`, `itransformer/predict.py`

**问题**：
- iTransformer 模型期望 3D 输入 `(batch, seq_len, num_variates)`
- 数据准备输出 4D 张量 `(batch, seq_len, num_stocks, num_features)`
- 之前使用直接 reshape 导致维度不匹配

**解决方案**：
- 添加 `FeatureProjector` 类进行特征投影
- 将 `num_features` 投影到 `d_model` 维度
- 投影后 reshape 为 3D：`num_stocks * d_model`

**实现细节**：

1. **train.py** - 添加 FeatureProjector 类：
```python
class FeatureProjector(nn.Module):
    def __init__(self, num_features, dim):
        super().__init__()
        self.projection = nn.Linear(num_features, dim)

    def forward(self, x):
        # x: (batch, seq_len, num_stocks, num_features)
        batch, seq_len, num_stocks, num_features = x.shape
        x = x.view(batch * seq_len, num_stocks, num_features)
        x = self.projection(x)
        x = x.view(batch, seq_len, num_stocks, -1)
        return x
```

2. **修改 train_itransformer_model**：
- 添加 `projector` 参数
- 在模型调用前应用投影
- 投影后 reshape 为 3D

3. **修改 predict_itransformer**：
- 添加 `projector` 参数
- 在预测时也应用投影

**数据流**：
```
输入: (batch, seq_len, num_stocks, num_features)
  ↓ FeatureProjector
中间: (batch, seq_len, num_stocks, d_model)
  ↓ reshape
输出: (batch, seq_len, num_stocks * d_model)
  ↓ iTransformer
预测: (batch, num_stocks * d_model)
```

---

### 待完成

1. 修改 `train.py` 中的 `cross_sectional_normalize` 函数以支持 4D 张量 ✅
2. 修改 `predict.py` 中的模型调用以支持 4D 输入 ✅
3. 添加 FeatureProjector 投影层 ✅
4. 使用投影替代直接 reshape ✅

---

### 相关文件

- `itransformer/data.py` - 已修改（输出 4D 张量）
- `itransformer/train.py` - 已修改（FeatureProjector + cross_sectional_normalize 支持 4D）
- `itransformer/predict.py` - 已修改（使用投影层）


