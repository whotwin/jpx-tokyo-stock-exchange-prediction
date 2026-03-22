# LGBM/XGBoost模型评估报告

## 1. 任务描述

利用前两个季度的股票特征数据，预测下一个季度的股价收益率。

评估指标：买入预测收益最高的200支股票，卖出预测收益最低的200支股票，计算Sharpe值。

## 2. 数据概况

### 2.1 数据集大小
- 训练集: 24,313条记录 (2017Q1 - 2020Q4, 共16个季度)
- 测试集: 4,853条记录 (2021Q1 - 2021Q3, 共3个季度)
- 股票数量: 约1,500-1,600支股票/季度

### 2.2 特征数量
- 原始特征: 28个
- 滞后特征: 56个 (2个季度的滞后 × 28个特征)
- 模型使用特征: 84个

### 2.3 特征列表
```
价格类: Open, High, Low, Close, Volume
财务类: NetSales, OperatingProfit, OrdinaryProfit, Profit, EarningsPerShare,
        TotalAssets, Equity, EquityToAssetRatio
期权类: opt_SettlementPrice, opt_TheoreticalPrice, opt_TradingVolume,
        opt_OpenInterest, opt_BaseVolatility, opt_ImpliedVolatility,
        opt_InterestRate, opt_DividendRate
交易类: trade_TotalSales, trade_TotalPurchases, trade_TotalTotal,
        trade_TotalBalance, trade_IndividualsBalance,
        trade_ForeignersBalance, trade_ProprietaryBalance
```

## 3. 模型评估结果 (XGBoost with GPU)

### 3.1 整体表现
| 指标 | 值 |
|------|-----|
| Overall Sharpe | -1.2048 |
| Kaggle Sharpe | -0.6127 |
| Hit Ratio | 47.2% |

### 3.2 各季度表现
| 季度 | RMSE | Spearman相关 | Hit Rate | Spread |
|------|------|--------------|----------|--------|
| 2021Q1 | 0.941 | -0.151 | 29.9% | -0.189 |
| 2021Q2 | 0.650 | 0.025 | 64.0% | 0.017 |
| 2021Q3 | 0.680 | 0.013 | 58.0% | 0.003 |

### 3.3 预测分布
- 预测值均值: -0.252 (实际值: 0.083)
- 预测值标准差: 0.133 (实际值: 0.675)
- 预测方向准确率: ~47%

### 3.4 最佳超参数 (XGBoost)
```
n_estimators: 200
learning_rate: 0.083
max_depth: 3
min_child_weight: 49
subsample: 0.916
colsample_bytree: 0.606
reg_alpha: 0.0008
reg_lambda: 0.0008
device: cuda (GPU)
```

## 4. 对比: LightGBM vs XGBoost

| 指标 | LightGBM | XGBoost |
|------|----------|---------|
| Overall Sharpe | -1.62 | -1.20 |
| Kaggle Sharpe | -0.73 | -0.61 |
| Hit Ratio | 48.5% | 47.2% |

XGBoost略优于LightGBM。

## 4. 问题分析

### 4.1 主要问题

#### 问题1: 预测值方差极小
- 模型预测值标准差仅为0.133，而实际值标准差为0.675
- 模型预测值几乎没有任何变异，集中在很小的范围内
- 这导致模型无法有效区分不同股票的收益差异

#### 问题2: 预测偏差严重
- 预测均值: -0.066
- 实际均值: +0.083
- 模型系统性地低估了股票收益

#### 问题3: 相关性接近零
- 最高Spearman相关系数仅为0.028
- 表明模型几乎没有学到任何有效的预测模式

#### 问题4: 方向准确率接近随机
- 方向准确率: 49.84% ≈ 50%
- 与随机猜测无异

### 4.2 根本原因分析

#### 原因1: 股票收益率预测本质困难
- 股票市场具有高度随机性
- 有效市场假说表明历史信息难以预测未来收益
- 特征与Target的相关性最高仅为0.36

#### 原因2: 数据分布偏移
- 训练集Target均值: -0.0006, 标准差: 1.04
- 测试集Target均值: 0.087, 标准差: 0.68
- 训练和测试期间市场环境发生显著变化

#### 原因3: 特征质量问题
- 特征值经过标准化处理（可能是z-score）
- 可能存在信息损失
- 滞后特征可能引入噪声

#### 原因4: 模型配置问题
- 默认超参数可能不适合此任务
- 需要更细致的超参数调优

## 5. 改进建议

### 5.1 特征工程
1. 添加技术指标特征（如RSI、MACD等）
2. 添加行业/板块特征
3. 添加市场情绪指标
4. 尝试不同滞后阶数

### 5.2 模型改进
1. 增加超参数搜索的trial数量
2. 尝试不同的模型（如XGBoost、神经网络）
3. 调整训练窗口大小
4. 集成多个模型

### 5.3 评估方式改进
1. 使用更多回测期进行验证
2. 考虑交易成本
3. 添加其他评估指标（如Information Ratio）

### 5.4 数据处理
1. 尝试不同的特征标准化方法
2. 处理数据分布偏移问题
3. 增加更多训练数据

## 6. 代码实现细节

### 6.1 超参数搜索
- 使用Optuna进行贝叶斯超参数优化
- 搜索参数：n_estimators, learning_rate, max_depth, num_leaves, min_child_samples, subsample, colsample_bytree, reg_alpha, reg_lambda
- 优化目标：最大化Sharpe值

### 6.2 Expanding Window训练
- 对每个测试季度，使用所有之前的历史数据进行训练
- 例如：预测2021Q1时，使用2017Q3-2020Q4的所有数据

### 6.3 特征使用
- 原始特征：28个（来自cleaned_data）
- 滞后特征：56个（2个季度 × 28个特征）
- 总特征数：84个

### 6.4 交叉验证方式
- 使用TimeSeriesSplit进行时间序列交叉验证
- 验证方式：
  - Fold 1: 训练 2017Q3-2017Q12 (5季度) -> 验证 2018Q1-2018Q3 (3季度)
  - Fold 2: 训练 2017Q3-2018Q3 (8季度) -> 验证 2018Q4-2019Q2 (3季度)
  - Fold 3: 训练 2017Q3-2019Q2 (11季度) -> 验证 2019Q3-2020Q4 (3季度)

### 6.5 Sharpe计算方式
按照Kaggle官方方法：
1. 按预测值排序
2. 买入预测收益最高的前200支股票
3. 卖出预测收益最低的后200支股票
4. 使用加权计算spread
5. Sharpe = mean(spread) / std(spread)

## 7. 结论

当前LGBM模型在股票收益率预测任务上表现不佳，主要原因是：
1. 股票收益率预测本身的困难性
2. 特征与目标之间的相关性较弱
3. 模型预测能力有限
4. 数据分布存在偏移
5. 超参数搜索可能被跳过（上次运行使用了--no-hyperopt）

建议进行更深入的特征工程和模型调优，以提高预测性能。

---
生成时间: 2026-03-12
最后更新: 2026-03-12
