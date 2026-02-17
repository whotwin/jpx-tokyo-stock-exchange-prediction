# JPX东京证券交易所股票预测项目

## 项目概述

本项目使用日本东京证券交易所(JPX)的真实市场数据，构建机器学习模型来预测股票未来收益。目标是辅助金融从业者的投资决策，通过预测结果来决定股票的买进和卖出，以实现收益最大化。

---

## 目录结构

```
jpx-tokyo-stock-exchange-prediction/
├── train_files/                    # 训练数据 (2017-2021)
│   ├── stock_prices.csv           # 股票价格数据
│   ├── secondary_stock_prices.csv # 创业板股票价格
│   ├── options.csv                # 期权数据
│   ├── trades.csv                 # 投资者交易数据
│   └── financials.csv             # 上市公司财务数据
│
├── supplemental_files/            # 补充数据
│
├── example_test_files/            # 测试数据示例
│   ├── sample_submission.csv     # 提交示例
│   └── ...
│
├── data_specifications/           # 数据规格说明
│   ├── stock_price_spec.csv       # 股票价格字段说明
│   ├── stock_fin_spec.csv         # 财务数据字段说明
│   ├── stock_list_spec.csv        # 股票列表字段说明
│   ├── options_spec.csv           # 期权数据字段说明
│   └── trades_spec.csv            # 交易数据字段说明
│
├── output_v2/                    # demo_v2实验结果
│   ├── ablation_summary.csv        # 特征消融实验结果
│   ├── pred_test_*.csv            # 各配置的预测结果
│   ├── weights_test_*.csv         # 组合权重
│   └── plots/                     # 可视化图表
│
├── output_horizon_compare/        # horizon模型对比实验结果
│   ├── horizon_model_metrics.csv  # 模型评估指标
│   ├── horizon_datasource_metrics.csv  # 数据源对比结果
│   ├── pred_*.csv                # 各模型预测
│   └── plots/                     # 可视化图表
│
├── output_lgbm_20d_all/          # 最佳模型输出
│   ├── predictions.csv            # LGBM 20d预测结果
│   ├── metrics.csv                # 评估指标
│   └── plots/                     # 可视化
│
├── demo_v2.py                    # 核心数据处理和模型训练
├── horizon_model_comparison.py    # 多模型多horizon对比实验
├── lgbm_20d_all.py              # 最佳模型独立运行脚本
└── PROJECT_SUMMARY.md            # 项目总结
```

---

## 数据集说明

### 1. 股票价格数据 (stock_prices.csv)

| 字段 | 说明 |
|------|------|
| Date | 交易日期 |
| SecuritiesCode | 证券代码 |
| Open/High/Low/Close | 开盘价/最高价/最低价/收盘价 |
| Volume | 成交量 |
| AdjustmentFactor | 拆股/合股调整因子 |
| SupervisionFlag | 监管标记 |
| ExpectedDividend | 预期分红 |
| **Target** | **预测目标**: t+2日相比t+1日的调整后收盘价变化率 |

### 2. 创业板股票价格 (secondary_stock_prices.csv)

- 创业板(Pro Market)股票的日线数据
- 包含字段与主版类似

### 3. 期权数据 (options.csv)

| 字段 | 说明 |
|------|------|
| Putcall | 看涨/看跌期权标识 |
| ContractMonth | 合约月份 |
| ImpliedVolatility | 期权隐含波动率 |
| TradingVolume | 成交量 |
| OpenInterest | 未平仓合约数 |
| SettlementPrice | 结算价 |
| BaseVolatility | 基础波动率 |

### 4. 投资者交易数据 (trades.csv)

按周汇总的各类投资者买卖金额：

- **个人投资者** (Individuals)
- **外资** (Foreigners)
- **证券公司** (Securities Cos)
- **投资信托** (Investment Trusts)
- **保险公司** (Insurance Cos)
- **城市银行/地方银行** (CityBKs/RegionalBKs)
- **信托银行** (Trust Banks)

### 5. 财务数据 (financials.csv)

上市公司财务报表关键指标：

| 字段 | 说明 |
|------|------|
| NetSales | 净销售额 |
| OperatingProfit | 营业利润 |
| OrdinaryProfit | 普通利润 |
| Profit | 净利润 |
| TotalAssets | 总资产 |
| Equity | 股东权益 |
| EquityToAssetRatio | 权益比率 |
| EarningsPerShare (EPS) | 每股收益 |
| ForecastEPS | 预期每股收益 |

---

## 任务目标

### 核心任务

**构建一个股票收益预测模型，辅助金融从业者进行投资决策**

### 具体目标

1. **预测未来收益**: 预测股票在未来1天、5天、20天的收益率
2. **生成交易信号**: 根据预测结果生成买/卖/持有信号
3. **构建投资组合**: 构建多头/空头组合，优化风险调整后收益

### 评价指标

| 指标 | 说明 |
|------|------|
| **RankIC** | 日级别Spearman相关系数，衡量排序能力 |
| **RankIC IR** | RankIC均值/标准差，信息比率 |
| **Hit Ratio** | 预测方向准确率 |
| **Sharpe Ratio** | 夏普比率，风险调整收益 |
| **Max Drawdown** | 最大回撤 |

---

## 模型方案

### 当前实现

1. **LightGBM** (主模型)
   - Walk-forward训练，每月重训
   - 2年滚动训练窗口
   - 最佳表现: RankIC ~0.025, Sharpe ~8.07

2. **Ridge回归** (基准)
   - 静态模型，一次训练

3. **LSTM/Transformer** (实验)
   - 深度学习时序模型
   - 当前表现不如LightGBM

### 特征工程

- **价格特征**: 收益率、布林带、成交量变化
- **技术指标**: 移动平均、波动率、偏度
- **期权特征**: 隐含波动率、波动率曲面
- **财务特征**: EPS、利润、资产变化
- **交易者特征**: 各类型投资者净买入

---

## 实验结果

### 最佳模型配置

| 配置 | 值 |
|------|------|
| 数据源 | stock+all (全部) |
| 模型 | LightGBM |
| Horizon | 20天 |
| Mean Daily RankIC | 0.025 |
| RankIC IR | 0.19 |
| 组合回报 | 1043% |
| 夏普比率 | 8.07 |

### 数据源影响

| 数据源 | RankIC提升 |
|--------|-----------|
| stock_only | 基准 |
| stock+all | **+50%** |

---

## 使用方法

### 运行完整实验

```bash
# 多模型多horizon对比
python horizon_model_comparison.py

# 最佳模型单独运行
python lgbm_20d_all.py
```

### 查看结果

```bash
# 模型对比结果
cat output_horizon_compare/horizon_model_metrics.csv

# 最佳模型指标
cat output_lgbm_20d_all/metrics.csv
```

---

## 注意事项

1. **数据时间范围**: 2017-01-04 至 2021-12-03
2. **测试年份**: 2021年作为留出测试集
3. **防数据泄露**: 训练数据不包含测试期信息
4. **交易成本**: 组合评估包含交易成本和滑点

---

## 扩展方向

1. 尝试更多特征工程
2. 集成学习方法
3. 更复杂的深度学习架构
4. 风险约束优化
