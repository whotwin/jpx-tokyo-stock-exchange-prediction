# JPX东京股票交易所预测 - 数据集与任务说明

## 竞赛概述

**Japan Exchange Group (JPX)** 是全球最大的股票交易所之一，主办此次竞赛旨在推动量化投资策略的研究。

### 竞赛目标
预测日本东京证券交易所股票的**次日收益率**，并按预期收益对股票进行排名。

---

## 数据集结构

### 1. stock_list.csv - 股票列表
| 字段 | 类型 | 说明 |
|------|------|------|
| SecuritiesCode | int | 股票代码 (唯一标识) |
| Name | string | 公司名称 |
| Section/Products | string | 板块分类 |
| 33SectorName | string | 33行业分类 |
| 17SectorName | string | 17行业分类 |
| MarketCapitalization | float | 市值 |
| **Universe0** | boolean | **预测目标标记 (True=需预测)** |

**关键**: 只有 `Universe0=True` 的约2000只股票需要预测

### 2. stock_prices.csv - 股票价格数据
| 字段 | 类型 | 说明 |
|------|------|------|
| Date | date | 交易日期 |
| Open | float | 开盘价 |
| High | float | 最高价 |
| Low | float | 最低价 |
| Close | float | 收盘价 |
| Volume | int | 成交量 |
| AdjustmentFactor | float | 调整因子 (股票分割/合并) |
| **Target** | float | **预测目标: t+2与t+1的调整后收盘价变化率** |

### 3. financials.csv - 财务报表数据
| 字段 | 说明 |
|------|------|
| NetSales | 净销售额 |
| OperatingProfit | 营业利润 |
| OrdinaryProfit | 经常利润 |
| Profit | 净利润 |
| EarningsPerShare | 每股收益 (EPS) |
| TotalAssets | 总资产 |
| Equity | 净资产 |
| EquityToAssetRatio | 净资产比率 |

### 4. trades.csv - 投资者交易数据 (周度)
| 字段 | 说明 |
|------|------|
| IndividualsPurchases/Sales | 个人投资者买卖额 |
| ForeignersPurchases/Sales | 外国投资者买卖额 |
| InvestmentTrustsPurchases/Sales | 投资信托买卖额 |

### 5. options.csv - 期权数据
| 字段 | 说明 |
|------|------|
| ImpliedVolatility | 隐含波动率 |
| Putcall | 看跌(1)/看涨(2) |
| StrikePrice | 行权价 |
| SettlementPrice | 结算价格 |

---

## 数据规模

| 数据集 | 记录数 | 时间范围 |
|--------|--------|----------|
| stock_prices | 约233万 | 2017-01 至 2021-12 |
| stock_list | 4,417只 | 全部股票 |
| financials | 约10万 | 季度数据 |
| trades | 约5,000 | 周度数据 |

---

## 任务需求

### 预测目标
- **Target**: 调整后收盘价从 t+1 到 t+2 的变化率
- 公式: `Target = (Close[t+2] - Close[t+1]) / Close[t+1]`

### 输出格式
| 字段 | 说明 |
|------|------|
| Date | 交易日期 |
| SecuritiesCode | 股票代码 |
| Rank | 排名 (0=预期收益最高, 1999=最低) |

### 评估指标
**Sharpe Ratio of Daily Spread Returns**

```
每天:
  - 买入预期收益最高的200只股票
  - 卖空预期收益最低的200只股票
  - 计算spread = 买入收益 - 卖空收益

整体:
  - Sharpe = mean(spread) / std(spread)
```

---

## 技术路径

### 方案1: LightGBM (当前实现)
```
特征工程 → LightGBM训练 → 交叉验证 → 排名预测
```
**特征**:
- 收益率: return_1d, return_5d, return_20d
- 波动率: volatility_5d, volatility_20d
- 移动平均: MA5, MA10, MA20, MA50
- 成交量: volume_change, volume_ma5

### 方案2: 深度学习
```
LSTM/Transformer → 时序建模 → 预测收益率
```
**适用**: 捕捉长期时序依赖

### 方案3: Ensemble
```
多模型融合 → LightGBM + XGBoost + TabNet
```
**适用**: 提升预测稳定性

---

## 当前模型性能

| 指标 | 数值 |
|------|------|
| Average Sharpe Ratio | 0.1964 |
| 5-Fold Std | 0.08 |

---

## 下一步优化方向

1. **特征工程**: 加入财务因子、市值因子、行业因子
2. **模型改进**: XGBoost、CatBoost融合
3. **时序建模**: LSTM/GRU捕捉长期模式
4. **风险管理**: 仓位控制、止损策略
