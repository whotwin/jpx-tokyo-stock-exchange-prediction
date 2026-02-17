"""
JPX 赛事测试环境数据探索报告
================================

## 目录结构

### example_test_files (测试环境模拟数据)
| 文件 | 大小 | 说明 |
|------|------|------|
| sample_submission.csv | 2.3MB | 提交模板 (Date, SecuritiesCode, Rank) |
| stock_prices.csv | 298KB | 股票价格数据 (测试集) |
| secondary_stock_prices.csv | 301KB | 次级股票价格 |
| financials.csv | 5.8KB | 财务报表 |
| trades.csv | 1.2KB | 交易数据 |
| options.csv | 1.8MB | 期权数据 |

### supplemental_files (补充训练数据)
| 文件 | 大小 | 说明 |
|------|------|------|
| stock_prices.csv | 25MB | 额外价格数据 |
| secondary_stock_prices.csv | 25MB | 额外次级股票数据 |
| financials.csv | 3.4MB | 额外财务数据 |
| trades.csv | 62KB | 额外交易数据 |
| options.csv | 109MB | 额外期权数据 |


## 数据结构详解

### 1. sample_submission.csv (提交格式)
```
Date,SecuritiesCode,Rank
2021-12-06,1301,0
2021-12-06,1332,1
...
```
- **Date**: 交易日期 (如 2021-12-06)
- **SecuritiesCode**: 股票代码
- **Rank**: 排名 (0 = 预期收益最高)

### 2. stock_prices.csv (测试集)
```
RowId,Date,SecuritiesCode,Open,High,Low,Close,Volume,AdjustmentFactor,ExpectedDividend,SupervisionFlag
20211206_1301,2021-12-06,1301,2982.0,2982.0,2965.0,2971.0,8900,1.0,,False
```
- **RowId**: 格式 YYYYMMDD_CODE
- **Date**: 交易日期
- **SecuritiesCode**: 股票代码
- **Open/High/Low/Close**: OHLC价格
- **Volume**: 成交量
- **AdjustmentFactor**: 调整因子
- **ExpectedDividend**: 预期股息
- **SupervisionFlag**: 监管标记

### 3. secondary_stock_prices.csv
- 结构与 stock_prices.csv 相同
- 包含创业板股票 (如 1305, 1306, 1308...)

### 4. financials.csv (财务报表)
包含丰富的财务字段:
- **NetSales**: 净销售额
- **OperatingProfit**: 营业利润
- **OrdinaryProfit**: 经常利润
- **Profit**: 净利润
- **EarningsPerShare**: 每股收益
- **TotalAssets**: 总资产
- **Equity**: 净资产
- **EquityToAssetRatio**: 净资产比率
- 以及预测值 (Forecast*) 和分红信息

### 5. trades.csv (投资者交易)
按投资者类型分类:
- **Individuals**: 个人投资者
- **Foreigners**: 外国投资者
- **InvestmentTrusts**: 投资信托
- **SecuritiesCos**: 证券公司
- **Brokerage**: 经纪商
- **等20+类别**

### 6. options.csv (期权数据)
- **Putcall**: 1=看跌, 2=看涨
- **StrikePrice**: 行权价
- **SettlementPrice**: 结算价格
- **ImpliedVolatility**: 隐含波动率
- **TheoreticalPrice**: 理论价格
- **BaseVolatility**: 基础波动率


## 赛事环境 API

```python
import jpx_tokyo_market_prediction

env = jpx_tokyo_market_prediction.make_env()
iter_test = env.iter_test()

# iter_test 返回的数据:
for (prices, options, financials, trades, secondary_prices, sample_prediction) in iter_test:
    # prices: 当日股票价格
    # options: 当日期权数据
    # financials: 当日财务公告
    # trades: 当周交易数据
    # secondary_prices: 当日次级股票价格
    # sample_prediction: 需要填写的提交模板

    # 预测并填入 Rank
    sample_prediction['Rank'] = ...

    # 提交
    env.predict(sample_prediction)
```


## 测试日期范围

从 sample_submission.csv 查看:
- 起始日期: 2021-12-06
- 包含约 2000 只股票/天
"""

import os

# 获取实际日期信息
sample_path = "example_test_files/sample_submission.csv"
if os.path.exists(sample_path):
    with open(sample_path, 'r') as f:
        lines = f.readlines()[:5]
        print("=== sample_submission.csv 样例 ===")
        for line in lines:
            print(line.strip())
        print(f"\n总行数: {sum(1 for _ in open(sample_path))}")

# 获取 stock_prices 日期
prices_path = "example_test_files/stock_prices.csv"
if os.path.exists(prices_path):
    with open(prices_path, 'r') as f:
        lines = f.readlines()[:10]
        print("\n=== stock_prices.csv (测试集) 样例 ===")
        for line in lines:
            print(line.strip())
