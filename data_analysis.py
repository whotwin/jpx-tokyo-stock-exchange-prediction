"""
JPX东京股票交易所 - 数据特征详解与探索性分析
============================================

竞赛目标: 预测股票的次日收益率，并按预期收益排名

================================================================================
                           特征使用指南
================================================================================

【预测目标 (Target)】
  - 变量名: Target
  - 位置: stock_prices.csv 中的 Target 列
  - 定义: 调整后收盘价从 t+1 到 t+2 的变化率
  - 注意: 这是我们要预测的值，不要用作特征!

【可用特征来源】
  1. stock_prices.csv (最重要)
     - 价格特征: Open, High, Low, Close, Volume
     - 调整因子: AdjustmentFactor
     - 预期股息: ExpectedDividend
     - 监管标志: SupervisionFlag

  2. stock_list.csv (静态信息)
     - 板块分类: Section/Products, NewMarketSegment
     - 行业分类: 33SectorName, 17SectorName
     - 市值信息: MarketCapitalization, IssuedShares

  3. financials.csv (季度/年度)
     - 盈利能力: NetSales, OperatingProfit, OrdinaryProfit, Profit
     - 估值指标: EarningsPerShare, BookValuePerShare
     - 资产状况: TotalAssets, Equity, EquityToAssetRatio
     - 预测数据: ForecastNetSales, ForecastOperatingProfit 等

  4. trades.csv (周度数据)
     - 投资者情绪: Foreigners, Individuals, InvestmentTrusts 等
     - 买卖方向: Sales (卖出) vs Purchases (买入)

  5. options.csv (期权数据)
     - 波动率指标: ImpliedVolatility, BaseVolatility
     - 价格信息: SettlementPrice, TheoreticalPrice
     - 交易量: TradingVolume, OpenInterest

【推荐特征组合 (初学者)】
  1. 基础价格特征: return_1d, return_5d, return_20d
  2. 波动率特征: volatility_5d, volatility_20d
  3. 成交量特征: volume_change, volume_ma5
  4. 技术指标: high_low_ratio, momentum
  5. 市值特征: MarketCapitalization (来自stock_list)

================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 设置绘图风格
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.size'] = 10

# 工作目录
DATA_DIR = r'd:\code\Competition\jpx-tokyo-stock-exchange-prediction'

print("=" * 70)
print("JPX东京股票交易所 - 数据探索与特征分析")
print("=" * 70)

# ==============================================================================
# 1. 数据加载
# ==============================================================================
print("\n" + "=" * 70)
print("【1】加载数据文件")
print("=" * 70)

# 加载股票列表
stock_list = pd.read_csv(f'{DATA_DIR}/stock_list.csv')
print(f"\n[OK] stock_list.csv - {len(stock_list):,} 条记录")

# 加载股价数据
stock_prices = pd.read_csv(f'{DATA_DIR}/train_files/stock_prices.csv')
print(f"[OK] stock_prices.csv - {len(stock_prices):,} 条记录")

# 加载提交样例
sample_sub = pd.read_csv(f'{DATA_DIR}/example_test_files/sample_submission.csv')
print(f"[OK] sample_submission.csv - {len(sample_sub):,} 条记录")

# 加载财务报表 (预览前几行获取结构)
financials = pd.read_csv(f'{DATA_DIR}/example_test_files/financials.csv', nrows=5)
print(f"[OK] financials.csv - 结构预览 (共需加载完整数据)")

# 加载交易数据 (预览)
trades = pd.read_csv(f'{DATA_DIR}/example_test_files/trades.csv', nrows=5)
print(f"[OK] trades.csv - 结构预览 (共需加载完整数据)")

# ==============================================================================
# 2. 完整特征说明
# ==============================================================================
print("\n" + "=" * 70)
print("【2】可用特征完整列表")
print("=" * 70)

print("""
┌─────────────────────────────────────────────────────────────────────────┐
│                         核心预测目标                                      │
├─────────────────────────────────────────────────────────────────────────┤
│  Target      : 调整后收盘价变化率 (t+2 vs t+1)  ← 预测这个值!            │
│  位置        : stock_prices.csv                                         │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                    【推荐】初级特征 (易实现,效果好)                        │
├─────────────────────────────────────────────────────────────────────────┤
│  来自 stock_prices.csv:                                                  │
│    • return_1d   : 1日收益率 (Close_t / Close_t-1 - 1)                   │
│    • return_5d   : 5日收益率                                             │
│    • return_20d  : 20日收益率                                            │
│    • volatility  : 历史波动率 (收益率标准差)                              │
│    • volume_change: 成交量变化率                                          │
│    • volume_ma5  : 5日平均成交量                                         │
│    • high_low_ratio: (High-Low)/Close  (日内波动幅度)                    │
│    • open_close_ratio: (Close-Open)/Open (日内涨跌)                      │
│                                                                         │
│  来自 stock_list.csv:                                                    │
│    • MarketCapitalization : 市值 (取对数)                                 │
│    • 33SectorName        : 行业分类 (one-hot编码)                         │
│    • NewMarketSegment    : 市场板块 (Prime/Standard)                      │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                    【进阶】中级特征 (需要数据处理)                         │
├─────────────────────────────────────────────────────────────────────────┤
│  来自 financials.csv (需按日期对齐):                                      │
│    • ROE          : 净资产收益率 (Profit / Equity)                        │
│    • PE Ratio    : 市盈率 (需计算)                                        │
│    • PB Ratio    : 市净率 (需计算)                                        │
│    • ProfitMargin : 利润率 (Profit / NetSales)                           │
│    • AssetTurnover: 资产周转率                                            │
│                                                                         │
│  来自 trades.csv (周度数据, 需按股票聚合):                                 │
│    • ForeignersRatio : 外国投资者买卖比例                                  │
│    • IndividualNet   : 个人投资者净买入                                    │
│    • InvestmentTrustNet: 投资信托净买入                                    │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                    【专家】高级特征 (复杂处理)                             │
├─────────────────────────────────────────────────────────────────────────┤
│  来自 options.csv:                                                        │
│    • ImpliedVolatility : 隐含波动率 (市场恐慌指标)                        │
│    • PutCallRatio     : 看跌/看涨期权比例                                 │
│    • VolatilitySkew   : 波动率偏斜                                        │
│                                                                         │
│  其他:                                                                    │
│    • 新闻情绪分析 (NLP)                                                  │
│    • 分析师预期数据                                                      │
│    • 管理层变动信息                                                      │
└─────────────────────────────────────────────────────────────────────────┘
""")

# ==============================================================================
# 3. 数据基本统计
# ==============================================================================
print("\n" + "=" * 70)
print("【3】数据基本统计")
print("=" * 70)

print("\n--- 股票列表 (stock_list) ---")
print(f"  总股票数: {len(stock_list):,}")
print(f"  目标股票数 (Universe0=True): {stock_list['Universe0'].sum():,}")
print(f"  覆盖行业数: {stock_list['33SectorName'].nunique()}")
print(f"  覆盖板块数: {stock_list['Section/Products'].nunique()}")

print("\n--- 股价数据 (stock_prices) ---")
print(f"  总记录数: {len(stock_prices):,}")
print(f"  唯一股票数: {stock_prices['SecuritiesCode'].nunique():,}")
print(f"  唯一日期数: {stock_prices['Date'].nunique()}")
print(f"  日期范围: {stock_prices['Date'].min()} 至 {stock_prices['Date'].max()}")

print("\n--- Target 统计 ---")
target_stats = stock_prices['Target'].describe()
print(f"  均值: {target_stats['mean']:.6f}")
print(f"  标准差: {target_stats['std']:.6f}")
print(f"  最小值: {target_stats['min']:.6f}")
print(f"  最大值: {target_stats['max']:.6f}")
print(f"  中位数: {target_stats['50%']:.6f}")

# ==============================================================================
# 4. Target分布分析
# ==============================================================================
print("\n" + "=" * 70)
print("【4】Target 分布分析")
print("=" * 70)

# 计算收益率统计
returns = stock_prices['Target'].dropna()

# 去除极端值
returns_clipped = returns.clip(returns.quantile(0.01), returns.quantile(0.99))

print(f"\n  原始数据:")
print(f"    - 正收益占比: {(returns > 0).mean()*100:.2f}%")
print(f"    - 负收益占比: {(returns < 0).mean()*100:.2f}%")
print(f"    - 零收益占比: {(returns == 0).mean()*100:.2f}%")

print(f"\n  极端值 (1%和99%分位):")
print(f"    - 1%分位: {returns.quantile(0.01):.4f}")
print(f"    - 99%分位: {returns.quantile(0.99):.4f}")

# ==============================================================================
# 5. 生成可视化
# ==============================================================================
print("\n" + "=" * 70)
print("【5】生成数据可视化...")
print("=" * 70)

fig = plt.figure(figsize=(16, 12))

# 5.1 Target分布直方图
ax1 = fig.add_subplot(2, 3, 1)
returns_clipped.hist(bins=100, ax=ax1, color='steelblue', edgecolor='white', alpha=0.7)
ax1.axvline(0, color='red', linestyle='--', linewidth=2)
ax1.axvline(returns.mean(), color='green', linestyle='--', linewidth=2, label=f'Mean: {returns.mean():.4f}')
ax1.set_title('Target Distribution (Clipped 1%-99%)', fontweight='bold')
ax1.set_xlabel('Daily Return')
ax1.set_ylabel('Frequency')
ax1.legend()

# 5.2 行业分布 (目标股票)
ax2 = fig.add_subplot(2, 3, 2)
target_stocks = stock_list[stock_list['Universe0'] == True]
sector_counts = target_stocks['33SectorName'].value_counts().head(12)
colors = plt.cm.tab20(np.linspace(0, 1, len(sector_counts)))
bars = ax2.barh(range(len(sector_counts)), sector_counts.values, color=colors)
ax2.set_yticks(range(len(sector_counts)))
ax2.set_yticklabels([s[:20] for s in sector_counts.index], fontsize=8)
ax2.set_xlabel('Number of Stocks')
ax2.set_title('Top 12 Sectors (Universe0)', fontweight='bold')
for i, v in enumerate(sector_counts.values):
    ax2.text(v + 5, i, str(v), va='center', fontsize=8)

# 5.3 市值分布 (对数)
ax3 = fig.add_subplot(2, 3, 3)
market_cap = target_stocks['MarketCapitalization'].dropna()
market_cap_log = np.log10(market_cap)
ax3.hist(market_cap_log, bins=50, color='coral', edgecolor='white', alpha=0.7)
ax3.axvline(market_cap_log.median(), color='red', linestyle='--', linewidth=2,
            label=f'Median: {10**market_cap_log.median()/1e9:.1f}B JPY')
ax3.set_title('Market Cap Distribution (Log10)', fontweight='bold')
ax3.set_xlabel('Log10(Market Cap)')
ax3.set_ylabel('Number of Stocks')
ax3.legend()

# 5.4 板块分布饼图
ax4 = fig.add_subplot(2, 3, 4)
section_counts = target_stocks['Section/Products'].value_counts()
ax4.pie(section_counts.values, labels=[s[:25] for s in section_counts.index],
         autopct='%1.1f%%', startangle=90, colors=plt.cm.Set3.colors)
ax4.set_title('Section Distribution (Universe0)', fontweight='bold')

# 5.5 成交量 vs 收益率 散点图
ax5 = fig.add_subplot(2, 3, 5)
sample_data = stock_prices[['Volume', 'Target']].dropna()
sample_data = sample_data.sample(min(10000, len(sample_data)))
volume_log = np.log10(sample_data['Volume'] + 1)
ax5.scatter(volume_log, sample_data['Target'], alpha=0.1, s=1, color='purple')
ax5.set_xlabel('Log10(Volume)')
ax5.set_ylabel('Target')
ax5.set_title('Volume vs Target', fontweight='bold')
ax5.set_xlim(volume_log.quantile(0.01), volume_log.quantile(0.99))
ax5.set_ylim(sample_data['Target'].quantile(0.01), sample_data['Target'].quantile(0.99))

# 5.6 收盘价分布 (按股票)
ax6 = fig.add_subplot(2, 3, 6)
close_prices = stock_prices.groupby('SecuritiesCode')['Close'].last().dropna()
close_log = np.log10(close_prices + 1)
ax6.hist(close_log, bins=50, color='teal', edgecolor='white', alpha=0.7)
ax6.set_title('Stock Price Distribution (Log10)', fontweight='bold')
ax6.set_xlabel('Log10(Close Price)')
ax6.set_ylabel('Number of Stocks')

plt.tight_layout()
plt.savefig(f'{DATA_DIR}/data_exploration.png', dpi=150, bbox_inches='tight')
print(f"\n[OK] 图表已保存: {DATA_DIR}/data_exploration.png")

# ==============================================================================
# 6. 特征工程示例代码
# ==============================================================================
print("\n" + "=" * 70)
print("【6】特征工程示例代码")
print("=" * 70)

FEATURE_ENGINEERING_CODE = '''
# =============================================================================
# 特征工程 - 基础版
# =============================================================================
import pandas as pd
import numpy as np

def create_features(df):
    """
    从股价数据创建基础特征
    """
    df = df.copy()
    df = df.sort_values(['SecuritiesCode', 'Date'])

    # 价格特征
    df['return_1d'] = df.groupby('SecuritiesCode')['Close'].pct_change(1)
    df['return_5d'] = df.groupby('SecuritiesCode')['Close'].pct_change(5)
    df['return_20d'] = df.groupby('SecuritiesCode')['Close'].pct_change(20)

    # 波动率特征
    df['volatility_5d'] = df.groupby('SecuritiesCode')['return_1d'].rolling(5).std().reset_index(level=0, drop=True)
    df['volatility_20d'] = df.groupby('SecuritiesCode')['return_1d'].rolling(20).std().reset_index(level=0, drop=True)

    # 成交量特征
    df['volume_change'] = df.groupby('SecuritiesCode')['Volume'].pct_change(1)
    df['volume_ma5'] = df.groupby('SecuritiesCode')['Volume'].rolling(5).mean().reset_index(level=0, drop=True)
    df['volume_ma20'] = df.groupby('SecuritiesCode')['Volume'].rolling(20).mean().reset_index(level=0, drop=True)

    # 日内特征
    df['high_low_ratio'] = (df['High'] - df['Low']) / df['Close']
    df['open_close_ratio'] = (df['Close'] - df['Open']) / df['Open']

    # 动量特征
    df['momentum_5d'] = df['Close'] / df.groupby('SecuritiesCode')['Close'].shift(5) - 1
    df['momentum_20d'] = df['Close'] / df.groupby('SecuritiesCode')['Close'].shift(20) - 1

    # 价格位置
    df['close_position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-8)

    return df

# =============================================================================
# 使用示例
# =============================================================================
# 加载数据
stock_prices = pd.read_csv('train_files/stock_prices.csv')
stock_list = pd.read_csv('stock_list.csv')

# 筛选目标股票
target_codes = stock_list[stock_list['Universe0'] == True]['SecuritiesCode']
df = stock_prices[stock_prices['SecuritiesCode'].isin(target_codes)]

# 创建特征
df = create_features(df)

# 移除NaN行用于训练
df = df.dropna()
print(f"特征创建完成! 数据量: {len(df)} 行, 特征数: {df.shape[1]}")
'''

print(FEATURE_ENGINEERING_CODE)

# ==============================================================================
# 7. 完整变量映射表
# ==============================================================================
print("\n" + "=" * 70)
print("【7】变量使用速查表")
print("=" * 70)

VARIABLE_GUIDE = """
┌──────────────────┬───────────────────────────────────────────────────────┐
│ 文件             │ 可用变量 (✓=推荐, △=进阶)                               │
├──────────────────┼───────────────────────────────────────────────────────┤
│ stock_prices.csv │ ✓ Date, Open, High, Low, Close, Volume               │
│                  │ ✓ AdjustmentFactor, SupervisionFlag                   │
│                  │ ✓ ExpectedDividend                                    │
│                  │ ★ Target (预测目标, 不要用作特征!)                     │
├──────────────────┼───────────────────────────────────────────────────────┤
│ stock_list.csv   │ ✓ SecuritiesCode (用于关联)                           │
│                  │ ✓ Section/Products, NewMarketSegment                  │
│                  │ ✓ 33SectorName, 17SectorName                          │
│                  │ ✓ MarketCapitalization (取log)                        │
│                  │ △ IssuedShares                                        │
│                  │ ✗ Universe0 (只用于筛选股票)                          │
├──────────────────┼───────────────────────────────────────────────────────┤
│ financials.csv   │ △ NetSales, OperatingProfit, OrdinaryProfit, Profit  │
│                  │ △ EarningsPerShare, BookValuePerShare                  │
│                  │ △ TotalAssets, Equity, EquityToAssetRatio              │
│                  │ △ ForecastNetSales, ForecastOperatingProfit           │
├──────────────────┼───────────────────────────────────────────────────────┤
│ trades.csv       │ △ ForeignersPurchases/Sales                          │
│                  │ △ IndividualsPurchases/Sales                          │
│                  │ △ InvestmentTrustsPurchases/Sales                      │
├──────────────────┼───────────────────────────────────────────────────────┤
│ options.csv      │ △ ImpliedVolatility, BaseVolatility                  │
│                  │ △ SettlementPrice, TheoreticalPrice                   │
│                  │ △ PutCall, StrikePrice                                │
└──────────────────┴───────────────────────────────────────────────────────┘

推荐初学者使用:
  1. 先只用 stock_prices.csv 的数据
  2. 创建: return_1d, return_5d, volatility, volume_change 等
  3. 加入 stock_list.csv 的 MarketCapitalization (log)
  4. 加入 33SectorName (one-hot编码)
"""

print(VARIABLE_GUIDE)

# ==============================================================================
# 8. 总结
# ==============================================================================
print("\n" + "=" * 70)
print("【总结】下一步操作")
print("=" * 70)

NEXT_STEPS = """
1. 使用 stock_prices.csv 作为主要数据源
2. 创建以下基础特征:
   - 收益率: return_1d, return_5d, return_20d
   - 波动率: volatility_5d, volatility_20d
   - 成交量: volume_change, volume_ma5
   - 日内特征: high_low_ratio, open_close_ratio
3. 加入 stock_list.csv 的:
   - MarketCapitalization (log转换)
   - 33SectorName (类别编码)
4. 预测目标: stock_prices.csv 中的 Target 列
5. 输出格式: 对每只股票预测收益率并排名 (Rank 0-1999)

祝你比赛顺利!
"""

print(NEXT_STEPS)

print("\n" + "=" * 70)
print("分析完成!")
print("=" * 70)
