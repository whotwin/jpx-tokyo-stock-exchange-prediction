"""
JPX东京股票交易所 - 各数据文件详细分析
=========================================
遍历所有CSV文件，分析每个文件的结构和内容意义
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

# 设置工作目录
DATA_DIR = r'd:\code\Competition\jpx-tokyo-stock-exchange-prediction'

print("=" * 80)
print("JPX东京股票交易所 - 各数据文件详细分析")
print("=" * 80)

# ==============================================================================
# 定义每个文件的数据字典（基于数据规范）
# ==============================================================================

DATA_DICT = {
    'stock_list.csv': {
        'description': '股票基本信息列表',
        'meaning': '包含所有股票的基本信息，用于筛选目标股票',
        'columns': {
            'SecuritiesCode': '股票代码 (唯一标识)',
            'EffectiveDate': '生效日期',
            'Name': '公司名称',
            'Section/Products': '所属板块 (主板/创业板等)',
            'NewMarketSegment': '新市场板块 (Prime/Standard Market)',
            '33SectorCode': '33行业代码',
            '33SectorName': '33行业名称 (如制造业、金融业)',
            '17SectorCode': '17行业代码',
            '17SectorName': '17行业名称 (大类)',
            'NewIndexSeriesSizeCode': 'TOPIX指数规模代码',
            'NewIndexSeriesSize': 'TOPIX指数规模 (如Large/Mid/Small)',
            'TradeDate': '交易日期 (用于计算市值)',
            'Close': '收盘价 (用于计算市值)',
            'IssuedShares': '已发行股份数',
            'MarketCapitalization': '市值 (收盘价 × 股份数)',
            'Universe0': '预测目标标记 (True=需要预测)'
        }
    },
    'stock_prices.csv': {
        'description': '股票价格历史数据',
        'meaning': '包含每只股票的每日价格和成交量信息，是最重要的特征来源',
        'columns': {
            'RowId': '唯一ID (日期_股票代码)',
            'Date': '交易日期',
            'SecuritiesCode': '股票代码',
            'Open': '开盘价',
            'High': '最高价',
            'Low': '最低价',
            'Close': '收盘价',
            'Volume': '成交量',
            'AdjustmentFactor': '调整因子 (股票分割/合并时调整)',
            'ExpectedDividend': '预期股息 (除权日前记录)',
            'SupervisionFlag': '监管标志 (是否被特别处理)',
            'Target': '【预测目标】调整后收盘价变化率 (t+2 vs t+1)'
        }
    },
    'financials.csv': {
        'description': '财务报表数据',
        'meaning': '公司季度/年度财务数据，可用于基本面分析',
        'columns': {
            'DisclosureNumber': '披露文档唯一ID',
            'DateCode': '日期_代码组合ID',
            'Date': '交易日期 (用于关联股价)',
            'SecuritiesCode': '股票代码',
            'DisclosedDate': '披露日期',
            'DisclosedTime': '披露时间',
            'TypeOfDocument': '文档类型 (1Q/2Q/3Q/FY)',
            'CurrentPeriodEndDate': '会计期间结束日',
            'NetSales': '净销售额',
            'OperatingProfit': '营业利润',
            'OrdinaryProfit': '经常利润',
            'Profit': '净利润',
            'EarningsPerShare': '每股收益 (EPS)',
            'TotalAssets': '总资产',
            'Equity': '净资产',
            'EquityToAssetRatio': '净资产比率',
            'BookValuePerShare': '每股净资产',
            'ForecastNetSales': '预测净销售额',
            'ForecastOperatingProfit': '预测营业利润',
            'ForecastOrdinaryProfit': '预测经常利润',
            'ForecastProfit': '预测净利润'
        }
    },
    'trades.csv': {
        'description': '投资者类型交易数据',
        'meaning': '按投资者类型分类的周度买卖数据，反映市场情绪',
        'columns': {
            'PublishedDate': '发布日期 (通常是周四)',
            'StartDate': '周起始交易日',
            'EndDate': '周结束交易日',
            'Section': '市场板块',
            'TotalSales': '总卖出额',
            'TotalPurchases': '总买入额',
            'ProprietarySales': '自营商卖出额',
            'ProprietaryPurchases': '自营商买入额',
            'BrokerageSales': '经纪商卖出额',
            'BrokeragePurchases': '经纪商买入额',
            'IndividualsSales': '个人投资者卖出额',
            'IndividualsPurchases': '个人投资者买入额',
            'ForeignersSales': '外国投资者卖出额',
            'ForeignersPurchases': '外国投资者买入额',
            'SecuritiesCosSales': '证券公司卖出额',
            'SecuritiesCosPurchases': '证券公司买入额',
            'InvestmentTrustsSales': '投资信托卖出额',
            'InvestmentTrustsPurchases': '投资信托买入额'
        }
    },
    'options.csv': {
        'description': '期权数据',
        'meaning': '期权合约的报价和波动率数据，反映市场预期',
        'columns': {
            'DateCode': '唯一ID',
            'Date': '交易日期时间',
            'OptionsCode': '期权代码',
            'WholeDayOpen': '全天开盘价',
            'WholeDayHigh': '全天最高价',
            'WholeDayLow': '全天最低价',
            'WholeDayClose': '全天收盘价',
            'NightSessionClose': '夜盘收盘价',
            'DaySessionClose': '日盘收盘价',
            'TradingVolume': '交易量',
            'OpenInterest': '未平仓合约数',
            'TradingValue': '交易价值',
            'ContractMonth': '合约月份',
            'StrikePrice': '行权价',
            'Putcall': '看跌(1)/看涨(2)',
            'SettlementPrice': '结算价格',
            'ImpliedVolatility': '隐含波动率',
            'InterestRate': '利率',
            'DividendRate': '股息率'
        }
    },
    'secondary_stock_prices.csv': {
        'description': '次要股票价格数据',
        'meaning': '补充的股票价格数据（可能包含更多股票或更多字段）',
        'columns': '(与stock_prices.csv类似)'
    },
    'sample_submission.csv': {
        'description': '提交格式样例',
        'meaning': '竞赛提交文件的格式示例',
        'columns': {
            'Date': '交易日期',
            'SecuritiesCode': '股票代码',
            'Rank': '预测排名 (0=预期收益最高)'
        }
    }
}

# ==============================================================================
# 分析函数
# ==============================================================================

def analyze_csv_file(filepath, data_dict_entry=None):
    """分析单个CSV文件"""
    filename = os.path.basename(filepath)

    print(f"\n{'='*80}")
    print(f"📁 文件: {filename}")
    print(f"{'='*80}")

    try:
        # 读取数据
        df = pd.read_csv(filepath, nrows=5)  # 先读5行看结构
        full_df = pd.read_csv(filepath)

        print(f"\n📊 文件路径: {filepath}")
        print(f"📈 总行数: {len(full_df):,}")
        print(f"📑 总列数: {len(df.columns)}")

        # 显示列信息
        print(f"\n{'─'*80}")
        print("列信息详情:")
        print(f"{'─'*80}")
        print(f"{'列名':<30} {'数据类型':<15} {'非空数量':<12} {'示例值':<30}")
        print(f"{'─'*80}")

        for col in df.columns:
            dtype = str(df[col].dtype)
            non_null = full_df[col].notna().sum()
            sample = str(df[col].iloc[0])[:30]

            # 获取列的中文解释
            col_meaning = ""
            if data_dict_entry and 'columns' in data_dict_entry:
                col_meaning = data_dict_entry['columns'].get(col, "")

            print(f"{col:<30} {dtype:<15} {non_null:>10,} {sample:<30}")

        # 显示数据字典中的解释
        if data_dict_entry:
            print(f"\n{'─'*80}")
            print("📖 字段含义说明:")
            print(f"{'─'*80}")

            if 'description' in data_dict_entry:
                print(f"  【文件说明】{data_dict_entry['description']}")
                print(f"  【使用意义】{data_dict_entry['meaning']}")

            if 'columns' in data_dict_entry:
                print("\n  各字段含义:")
                for col, meaning in data_dict_entry['columns'].items():
                    print(f"    • {col}: {meaning}")

    except Exception as e:
        print(f"❌ 读取文件出错: {e}")

# ==============================================================================
# 遍历所有CSV文件
# ==============================================================================

# 需要分析的文件列表 (相对于DATA_DIR)
csv_files = [
    ('stock_list.csv', DATA_DICT.get('stock_list.csv', {})),
    ('train_files/stock_prices.csv', DATA_DICT.get('stock_prices.csv', {})),
    ('train_files/financials.csv', DATA_DICT.get('financials.csv', {})),
    ('train_files/trades.csv', DATA_DICT.get('trades.csv', {})),
    ('train_files/options.csv', DATA_DICT.get('options.csv', {})),
    ('train_files/secondary_stock_prices.csv', DATA_DICT.get('secondary_stock_prices.csv', {})),
    ('example_test_files/sample_submission.csv', DATA_DICT.get('sample_submission.csv', {})),
]

# 添加 supplemental_files 中的文件
supplemental_files = [
    'supplemental_files/stock_prices.csv',
    'supplemental_files/trades.csv',
    'supplemental_files/options.csv',
    'supplemental_files/financials.csv',
    'supplemental_files/secondary_stock_prices.csv',
]

# 打印总结
print("\n")
print("=" * 80)
print("📋 数据文件使用总结")
print("=" * 80)

SUMMARY = """
┌──────────────────────────────────────────────────────────────────────────────┐
│                          数据文件使用指南                                      │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  【必须使用】                                                                 │
│                                                                              │
│    stock_prices.csv (训练集)                                                  │
│      → 主要数据源，包含价格、成交量、Target                                   │
│      → 预测目标: Target列                                                   │
│      → 可用特征: Open/High/Low/Close/Volume/AdjustmentFactor               │
│                                                                              │
│  【推荐使用】                                                                 │
│                                                                              │
│    stock_list.csv                                                            │
│      → 筛选目标股票 (Universe0=True)                                        │
│      → 获取市值、行业分类信息                                                 │
│                                                                              │
│  【可选使用】                                                                 │
│                                                                              │
│    financials.csv   → 财务指标 (PE、ROE等)                                  │
│    trades.csv       → 投资者情绪 (外国/个人投资者买卖)                        │
│    options.csv     → 波动率指标 (隐含波动率)                                 │
│                                                                              │
│  【提交格式】                                                                 │
│                                                                              │
│    sample_submission.csv                                                     │
│      → 输出格式: Date + SecuritiesCode + Rank                              │
│      → Rank=0表示预期收益最高                                               │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
"""

print(SUMMARY)

print("\n" + "=" * 80)
print("开始详细分析...")
print("=" * 80)

# 分析每个文件
for relative_path, data_dict_entry in csv_files:
    filepath = os.path.join(DATA_DIR, relative_path)
    if os.path.exists(filepath):
        analyze_csv_file(filepath, data_dict_entry)
    else:
        print(f"\n⚠️  文件不存在: {filepath}")

# 分析 supplemental_files
print("\n\n")
print("=" * 80)
print("📂 补充数据文件 (supplemental_files)")
print("=" * 80)

for filename in supplemental_files:
    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath):
        # 根据文件名确定数据类型
        if 'stock_prices' in filename:
            dict_entry = DATA_DICT.get('stock_prices.csv', {})
        elif 'financials' in filename:
            dict_entry = DATA_DICT.get('financials.csv', {})
        elif 'trades' in filename:
            dict_entry = DATA_DICT.get('trades.csv', {})
        elif 'options' in filename:
            dict_entry = DATA_DICT.get('options.csv', {})
        else:
            dict_entry = {}

        dict_entry = dict_entry.copy()
        dict_entry['description'] = '补充数据 - ' + dict_entry.get('description', '')
        analyze_csv_file(filepath, dict_entry)

print("\n" + "=" * 80)
print("✅ 分析完成!")
print("=" * 80)
