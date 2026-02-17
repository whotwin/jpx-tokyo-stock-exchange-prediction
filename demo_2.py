"""
JPX东京股票交易所预测 - 完整流程
===============================
基于LightGBM的股票收益率预测模型

功能:
    1. 数据加载与探索
    2. 特征工程
    3. 模型训练与交叉验证
    4. 预测提交

运行方式:
    python demo_2.py
"""

import warnings
import os
import gc
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 无GUI后端
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
from decimal import ROUND_HALF_UP, Decimal

warnings.filterwarnings("ignore")

# ==================== 配置 ====================
OUTPUT_DIR = "output"
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
DATA_OVERVIEW_DIR = os.path.join(PLOT_DIR, "data_overview")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(DATA_OVERVIEW_DIR, exist_ok=True)

plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


# ==================== 工具函数 ====================

def save_fig(filename, save_dir):
    """保存图表到指定目录"""
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    return filepath


def log_info(message):
    """简洁的日志输出"""
    print(f"[INFO] {message}")


# ==================== 核心函数 ====================

def adjust_price(price):
    """
    根据调整因子计算调整后收盘价
    """
    price = price.copy()
    price["Date"] = pd.to_datetime(price["Date"], format="%Y-%m-%d")

    def generate_adjusted_close(df):
        df = df.sort_values("Date", ascending=False)
        df["CumulativeAdjustmentFactor"] = df["AdjustmentFactor"].cumprod()
        df["AdjustedClose"] = (
            df["CumulativeAdjustmentFactor"] * df["Close"]
        ).map(lambda x: float(Decimal(str(x)).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)))
        df = df.sort_values("Date")
        df.loc[df["AdjustedClose"] == 0, "AdjustedClose"] = np.nan
        df["AdjustedClose"] = df["AdjustedClose"].ffill()
        return df

    price = price.sort_values(["SecuritiesCode", "Date"])
    price = price.groupby("SecuritiesCode").apply(generate_adjusted_close).reset_index(drop=True)
    return price


def create_features(df):
    """
    从价格数据创建技术指标特征
    """
    df = df.copy()
    col = 'AdjustedClose'
    periods = [5, 10, 20, 30, 50]

    for period in periods:
        df[f"Return_{period}Day"] = df.groupby("SecuritiesCode")[col].pct_change(period)
        df[f"MovingAvg_{period}Day"] = df.groupby("SecuritiesCode")[col].rolling(window=period).mean().values
        df[f"ExpMovingAvg_{period}Day"] = df.groupby("SecuritiesCode")[col].ewm(span=period, adjust=False).mean().values
        df[f"Volatility_{period}Day"] = np.log(df[col]).groupby(df["SecuritiesCode"]).diff().rolling(period).std()

    return df


def calc_spread_return_sharpe(df: pd.DataFrame, portfolio_size: int = 200, toprank_weight_ratio: float = 2) -> float:
    """
    计算Sharpe Ratio (竞赛评估指标)
    """
    def _calc_spread_return_per_day(df, portfolio_size, toprank_weight_ratio):
        weights = np.linspace(start=toprank_weight_ratio, stop=1, num=portfolio_size)
        purchase = (df.sort_values(by='Rank')['Target'][:portfolio_size] * weights).sum() / weights.mean()
        short = (df.sort_values(by='Rank', ascending=False)['Target'][:portfolio_size] * weights).sum() / weights.mean()
        return purchase - short

    buf = df.groupby('Date').apply(_calc_spread_return_per_day, portfolio_size, toprank_weight_ratio)
    sharpe_ratio = buf.mean() / buf.std()
    return sharpe_ratio


# ==================== 数据加载 ====================

def load_data():
    """加载训练数据"""
    log_info("Loading training data...")
    train = pd.read_csv("train_files/stock_prices.csv", parse_dates=['Date'])
    log_info(f"Data loaded: {len(train):,} records, Date range: {train['Date'].min().date()} to {train['Date'].max().date()}")
    return train


def load_stock_list():
    """加载股票列表"""
    stock_list = pd.read_csv("stock_list.csv")
    return stock_list


# ==================== 数据探索 ====================

def plot_market_summary(train, save_dir):
    """绘制市场概况"""
    train_date = train.Date.unique()
    returns = train.groupby('Date')['Target'].mean().mul(100)
    close_avg = train.groupby('Date')['Close'].mean()
    vol_avg = train.groupby('Date')['Volume'].mean()

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    axes[0].plot(train_date, returns, color='blue', linewidth=0.8)
    axes[0].set_ylabel('Return (%)')
    axes[0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[0].set_title('Average Daily Return')

    axes[1].plot(train_date, close_avg, color='green', linewidth=0.8)
    axes[1].set_ylabel('Price')
    axes[1].set_title('Average Closing Price')

    axes[2].plot(train_date, vol_avg, color='purple', linewidth=0.8)
    axes[2].set_ylabel('Volume')
    axes[2].set_title('Average Trading Volume')
    axes[2].set_xlabel('Date')

    fig.suptitle('Market Summary: Return, Price, Volume', fontsize=14, fontweight='bold')
    plt.tight_layout()

    save_fig("01_market_summary.png", save_dir)
    log_info("Saved: 01_market_summary.png")


def plot_target_distribution(train_df, save_dir):
    """绘制Target分布"""
    target_clipped = train_df['Target'].clip(-0.1, 0.1)

    plt.figure(figsize=(10, 6))
    plt.hist(target_clipped * 100, bins=100, color='steelblue', edgecolor='white', alpha=0.7)
    plt.axvline(x=0, color='red', linestyle='--', linewidth=2)
    plt.xlabel('Daily Return (%)')
    plt.ylabel('Frequency')
    plt.title('Target Distribution (clipped to [-10%, 10%])')
    plt.tight_layout()

    save_fig("02_target_distribution.png", save_dir)
    log_info("Saved: 02_target_distribution.png")


def plot_sector_returns(train_df, save_dir):
    """绘制各行业收益对比"""
    sector_returns = train_df.groupby('SectorName')['Target'].mean().mul(100).sort_values()

    plt.figure(figsize=(10, 8))
    colors = ['red' if x < 0 else 'green' for x in sector_returns.values]
    plt.barh(range(len(sector_returns)), sector_returns.values, color=colors, alpha=0.7)
    plt.yticks(range(len(sector_returns)), sector_returns.index, fontsize=8)
    plt.xlabel('Average Return (%)')
    plt.title('Average Return by Sector')
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    plt.tight_layout()

    save_fig("03_sector_returns.png", save_dir)
    log_info("Saved: 03_sector_returns.png")


def plot_sector_boxplot(train_df, save_dir):
    """绘制各行业Target箱线图"""
    sectors = train_df['SectorName'].value_counts().index[:12]
    data = [train_df[train_df['SectorName'] == s]['Target'].clip(-0.1, 0.1).values * 100 for s in sectors]

    plt.figure(figsize=(12, 8))
    bp = plt.boxplot(data, labels=[s[:15] for s in sectors], patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.ylabel('Daily Return (%)')
    plt.title('Target Distribution by Sector')
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    plt.tight_layout()

    save_fig("04_sector_boxplot.png", save_dir)
    log_info("Saved: 04_sector_boxplot.png")


def plot_top_bottom_stocks(train_df, save_dir):
    """绘制表现最好和最差的股票"""
    stock_returns = train_df.groupby('Name')['Target'].mean().mul(100)
    top10 = stock_returns.nlargest(10)
    bottom10 = stock_returns.nsmallest(10)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].barh(range(10), top10.values[::-1], color='green', alpha=0.7)
    axes[0].set_yticks(range(10))
    axes[0].set_yticklabels([n[:20] for n in top10.index[::-1]], fontsize=7)
    axes[0].set_xlabel('Average Return (%)')
    axes[0].set_title('Top 10 Stocks')

    axes[1].barh(range(10), bottom10.values, color='red', alpha=0.7)
    axes[1].set_yticks(range(10))
    axes[1].set_yticklabels([n[:20] for n in bottom10.index[::-1]], fontsize=7)
    axes[1].set_xlabel('Average Return (%)')
    axes[1].set_title('Bottom 10 Stocks')

    fig.suptitle('Best and Worst Performing Stocks', fontsize=14, fontweight='bold')
    plt.tight_layout()

    save_fig("05_top_bottom_stocks.png", save_dir)
    log_info("Saved: 05_top_bottom_stocks.png")


def plot_feature_importance(importance_df, save_dir):
    """绘制特征重要性"""
    importance_df = importance_df.sort_values(by='avg')

    plt.figure(figsize=(10, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(importance_df)))
    plt.barh(range(len(importance_df)), importance_df['avg'].values, color=colors)
    plt.yticks(range(len(importance_df)), importance_df.index, fontsize=8)
    plt.xlabel('Average Importance')
    plt.title('Feature Importance (LightGBM)')
    plt.tight_layout()

    save_fig("feature_importance.png", save_dir)
    log_info("Saved: feature_importance.png")


# ==================== 模型训练 ====================

def train_model(features_df):
    """
    使用LightGBM训练模型并进行时间序列交叉验证
    """
    log_info("Starting model training...")

    # 确保 Date 列存在
    if 'Date' not in features_df.columns:
        raise ValueError("Date column not found in features_df!")

    params = {
        'n_estimators': 300,
        'num_leaves': 64,
        'learning_rate': 0.1,
        'colsample_bytree': 0.9,
        'subsample': 0.8,
        'reg_alpha': 0.4,
        'metric': 'mae',
        'random_state': 21
    }

    # 保存日期列
    dates = features_df['Date'].values

    # 保留Date用于分组，但删除SecuritiesCode
    feature_cols = [c for c in features_df.columns if c not in ['Target', 'SecuritiesCode', 'SectorName', 'Date']]
    X = features_df[feature_cols].copy()
    y = features_df['Target'].values

    ts_fold = TimeSeriesSplit(n_splits=5, gap=5000)

    feature_importance = pd.DataFrame()
    sharpe_scores = []

    for fold, (train_idx, val_idx) in enumerate(ts_fold.split(X, y)):
        print(f"[Fold {fold+1}/5] Training...", end='\r')

        X_train = X.iloc[train_idx]
        y_train = y[train_idx]
        X_valid = X.iloc[val_idx]
        y_val = y[val_idx]

        # 使用保存的日期数组获取验证集的日期
        val_dates = dates[val_idx]
        val_dates_unique = np.unique(val_dates)

        # 训练模型
        gbm = LGBMRegressor(**params).fit(X_train, y_train)

        # 预测
        y_pred = gbm.predict(X_valid)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))

        # 计算排名和Sharpe
        if len(val_dates_unique) > 2:
            # 构建验证DataFrame用于计算排名
            val_df = X_valid.copy()
            val_df['Date'] = val_dates  # 添加日期列
            val_df['Target'] = y_val
            val_df['pred'] = y_pred
            val_df['Rank'] = (val_df['pred'].rank(method='first', ascending=False) - 1).astype(int)

            # 按日期分组计算Sharpe
            sharpe = calc_spread_return_sharpe(val_df)
            sharpe_scores.append(sharpe)
            print(f"[Fold {fold+1}/5] Sharpe: {sharpe:.4f}, RMSE: {rmse:.6f}")
        else:
            sharpe_scores.append(0)
            print(f"[Fold {fold+1}/5] RMSE: {rmse:.6f}")

        # 记录特征重要性
        feature_importance[f"Importance_Fold{fold}"] = gbm.feature_importances_

        del X_train, y_train, X_valid, y_val
        gc.collect()

    # 计算平均特征重要性
    feature_importance['avg'] = feature_importance.mean(axis=1)
    feature_importance.index = feature_cols

    log_info(f"Average Sharpe Ratio: {np.mean(sharpe_scores):.4f} (std: {np.std(sharpe_scores):.2f})")

    return gbm, feature_importance


# ==================== 预测提交 ====================

def make_submission(model, features_df):
    """生成预测提交文件"""
    log_info("Starting prediction submission...")

    try:
        import jpx_tokyo_market_prediction
        env = jpx_tokyo_market_prediction.make_env()
        iter_test = env.iter_test()
        log_info("Competition environment initialized")
    except ImportError:
        log_info("Competition environment not available, skipping submission")
        return

    cols = ['Date', 'SecuritiesCode', 'Open', 'High', 'Low', 'Close', 'Volume', 'AdjustmentFactor']
    historical_data = features_df[cols].copy()
    historical_data = historical_data[historical_data['Date'] >= '2021-08-01']

    counter = 0
    for (prices, *_), sample_prediction in iter_test:
        current_date = prices["Date"].iloc[0]
        print(f"Processing: {current_date}", end='\r')

        if counter == 0:
            df_price_raw = historical_data[historical_data["Date"] < current_date]

        df_price_raw = pd.concat([df_price_raw, prices[cols]]).reset_index(drop=True)
        df_price = adjust_price(df_price_raw)
        df_features = create_features(df=df_price)

        feature_cols = ['Return_5Day', 'Return_10Day', 'Return_20Day',
                        'MovingAvg_5Day', 'MovingAvg_10Day', 'MovingAvg_20Day',
                        'ExpMovingAvg_5Day', 'ExpMovingAvg_10Day', 'ExpMovingAvg_20Day',
                        'Volatility_5Day', 'Volatility_10Day', 'Volatility_20Day',
                        'Open', 'High', 'Low']

        feat = df_features[df_features.Date == current_date][feature_cols]
        feat["pred"] = model.predict(feat)
        feat["Rank"] = (feat["pred"].rank(method="first", ascending=False) - 1).astype(int)

        sample_prediction["Rank"] = feat["Rank"].values

        assert sample_prediction["Rank"].notna().all()
        assert sample_prediction["Rank"].min() == 0
        assert sample_prediction["Rank"].max() == len(sample_prediction["Rank"]) - 1

        env.predict(sample_prediction)
        counter += 1

    log_info("Submission complete!")


# ==================== 主流程 ====================

def main():
    log_info("=" * 50)
    log_info("JPX Tokyo Stock Exchange Prediction")
    log_info("=" * 50)

    # 1. 数据加载
    train = load_data()
    stock_list = load_stock_list()

    # 2. 数据探索
    train_df = train[train['Date'] > '2020-12-23'].copy()
    stock_list['SectorName'] = [i.rstrip().lower().capitalize() for i in stock_list['17SectorName']]
    stock_list['Name'] = [i.rstrip().lower().capitalize() for i in stock_list['Name']]
    train_df = train_df.merge(stock_list[['SecuritiesCode', 'Name', 'SectorName']], on='SecuritiesCode', how='left')
    log_info(f"Exploration data: {len(train_df):,} records")

    plot_market_summary(train, DATA_OVERVIEW_DIR)
    plot_target_distribution(train_df, DATA_OVERVIEW_DIR)
    plot_sector_returns(train_df, DATA_OVERVIEW_DIR)
    plot_sector_boxplot(train_df, DATA_OVERVIEW_DIR)
    plot_top_bottom_stocks(train_df, DATA_OVERVIEW_DIR)

    # 3. 特征工程
    log_info("Creating features...")
    train_clean = train.drop('ExpectedDividend', axis=1).fillna(0)
    prices = adjust_price(train_clean)
    features = create_features(df=prices)
    features = features.merge(stock_list[['SecuritiesCode', 'SectorName']], on='SecuritiesCode')
    # 保留 Date 为列，不设为 index
    features = features[features['Date'] >= '2020-12-29']
    # 删除非数值列（RowId是字符串）
    features = features.drop(columns=['RowId'], errors='ignore')
    features.fillna(0, inplace=True)

    # 4. 模型训练
    model, feature_importance = train_model(features)

    # 5. 特征重要性可视化
    plot_feature_importance(feature_importance, PLOT_DIR)

    # 6. 预测提交
    make_submission(model, features)

    log_info("=" * 50)
    log_info("Pipeline completed!")
    log_info("=" * 50)


if __name__ == "__main__":
    main()
