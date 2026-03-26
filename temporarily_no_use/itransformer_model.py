"""
JPX Stock Prediction - iTransformer Model

Training Strategy:
- Use iTransformer to model all stocks simultaneously
- Inverted Transformer: treats each stock as a variate (variable)
- Attention operates across stocks (variates) rather than time steps
- This captures cross-sectional relationships between stocks

Expanding Window Training:
- Round 1: Train 2017 → Validate 2018
- Round 2: Train 2017-2018 → Validate 2019
- Round 3: Train 2017-2019 → Validate 2020
- Final: Train 2017-2020 → Predict 2021

Input: 20-day lookback window of features for all stocks
Output: 30-day forward return prediction for each stock
"""

import os
import warnings
import gc
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from iTransformer import iTransformer

warnings.filterwarnings("ignore")

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

OUTPUT_DIR = "output_itransformer"
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# Configuration
TEST_YEAR = 2021
TARGET_HORIZON = 30
TOP_K = 200
BOTTOM_K = 200
SEQ_LENGTH = 20  # 20-day lookback window

# Trading costs - same as train.py
TRADING_COST_RATE = 0.0004
SLIPPAGE_RATE = 0.0002

# iTransformer hyperparameters
ITRANSFORMER_DIM = 64
ITRANSFORMER_DEPTH = 2
ITRANSFORMER_HEADS = 4
ITRANSFORMER_EPOCHS = 10
ITRANSFORMER_BATCH_SIZE = 4  # Reduced from 16 for multi-feature (more memory)
ITRANSFORMER_LEARNING_RATE = 0.001

# Multi-feature configuration
# Set to None to use all stocks, or set to a number (e.g., 500) for memory constraints
MAX_NUM_STOCKS = None  # None = all stocks with sufficient data

# Filter top stocks by data availability
MIN_DATA_POINTS = 300  # Minimum data points per stock


def log(msg):
    print(f"[INFO] {msg}")


def to_num(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_data_sources(data_dir="train_files"):
    """Load data sources - same as train.py."""
    sources = {}

    stock_prices = pd.read_csv(os.path.join(data_dir, "stock_prices.csv"))
    stock_prices = to_num(stock_prices, ["Open", "High", "Low", "Close", "Volume", "AdjustmentFactor", "ExpectedDividend", "Target", "SupervisionFlag"])
    stock_prices["Date"] = pd.to_datetime(stock_prices["Date"])
    sources["stock_prices"] = stock_prices

    # Load stock list for market cap and sector
    if os.path.exists(os.path.join(data_dir, "stock_list.csv")):
        stock_list = pd.read_csv(os.path.join(data_dir, "stock_list.csv"),
                                 usecols=["SecuritiesCode", "MarketCapitalization", "33SectorName", "NewMarketSegment"])
        sources["stock_list"] = stock_list
    elif os.path.exists("stock_list.csv"):
        stock_list = pd.read_csv("stock_list.csv",
                                 usecols=["SecuritiesCode", "MarketCapitalization", "33SectorName", "NewMarketSegment"])
        sources["stock_list"] = stock_list

    if os.path.exists(os.path.join(data_dir, "secondary_stock_prices.csv")):
        secondary = pd.read_csv(os.path.join(data_dir, "secondary_stock_prices.csv"))
        secondary = to_num(secondary, ["Open", "High", "Low", "Close", "Volume", "AdjustmentFactor"])
        secondary["Date"] = pd.to_datetime(secondary["Date"])
        sources["secondary_stock_prices"] = secondary

    if os.path.exists(os.path.join(data_dir, "options.csv")):
        opts = pd.read_csv(os.path.join(data_dir, "options.csv"))
        opts = to_num(opts, ["ImpliedVolatility", "TradingVolume", "OpenInterest", "SettlementPrice", "BaseVolatility"])
        opts["Date"] = pd.to_datetime(opts["Date"])
        sources["options"] = opts

    if os.path.exists(os.path.join(data_dir, "trades.csv")):
        trades = pd.read_csv(os.path.join(data_dir, "trades.csv"))
        trades = to_num(trades, ["Individual", "Foreigners", "SecuritiesCos", "InvestmentTrusts", "InsuranceCos", "CityBKs", "RegionalBKs", "TrustBanks"])
        trades["Date"] = pd.to_datetime(trades["Date"])
        sources["trades"] = trades

    if os.path.exists(os.path.join(data_dir, "financials.csv")):
        financials = pd.read_csv(os.path.join(data_dir, "financials.csv"), low_memory=False)
        financials = to_num(financials, ["NetSales", "OperatingProfit", "OrdinaryProfit", "Profit", "TotalAssets", "Equity", "EquityToAssetRatio", "EarningsPerShare", "ForecastedEarningsPerShare"])
        financials["Date"] = pd.to_datetime(financials["Date"])
        sources["financials"] = financials

    log(f"Loaded data sources: {list(sources.keys())}")
    return sources


def stock_features(px, stock_list=None):
    """Extract stock-level features from price data."""
    px = px.sort_values(["SecuritiesCode", "Date"]).reset_index(drop=True)
    g = px.groupby("SecuritiesCode", sort=False)
    c = px["Close"].replace(0, np.nan)

    # Returns
    for w in [1, 2, 3, 5, 10, 20]:
        px[f"stk_ret_{w}"] = g["Close"].pct_change(w)

    # Log returns
    px["stk_logret_1"] = np.log(c).groupby(px["SecuritiesCode"]).diff(1)

    # Spread features
    px["stk_hl_spread"] = (px["High"] - px["Low"]) / c
    px["stk_oc_spread"] = (px["Close"] - px["Open"]) / px["Open"].replace(0, np.nan)

    # Volume change
    px["stk_volume_chg_1"] = g["Volume"].pct_change(1)

    # Volatility (rolling std of log returns)
    for w in [5, 10, 20]:
        px[f"stk_vol_{w}"] = g["stk_logret_1"].transform(lambda x: x.rolling(w, min_periods=w).std())

    # Rolling mean returns
    for w in [5, 10, 20]:
        px[f"stk_ret_mean_{w}"] = g["stk_ret_1"].transform(lambda x: x.rolling(w, min_periods=w).mean())

    # Moving averages
    for w in [5, 10, 20]:
        ma = g["Close"].transform(lambda x: x.rolling(w, min_periods=w).mean())
        px[f"stk_close_to_ma_{w}"] = px["Close"] / ma - 1

    # Volume to MA
    for w in [5, 10, 20]:
        vma = g["Volume"].transform(lambda x: x.rolling(w, min_periods=w).mean())
        px[f"stk_volume_to_ma_{w}"] = px["Volume"] / vma - 1

    # Skewness
    px["stk_skew_20"] = g["stk_logret_1"].transform(lambda x: x.rolling(20, min_periods=20).skew())

    # Day of week and month
    px["stk_dayofweek"] = px["Date"].dt.dayofweek
    px["stk_month"] = px["Date"].dt.month

    # Expected dividend
    px["stk_expected_dividend"] = px["ExpectedDividend"].fillna(0)

    # Supervision flag
    px["stk_supervision_flag"] = px["SupervisionFlag"].astype(str).str.lower().eq("true").astype(int)

    # Add market cap and sector features from stock_list
    if stock_list is not None:
        stock_list = stock_list.copy()
        px = px.merge(
            stock_list[["SecuritiesCode", "MarketCapitalization", "33SectorName", "NewMarketSegment"]],
            on="SecuritiesCode",
            how="left"
        )
        px["stk_mcap"] = np.log(px["MarketCapitalization"].fillna(1e8) / 1e8 + 1)
        px["stk_sector"] = pd.Categorical(px["33SectorName"]).codes
        px["stk_market_segment"] = pd.Categorical(px["NewMarketSegment"]).codes
        px = px.drop(columns=["MarketCapitalization", "33SectorName", "NewMarketSegment"], errors="ignore")

    return px


def build_30d_labels(stock_prices):
    """Build 30-day forward return labels."""
    px = stock_prices[["Date", "SecuritiesCode", "Close"]].copy()
    px["Close"] = pd.to_numeric(px["Close"], errors="coerce")
    px = px.sort_values(["SecuritiesCode", "Date"]).reset_index(drop=True)

    px["target_30d"] = px.groupby("SecuritiesCode")["Close"].shift(-30) / px["Close"] - 1.0

    return px[["Date", "SecuritiesCode", "target_30d"]]


def options_features(opts):
    """Extract options features - implied volatility."""
    if opts is None or opts.empty:
        return pd.DataFrame()

    opts = opts.sort_values("Date").reset_index(drop=True)

    if "ImpliedVolatility" in opts.columns:
        iv = opts.groupby("Date")["ImpliedVolatility"].mean().reset_index()
        iv.columns = ["Date", "iv_avg"]
        return iv

    return pd.DataFrame()


def trades_features(trades_df):
    """Extract trades features - investor trading data."""
    if trades_df is None or trades_df.empty:
        return pd.DataFrame()

    trades_df = trades_df.sort_values("Date").reset_index(drop=True)

    investor_cols = ["Individual", "Foreigners", "SecuritiesCos", "InvestmentTrusts"]
    available_cols = [c for c in investor_cols if c in trades_df.columns]

    if not available_cols:
        return pd.DataFrame()

    result = trades_df.groupby("Date")[available_cols].mean().reset_index()
    result.columns = ["Date"] + [f"trd_{c.lower()}" for c in available_cols]

    return result


def financials_features(fn):
    """Extract financials features."""
    if fn is None or fn.empty:
        return pd.DataFrame()

    fn = fn.sort_values(["SecuritiesCode", "Date"]).reset_index(drop=True)

    num_cols = ["NetSales", "OperatingProfit", "OrdinaryProfit", "Profit", "TotalAssets", "Equity"]
    available_cols = [c for c in num_cols if c in fn.columns]

    if not available_cols:
        return pd.DataFrame()

    fn[available_cols] = fn.groupby("SecuritiesCode")[available_cols].ffill()

    result = fn.groupby("SecuritiesCode", as_index=False)[available_cols].last()
    result.columns = ["SecuritiesCode"] + [f"fin_{c.lower()}" for c in available_cols]

    return result


def load_all_data():
    """Load all data sources."""
    log("Loading all data sources...")
    sources = load_data_sources("train_files")

    log("Building full feature table...")
    prices = sources["stock_prices"].copy()
    prices = prices[(prices["Date"] >= pd.to_datetime("2017-01-04")) &
                    (prices["Date"] <= pd.to_datetime("2021-12-03"))]

    df = prices[["Date", "SecuritiesCode", "Close", "Volume", "High", "Low", "Open", "ExpectedDividend", "SupervisionFlag"]].copy()

    stock_list = sources.get("stock_list", None)
    px_with_features = stock_features(prices, stock_list=stock_list)

    stock_cols = [c for c in px_with_features.columns if c.startswith("stk_")]
    df = df.merge(px_with_features[["Date", "SecuritiesCode"] + stock_cols],
                  on=["Date", "SecuritiesCode"], how="left")

    # Add options features (implied volatility)
    if "options" in sources:
        opt_feat = options_features(sources["options"])
        if not opt_feat.empty:
            df = df.merge(opt_feat, on="Date", how="left")
            opt_cols = [c for c in df.columns if c.startswith("iv_")]
            log(f"Added options features: {opt_cols}")

    # Add trades features (investor trading data)
    if "trades" in sources:
        trd_feat = trades_features(sources["trades"])
        if not trd_feat.empty:
            df = df.merge(trd_feat, on="Date", how="left")
            trd_cols = [c for c in df.columns if c.startswith("trd_")]
            log(f"Added trades features: {trd_cols}")

    # Add financials features
    if "financials" in sources:
        fin_feat = financials_features(sources["financials"])
        if not fin_feat.empty:
            df = df.merge(fin_feat, on="SecuritiesCode", how="left")
            fin_cols = [c for c in df.columns if c.startswith("fin_")]
            log(f"Added financials features: {fin_cols}")

    # Shift features by 1 to avoid leakage
    all_feature_cols = [c for c in df.columns if c not in ["Date", "SecuritiesCode", "Close", "Volume", "High", "Low", "Open", "ExpectedDividend", "SupervisionFlag", "target_30d"]]
    for col in all_feature_cols:
        if col in df.columns:
            df[col] = df.groupby("SecuritiesCode", sort=False)[col].shift(1)

    df = df.dropna(subset=["Date", "SecuritiesCode", "Close"])
    df[all_feature_cols] = df[all_feature_cols].fillna(0)

    # Build 30-day forward return labels
    labels = build_30d_labels(sources["stock_prices"])
    df = df.merge(labels, on=["Date", "SecuritiesCode"], how="left")
    df = df.sort_values(["Date", "SecuritiesCode"]).reset_index(drop=True)

    log(f"Loaded: {len(df)} rows, {len(all_feature_cols)} features")

    return df, all_feature_cols


# ============== iTransformer Data Preparation ==============

def prepare_itransformer_data(df, feature_cols, stock_codes):
    """
    Prepare data for iTransformer.

    iTransformer expects: (batch, lookback_len, num_variates)
    - lookback_len: number of time steps (days)
    - num_variates: number of stocks (each stock is a variate)

    For each time point, we have all stocks' features as different variates.
    """
    log(f"Preparing iTransformer data for {len(stock_codes)} stocks...")

    # Filter to only include selected stocks
    df_filtered = df[df["SecuritiesCode"].isin(stock_codes)].copy()
    df_filtered = df_filtered.sort_values(["Date", "SecuritiesCode"]).reset_index(drop=True)

    # Create a pivot table: dates as rows, stocks as columns
    dates = sorted(df_filtered["Date"].unique())
    log(f"Total dates: {len(dates)}")

    return df_filtered, dates


def create_itransformer_sequences(df, feature_cols, stock_codes, seq_length=20, target_col="target_30d"):
    """
    Create sequences for iTransformer with multiple features.

    iTransformer input shape: (batch, lookback_len, num_variates)
    - batch: number of samples
    - lookback_len: 20 days
    - num_variates: num_stocks × num_features (each stock-feature pair is a variate)

    Target: (batch, num_stocks) - 30-day return for each stock
    """
    log(f"Creating iTransformer sequences with seq_length={seq_length}, stocks={len(stock_codes)}...")

    df = df[df["SecuritiesCode"].isin(stock_codes)].copy()
    df = df.sort_values(["Date", "SecuritiesCode"]).reset_index(drop=True)

    # Multi-feature set: treat (stock, feature) pairs as variates
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
    # Filter to only features that exist in the dataframe
    use_features = [f for f in use_features if f in df.columns]
    num_features = len(use_features)
    if num_features == 0:
        use_features = [feature_cols[0]]
        num_features = 1

    log(f"Using {num_features} features: {use_features}")

    # Create pivot tables for features
    feature_pivots = {}
    for col in use_features:
        pivot = df.pivot(index="Date", columns="SecuritiesCode", values=col)
        feature_pivots[col] = pivot

    target_pivot = df.pivot(index="Date", columns="SecuritiesCode", values=target_col)

    dates = sorted(df["Date"].unique())

    # NOTE: Feature normalization is done in predict_with_itransformer per training window
    # to avoid data leakage. Here we just prepare raw features.

    sequences = []
    targets = []
    valid_dates = []

    # Create sequences
    for i in range(seq_length, len(dates)):
        lookback_dates = dates[i - seq_length:i]
        target_date = dates[i]

        if target_date not in target_pivot.index:
            continue

        # Stack all features
        seq_features_list = []
        for col in use_features:
            pivot = feature_pivots[col]
            values = pivot.loc[lookback_dates].values.copy()
            # Fill NaN with 0 (no normalization here - done per window)
            values = np.nan_to_num(values, 0)
            # Clip extreme values
            values = np.clip(values, -10, 10)
            seq_features_list.append(values)

        # Stack: (seq_length, num_stocks, num_features)
        seq_features_stacked = np.stack(seq_features_list, axis=-1)

        # Reshape: (seq_length, num_stocks × num_features)
        # Order: stock1_feat1, stock1_feat2, ..., stock1_featN, stock2_feat1, ...
        seq_features = seq_features_stacked.reshape(seq_length, -1)

        # Get targets: (num_stocks,)
        target_values = target_pivot.loc[target_date].values

        # Skip if too many NaN in targets
        valid_mask = ~np.isnan(target_values)
        if valid_mask.sum() < len(stock_codes) * 0.5:
            continue

        sequences.append(seq_features)
        targets.append(target_values)
        valid_dates.append(target_date)

    X = np.array(sequences, dtype=np.float32)
    y = np.array(targets, dtype=np.float32)
    dates_arr = np.array(valid_dates)

    log(f"Created sequences: X={X.shape}, y={y.shape}")
    log(f"num_variates={X.shape[2]} = {len(stock_codes)} stocks × {num_features} features")

    return X, y, dates_arr, num_features


# ============== Training Functions ==============

def train_itransformer_model(model, train_loader, epochs=10, lr=0.001):
    """Train iTransformer model."""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0

        for batch_X, batch_y in train_loader:
            # batch_X: (batch, seq_length, num_variates, num_features)
            # For iTransformer, we need: (batch, seq_length, num_variates)
            # Take first feature or reduce
            batch_X = batch_X.to(device)  # Shape: (batch, seq_length, num_variates)
            batch_y = batch_y.to(device)

            # Skip batches with NaN targets
            if torch.isnan(batch_y).any():
                continue

            optimizer.zero_grad()
            pred = model(batch_X)

            # Skip if predictions are NaN
            if torch.isnan(pred).any():
                continue

            # pred shape: (batch, pred_length, num_variates)
            # Take only the first prediction step and all variates
            pred = pred[:, 0, :]

            loss = criterion(pred, batch_y)

            if torch.isnan(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        if num_batches > 0 and (epoch + 1) % 2 == 0:
            log(f"  Epoch {epoch+1}/{epochs}, Loss: {total_loss/num_batches:.6f}")

    return model


def cross_sectional_normalize(X, num_features=1, eps=1e-8):
    """
    对每个时间步，在所有股票间做z-score归一化（横截面归一化）
    支持多特征：每个特征独立进行横截面归一化

    X: (batch, seq_len, num_variates)
       where num_variates = num_stocks × num_features

    num_features: 特征数量，用于将 num_variates 分解为 (num_stocks, num_features)
                  如果 num_features=1，使用原始行为

    这会移除市场整体影响，让模型专注于学习股票间的相对差异
    """
    batch_size, seq_len, num_variates = X.shape
    X_normalized = np.zeros_like(X)

    if num_features == 1:
        # 原始行为：在所有变量间归一化
        for t in range(seq_len):
            x_t = X[:, t, :]  # (batch, num_variates)
            mean_t = x_t.mean(axis=1, keepdims=True)  # (batch, 1)
            std_t = x_t.std(axis=1, keepdims=True)    # (batch, 1)
            std_t = np.maximum(std_t, eps)
            X_normalized[:, t, :] = (x_t - mean_t) / std_t
    else:
        # 多特征：每个特征独立在股票间归一化
        num_stocks = num_variates // num_features

        for t in range(seq_len):
            x_t = X[:, t, :]  # (batch, num_stocks × num_features)

            # 重塑为 (batch, num_stocks, num_features)
            x_t_reshaped = x_t.reshape(batch_size, num_stocks, num_features)

            # 每个特征独立在股票间归一化
            for f in range(num_features):
                x_t_feature = x_t_reshaped[:, :, f]  # (batch, num_stocks)
                mean_f = x_t_feature.mean(axis=1, keepdims=True)  # (batch, 1)
                std_f = x_t_feature.std(axis=1, keepdims=True)    # (batch, 1)
                std_f = np.maximum(std_f, eps)
                x_t_reshaped[:, :, f] = (x_t_feature - mean_f) / std_f

            # 重塑回 (batch, num_stocks × num_features)
            X_normalized[:, t, :] = x_t_reshaped.reshape(batch_size, -1)

    return X_normalized


def predict_itransformer(model, test_loader):
    """Predict with iTransformer model."""
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch_data in test_loader:
            batch_X = batch_data[0].to(device)
            batch_X = batch_X  # Shape: (batch, seq_length, num_variates)
            pred = model(batch_X)
            pred = pred[:, 0, :].cpu().numpy()
            predictions.append(pred)
    return np.concatenate(predictions, axis=0)


# ============== Evaluation Functions ==============

def calc_spread_return_sharpe(df: pd.DataFrame, portfolio_size: int = 200, toprank_weight_ratio: float = 2) -> float:
    """Kaggle official evaluation function."""
    def _calc_spread_return_per_day(df, portfolio_size, toprank_weight_ratio):
        # Handle case where we have fewer stocks than portfolio_size
        actual_size = min(len(df), portfolio_size)
        if actual_size < 10:  # Skip days with too few stocks
            return 0.0

        assert df['Rank'].min() == 0
        assert df['Rank'].max() == len(df['Rank']) - 1
        weights = np.linspace(start=toprank_weight_ratio, stop=1, num=actual_size)
        purchase = (df.sort_values(by='Rank')['Target'][:actual_size] * weights).sum() / weights.mean()
        short = (df.sort_values(by='Rank', ascending=False)['Target'][:actual_size] * weights).sum() / weights.mean()
        return purchase - short

    buf = df.groupby('Date').apply(_calc_spread_return_per_day, portfolio_size, toprank_weight_ratio)
    buf = buf[buf != 0]  # Remove zero entries
    if len(buf) == 0:
        return np.nan
    sharpe_ratio = buf.mean() / buf.std()
    return sharpe_ratio


def prepare_for_kaggle_eval(pred_df):
    """Prepare prediction DataFrame for Kaggle official evaluation."""
    df = pred_df.copy()
    df = df.sort_values(["Date", "pred"], ascending=[True, False]).reset_index(drop=True)
    df["Rank"] = df.groupby("Date").cumcount()
    df = df.rename(columns={"y_true": "Target"})
    return df


def evaluate_portfolio_kaggle(pred_df, portfolio_size=200, toprank_weight_ratio=2):
    """Evaluate using Kaggle official method."""
    if pred_df.empty:
        return {"kaggle_sharpe": np.nan}

    df = prepare_for_kaggle_eval(pred_df)
    sharpe = calc_spread_return_sharpe(df, portfolio_size=portfolio_size, toprank_weight_ratio=toprank_weight_ratio)

    return {"kaggle_sharpe": float(sharpe)}


def evaluate_portfolio(pred_df):
    """Evaluate portfolio performance with daily rebalancing."""
    if pred_df.empty:
        return {"num_days": 0, "sharpe": np.nan, "hit_ratio": np.nan, "spread": np.nan}

    pred_df = pred_df.sort_values("Date").reset_index(drop=True)
    dates = sorted(pred_df["Date"].unique())

    # Use all dates for daily rebalancing (not just monthly)
    daily_dates = dates

    daily_results = []
    prev_top = set()
    prev_bottom = set()

    for rebal_date in daily_dates:
        day_pred = pred_df[pred_df["Date"] == rebal_date].copy()
        if len(day_pred) < TOP_K + BOTTOM_K:
            continue

        sorted_pred = day_pred.sort_values("pred", ascending=False).reset_index(drop=True)

        top200 = set(sorted_pred.head(TOP_K)["SecuritiesCode"].astype(int).tolist())
        bottom200 = set(sorted_pred.tail(BOTTOM_K)["SecuritiesCode"].astype(int).tolist())

        turnover = len(top200 - prev_top) + len(bottom200 - prev_bottom)
        turnover = turnover / (TOP_K + BOTTOM_K)

        prev_top = top200
        prev_bottom = bottom200

        top_ret = sorted_pred.head(TOP_K)["y_true"].mean()
        bottom_ret = sorted_pred.tail(BOTTOM_K)["y_true"].mean()
        spread = top_ret - bottom_ret

        top_correct = (sorted_pred.head(TOP_K)["y_true"] > 0).sum()
        bottom_correct = (sorted_pred.tail(BOTTOM_K)["y_true"] < 0).sum()
        hit = (top_correct + bottom_correct) / (TOP_K + BOTTOM_K)

        daily_results.append({
            "date": rebal_date,
            "spread": spread,
            "turnover": turnover,
            "hit_ratio": hit,
        })

    if not daily_results:
        return {"num_days": 0, "sharpe": np.nan, "hit_ratio": np.nan, "spread": np.nan}

    daily_df = pd.DataFrame(daily_results)
    daily_df["spread_after_cost"] = daily_df["spread"] - daily_df["turnover"] * (TRADING_COST_RATE + SLIPPAGE_RATE) * 2

    avg_spread = daily_df["spread_after_cost"].mean()
    std_spread = daily_df["spread_after_cost"].std()
    # Annualize with sqrt(252) for daily rebalancing (252 trading days per year)
    sharpe = (avg_spread / std_spread * np.sqrt(252)) if std_spread > 0 else np.nan

    return {
        "num_days": len(daily_df),
        "sharpe": float(sharpe),
        "hit_ratio": float(daily_df["hit_ratio"].mean()),
        "spread": float(daily_df["spread"].sum()),
        "daily_df": daily_df,
    }


def evaluate_predictions(pred_df):
    """Evaluate prediction accuracy."""
    if pred_df.empty:
        return {"rmse": np.nan, "spearman": np.nan, "hit": np.nan}

    # Filter out NaN values
    pred_df = pred_df.dropna(subset=["y_true", "pred"])

    y = pred_df["y_true"].values
    p = pred_df["pred"].values

    # Filter out any remaining inf values
    valid_mask = np.isfinite(y) & np.isfinite(p)
    y = y[valid_mask]
    p = p[valid_mask]

    if len(y) == 0:
        return {"rmse": np.nan, "spearman": np.nan, "hit": np.nan}

    rmse = float(np.sqrt(mean_squared_error(y, p)))
    spearman = float(pd.Series(y).corr(pd.Series(p), method="spearman"))
    hit = float(np.mean(np.sign(y) == np.sign(p)))

    return {"rmse": rmse, "spearman": spearman, "hit": hit}


# ============== Main Training Function ==============

def predict_with_itransformer(df, feature_cols, stock_codes):
    """Run iTransformer prediction with expanding window training and multiple features."""
    log(f"Running iTransformer with expanding window training...")
    log(f"Using {len(stock_codes)} stocks")

    df = df.copy()
    df["Year"] = df["Date"].dt.year

    # Prepare sequences (now returns num_features)
    X, y, dates_arr, num_features = create_itransformer_sequences(
        df, feature_cols, stock_codes,
        seq_length=SEQ_LENGTH, target_col="target_30d"
    )

    if len(X) == 0:
        log("No sequences created!")
        return pd.DataFrame()

    num_stocks = len(stock_codes)
    num_variates = num_stocks * num_features
    log(f"Created sequences: X={X.shape}, y={y.shape}")
    log(f"num_variates={num_variates} = {num_stocks} stocks × {num_features} features")

    # NOTE: Normalization is done PER TRAINING WINDOW to avoid data leakage
    # See each training round below for normalization

    # Convert dates for filtering
    dates_pd = pd.to_datetime(dates_arr)
    years = dates_pd.year

    pred_parts = []

    # ======== Round 1: Train 2017 → Validate 2018 ========
    log("\n" + "="*50)
    log("Round 1: Train 2017 → Validate 2018")
    log("="*50)

    train_mask = years == 2017
    test_mask = years == 2018

    X_train = X[train_mask]
    y_train = y[train_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]
    dates_test = dates_arr[test_mask]

    log(f"  Train: {len(X_train):,}, Test: {len(X_test):,}")

    if len(X_train) > 0 and len(X_test) > 0:
        # Compute normalization stats using ONLY training data (avoid data leakage)
        # Feature normalization (per-variate mean/std across training time steps)
        x_mean = np.nanmean(X_train, axis=(0, 1), keepdims=True)  # (1, 1, num_variates)
        x_std = np.nanstd(X_train, axis=(0, 1), keepdims=True)
        x_std[x_std < 1e-8] = 1.0

        # Target normalization (per stock)
        y_mean = np.nanmean(y_train, axis=0, keepdims=True)  # (1, num_stocks)
        y_std = np.nanstd(y_train, axis=0, keepdims=True)
        y_std[y_std < 1e-8] = 1.0

        # Normalize training data
        X_train_norm = (X_train - x_mean) / x_std
        y_train_norm = (y_train - y_mean) / y_std

        # Normalize test data using training stats
        X_test_norm = (X_test - x_mean) / x_std
        # y_test remains unnormalized for evaluation

        # Apply cross-sectional normalization (per time step across stocks, per feature)
        X_train_norm = cross_sectional_normalize(X_train_norm, num_features=num_features)
        X_test_norm = cross_sectional_normalize(X_test_norm, num_features=num_features)

        log(f"  Feature stats: mean={np.nanmean(x_mean):.6f}, std={np.nanmean(x_std):.6f}")
        log(f"  Target stats: mean={np.nanmean(y_mean):.4f}, std={np.nanmean(y_std):.4f}")

        # Train model
        train_dataset = TensorDataset(torch.FloatTensor(X_train_norm), torch.FloatTensor(y_train_norm))
        train_loader = DataLoader(train_dataset, batch_size=ITRANSFORMER_BATCH_SIZE, shuffle=True, num_workers=0)

        model = iTransformer(
            num_variates=num_variates,  # num_stocks × num_features
            lookback_len=SEQ_LENGTH,
            depth=ITRANSFORMER_DEPTH,
            dim=ITRANSFORMER_DIM,
            heads=ITRANSFORMER_HEADS,
            pred_length=1,
            flash_attn=True
        )
        model = train_itransformer_model(model, train_loader, epochs=ITRANSFORMER_EPOCHS, lr=ITRANSFORMER_LEARNING_RATE)

        # Predict (normalized)
        test_dataset = TensorDataset(torch.FloatTensor(X_test_norm))
        test_loader = DataLoader(test_dataset, batch_size=ITRANSFORMER_BATCH_SIZE, shuffle=False, num_workers=0)
        pred_norm = predict_itransformer(model, test_loader)  # (batch, num_stocks × num_features)

        # Extract predictions: reshape to (batch, num_stocks, num_features)
        pred_reshaped = pred_norm.reshape(pred_norm.shape[0], num_stocks, num_features)

        # Use first feature's prediction for each stock (most direct signal)
        pred_per_stock = pred_reshaped[:, :, 0]  # (batch, num_stocks)

        # Denormalize predictions
        pred = pred_per_stock * y_std[0, :] + y_mean[0, :]

        # Flatten predictions and targets
        for i, date in enumerate(dates_test):
            for j, code in enumerate(stock_codes):
                if not np.isnan(y_test[i, j]):
                    pred_parts.append({
                        "Date": date,
                        "SecuritiesCode": code,
                        "y_true": y_test[i, j],
                        "pred": pred[i, j],
                        "train_year": 2017
                    })

        del model
        gc.collect()

    # ======== Round 2: Train 2017-2018 → Validate 2019 ========
    log("\n" + "="*50)
    log("Round 2: Train 2017-2018 → Validate 2019")
    log("="*50)

    train_mask = (years == 2017) | (years == 2018)
    test_mask = years == 2019

    X_train = X[train_mask]
    y_train = y[train_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]
    dates_test = dates_arr[test_mask]

    log(f"  Train: {len(X_train):,}, Test: {len(X_test):,}")

    if len(X_train) > 0 and len(X_test) > 0:
        # Compute normalization stats using ONLY training data
        x_mean = np.nanmean(X_train, axis=(0, 1), keepdims=True)
        x_std = np.nanstd(X_train, axis=(0, 1), keepdims=True)
        x_std[x_std < 1e-8] = 1.0

        y_mean = np.nanmean(y_train, axis=0, keepdims=True)
        y_std = np.nanstd(y_train, axis=0, keepdims=True)
        y_std[y_std < 1e-8] = 1.0

        # Normalize
        X_train_norm = (X_train - x_mean) / x_std
        y_train_norm = (y_train - y_mean) / y_std
        X_test_norm = (X_test - x_mean) / x_std

        # Apply cross-sectional normalization with num_features
        X_train_norm = cross_sectional_normalize(X_train_norm, num_features=num_features)
        X_test_norm = cross_sectional_normalize(X_test_norm, num_features=num_features)

        train_dataset = TensorDataset(torch.FloatTensor(X_train_norm), torch.FloatTensor(y_train_norm))
        train_loader = DataLoader(train_dataset, batch_size=ITRANSFORMER_BATCH_SIZE, shuffle=True, num_workers=0)

        model = iTransformer(
            num_variates=num_variates,
            lookback_len=SEQ_LENGTH,
            depth=ITRANSFORMER_DEPTH,
            dim=ITRANSFORMER_DIM,
            heads=ITRANSFORMER_HEADS,
            pred_length=1,
            flash_attn=True
        )
        model = train_itransformer_model(model, train_loader, epochs=ITRANSFORMER_EPOCHS, lr=ITRANSFORMER_LEARNING_RATE)

        test_dataset = TensorDataset(torch.FloatTensor(X_test_norm))
        test_loader = DataLoader(test_dataset, batch_size=ITRANSFORMER_BATCH_SIZE, shuffle=False, num_workers=0)
        pred_norm = predict_itransformer(model, test_loader)

        # Extract first feature's prediction
        pred_reshaped = pred_norm.reshape(pred_norm.shape[0], num_stocks, num_features)
        pred_per_stock = pred_reshaped[:, :, 0]
        pred = pred_per_stock * y_std[0, :] + y_mean[0, :]

        for i, date in enumerate(dates_test):
            for j, code in enumerate(stock_codes):
                if not np.isnan(y_test[i, j]):
                    pred_parts.append({
                        "Date": date,
                        "SecuritiesCode": code,
                        "y_true": y_test[i, j],
                        "pred": pred[i, j],
                        "train_year": 2018
                    })

        del model
        gc.collect()

    # ======== Round 3: Train 2017-2019 → Validate 2020 ========
    log("\n" + "="*50)
    log("Round 3: Train 2017-2019 → Validate 2020")
    log("="*50)

    train_mask = (years == 2017) | (years == 2018) | (years == 2019)
    test_mask = years == 2020

    X_train = X[train_mask]
    y_train = y[train_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]
    dates_test = dates_arr[test_mask]

    log(f"  Train: {len(X_train):,}, Test: {len(X_test):,}")

    if len(X_train) > 0 and len(X_test) > 0:
        # Compute normalization stats using ONLY training data
        x_mean = np.nanmean(X_train, axis=(0, 1), keepdims=True)
        x_std = np.nanstd(X_train, axis=(0, 1), keepdims=True)
        x_std[x_std < 1e-8] = 1.0

        y_mean = np.nanmean(y_train, axis=0, keepdims=True)
        y_std = np.nanstd(y_train, axis=0, keepdims=True)
        y_std[y_std < 1e-8] = 1.0

        # Normalize
        X_train_norm = (X_train - x_mean) / x_std
        y_train_norm = (y_train - y_mean) / y_std
        X_test_norm = (X_test - x_mean) / x_std

        # Apply cross-sectional normalization with num_features
        X_train_norm = cross_sectional_normalize(X_train_norm, num_features=num_features)
        X_test_norm = cross_sectional_normalize(X_test_norm, num_features=num_features)

        train_dataset = TensorDataset(torch.FloatTensor(X_train_norm), torch.FloatTensor(y_train_norm))
        train_loader = DataLoader(train_dataset, batch_size=ITRANSFORMER_BATCH_SIZE, shuffle=True, num_workers=0)

        model = iTransformer(
            num_variates=num_variates,
            lookback_len=SEQ_LENGTH,
            depth=ITRANSFORMER_DEPTH,
            dim=ITRANSFORMER_DIM,
            heads=ITRANSFORMER_HEADS,
            pred_length=1,
            flash_attn=True
        )
        model = train_itransformer_model(model, train_loader, epochs=ITRANSFORMER_EPOCHS, lr=ITRANSFORMER_LEARNING_RATE)

        test_dataset = TensorDataset(torch.FloatTensor(X_test_norm))
        test_loader = DataLoader(test_dataset, batch_size=ITRANSFORMER_BATCH_SIZE, shuffle=False, num_workers=0)
        pred_norm = predict_itransformer(model, test_loader)

        # Extract first feature's prediction
        pred_reshaped = pred_norm.reshape(pred_norm.shape[0], num_stocks, num_features)
        pred_per_stock = pred_reshaped[:, :, 0]
        pred = pred_per_stock * y_std[0, :] + y_mean[0, :]

        for i, date in enumerate(dates_test):
            for j, code in enumerate(stock_codes):
                if not np.isnan(y_test[i, j]):
                    pred_parts.append({
                        "Date": date,
                        "SecuritiesCode": code,
                        "y_true": y_test[i, j],
                        "pred": pred[i, j],
                        "train_year": 2019
                    })

        del model
        gc.collect()

    # ======== Final: Train 2017-2020 → Predict 2021 ========
    log("\n" + "="*50)
    log("Final: Train 2017-2020 → Predict 2021")
    log("="*50)

    train_mask = (years == 2017) | (years == 2018) | (years == 2019) | (years == 2020)
    test_mask = years == 2021

    X_train = X[train_mask]
    y_train = y[train_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]
    dates_test = dates_arr[test_mask]

    log(f"  Train: {len(X_train):,}, Test: {len(X_test):,}")

    if len(X_train) > 0 and len(X_test) > 0:
        # Compute normalization stats using ONLY training data
        x_mean = np.nanmean(X_train, axis=(0, 1), keepdims=True)
        x_std = np.nanstd(X_train, axis=(0, 1), keepdims=True)
        x_std[x_std < 1e-8] = 1.0

        y_mean = np.nanmean(y_train, axis=0, keepdims=True)
        y_std = np.nanstd(y_train, axis=0, keepdims=True)
        y_std[y_std < 1e-8] = 1.0

        # Normalize
        X_train_norm = (X_train - x_mean) / x_std
        y_train_norm = (y_train - y_mean) / y_std
        X_test_norm = (X_test - x_mean) / x_std

        # Apply cross-sectional normalization with num_features
        X_train_norm = cross_sectional_normalize(X_train_norm, num_features=num_features)
        X_test_norm = cross_sectional_normalize(X_test_norm, num_features=num_features)

        train_dataset = TensorDataset(torch.FloatTensor(X_train_norm), torch.FloatTensor(y_train_norm))
        train_loader = DataLoader(train_dataset, batch_size=ITRANSFORMER_BATCH_SIZE, shuffle=True, num_workers=0)

        model = iTransformer(
            num_variates=num_variates,
            lookback_len=SEQ_LENGTH,
            depth=ITRANSFORMER_DEPTH,
            dim=ITRANSFORMER_DIM,
            heads=ITRANSFORMER_HEADS,
            pred_length=1,
            flash_attn=True
        )
        model = train_itransformer_model(model, train_loader, epochs=ITRANSFORMER_EPOCHS, lr=ITRANSFORMER_LEARNING_RATE)

        test_dataset = TensorDataset(torch.FloatTensor(X_test_norm))
        test_loader = DataLoader(test_dataset, batch_size=ITRANSFORMER_BATCH_SIZE, shuffle=False, num_workers=0)
        pred_norm = predict_itransformer(model, test_loader)

        # Extract first feature's prediction
        pred_reshaped = pred_norm.reshape(pred_norm.shape[0], num_stocks, num_features)
        pred_per_stock = pred_reshaped[:, :, 0]
        pred = pred_per_stock * y_std[0, :] + y_mean[0, :]

        for i, date in enumerate(dates_test):
            for j, code in enumerate(stock_codes):
                if not np.isnan(y_test[i, j]):
                    pred_parts.append({
                        "Date": date,
                        "SecuritiesCode": code,
                        "y_true": y_test[i, j],
                        "pred": pred[i, j],
                        "train_year": 2020
                    })

        del model
        gc.collect()

    if pred_parts:
        out = pd.DataFrame(pred_parts).sort_values(["Date", "SecuritiesCode"]).reset_index(drop=True)
    else:
        out = pd.DataFrame()

    log(f"Total predictions: {len(out):,}")
    return out


def main():
    import time
    start_time = time.time()

    log("=" * 60)
    log("JPX 30-Day Horizon - iTransformer Model (Multi-Feature)")
    log(f"Configuration: seq_length={SEQ_LENGTH}, horizon={TARGET_HORIZON}")
    log(f"iTransformer: dim={ITRANSFORMER_DIM}, depth={ITRANSFORMER_DEPTH}, heads={ITRANSFORMER_HEADS}")
    log("=" * 60)

    # Load data
    data, feature_cols = load_all_data()

    # Filter stocks with enough data
    log("Filtering stocks with sufficient data...")
    stock_counts = data.groupby("SecuritiesCode").size()
    valid_stocks = stock_counts[stock_counts >= MIN_DATA_POINTS].index.tolist()
    log(f"Stocks with >= {MIN_DATA_POINTS} data points: {len(valid_stocks)}")

    # Optional: Limit to number of stocks for memory management
    if MAX_NUM_STOCKS and len(valid_stocks) > MAX_NUM_STOCKS:
        log(f"Limiting to top {MAX_NUM_STOCKS} stocks to reduce memory usage...")
        # Sort by data availability
        valid_stocks = valid_stocks[:MAX_NUM_STOCKS]

    log(f"Using {len(valid_stocks)} stocks with multi features for cross-sectional modeling")

    # Run iTransformer prediction
    pred = predict_with_itransformer(data, feature_cols, valid_stocks)

    if pred.empty:
        log("No predictions generated!")
        return

    # Save predictions
    pred_path = os.path.join(OUTPUT_DIR, "predictions.csv")
    pred.to_csv(pred_path, index=False)
    log(f"Saved: {pred_path}")

    # Evaluate
    port_metrics = evaluate_portfolio(pred)
    pred_metrics = evaluate_predictions(pred)
    kaggle_metrics = evaluate_portfolio_kaggle(pred)

    log("\n" + "=" * 40)
    log("PREDICTION METRICS")
    log("=" * 40)
    log(f"RMSE: {pred_metrics['rmse']:.6f}")
    log(f"Spearman: {pred_metrics['spearman']:.4f}")
    log(f"Hit Ratio: {pred_metrics['hit']:.2%}")

    log("\n" + "=" * 40)
    log("PORTFOLIO METRICS")
    log("=" * 40)
    log(f"Rebalance Days: {port_metrics['num_days']}")
    log(f"Total Spread: {port_metrics['spread']:.4%}")
    log(f"Sharpe: {port_metrics['sharpe']:.4f}")
    log(f"Hit Ratio: {port_metrics['hit_ratio']:.2%}")

    log("\n" + "=" * 40)
    log("KAGGLE OFFICIAL EVALUATION")
    log("=" * 40)
    log(f"Kaggle Sharpe: {kaggle_metrics['kaggle_sharpe']:.4f}")

    # Save metrics
    metrics = {**pred_metrics, **port_metrics, **kaggle_metrics}
    del port_metrics['daily_df']  # Remove daily_df for cleaner output
    metrics_df = pd.DataFrame([metrics])
    metrics_path = os.path.join(OUTPUT_DIR, "metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)

    total_time = time.time() - start_time
    log(f"\nDone! Time: {total_time/60:.1f} min")


if __name__ == "__main__":
    main()
