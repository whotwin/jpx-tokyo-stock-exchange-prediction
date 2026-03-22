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

warnings.filterwarnings("ignore")

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

OUTPUT_DIR = "output_lstm"
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

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

# Configuration
TEST_YEAR = 2021
ROLL_TRAIN_YEARS = 2
TARGET_HORIZON = 30
TOP_K = 200
BOTTOM_K = 200
SEQ_LENGTH = 90  # 60-day lookback window with raw OHLCV data

# Trading costs - same as train.py
TRADING_COST_RATE = 0.0004
SLIPPAGE_RATE = 0.0002

# LSTM hyperparameters
LSTM_HIDDEN_SIZE = 64
LSTM_NUM_LAYERS = 2
LSTM_DROPOUT = 0.2
LSTM_EPOCHS = 50
LSTM_BATCH_SIZE = 1024
LSTM_LEARNING_RATE = 0.005


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
    """Extract stock-level features from price data - same as train.py."""
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


def options_features(opts):
    """Extract options features - same as train.py."""
    if opts is None or opts.empty:
        return pd.DataFrame()

    opts = opts.sort_values("Date").reset_index(drop=True)

    if "ImpliedVolatility" in opts.columns:
        iv = opts.groupby("Date")["ImpliedVolatility"].mean().reset_index()
        iv.columns = ["Date", "iv_avg"]
        return iv

    return pd.DataFrame()


def trades_features(trades_df):
    """Extract trades features - same as train.py."""
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
    """Extract financials features - same as train.py."""
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


def build_feature_table(sources, start_date=None, end_date=None):
    """Build full feature table - same as train.py."""
    prices = sources["stock_prices"].copy()

    if start_date:
        prices = prices[prices["Date"] >= pd.to_datetime(start_date)]
    if end_date:
        prices = prices[prices["Date"] <= pd.to_datetime(end_date)]

    df = prices[["Date", "SecuritiesCode", "Close", "Volume", "High", "Low", "Open", "ExpectedDividend", "SupervisionFlag"]].copy()

    stock_list = sources.get("stock_list", None)
    px_with_features = stock_features(prices, stock_list=stock_list)

    stock_cols = [c for c in px_with_features.columns if c.startswith("stk_")]
    df = df.merge(px_with_features[["Date", "SecuritiesCode"] + stock_cols],
                  on=["Date", "SecuritiesCode"], how="left")

    if "options" in sources:
        opt_feat = options_features(sources["options"])
        if not opt_feat.empty:
            df = df.merge(opt_feat, on="Date", how="left")

    if "trades" in sources:
        trd_feat = trades_features(sources["trades"])
        if not trd_feat.empty:
            df = df.merge(trd_feat, on="Date", how="left")

    if "financials" in sources:
        fin_feat = financials_features(sources["financials"])
        if not fin_feat.empty:
            df = df.merge(fin_feat, on="SecuritiesCode", how="left")

    non_feature_cols = ["Date", "SecuritiesCode", "Close", "Volume", "High", "Low", "Open", "ExpectedDividend", "SupervisionFlag"]
    feature_cols = [c for c in df.columns if c not in non_feature_cols]

    # Shift features by 1 to avoid leakage
    for col in feature_cols:
        df[col] = df.groupby("SecuritiesCode", sort=False)[col].shift(1)

    df = df.dropna(subset=["Date", "SecuritiesCode", "Close"])
    df[feature_cols] = df[feature_cols].fillna(0)

    log(f"Built feature table: {len(df)} rows, {len(feature_cols)} features")

    return df, feature_cols


def build_30d_labels(stock_prices):
    """Build 30-day forward return labels - same as train.py."""
    px = stock_prices[["Date", "SecuritiesCode", "Close"]].copy()
    px["Close"] = pd.to_numeric(px["Close"], errors="coerce")
    px = px.sort_values(["SecuritiesCode", "Date"]).reset_index(drop=True)

    px["target_30d"] = px.groupby("SecuritiesCode")["Close"].shift(-30) / px["Close"] - 1.0

    return px[["Date", "SecuritiesCode", "target_30d"]]


def load_all_data():
    """Load all data sources - same as train.py."""
    log("Loading all data sources...")
    sources = load_data_sources("train_files")

    log("Building full feature table...")
    full_df, feature_cols = build_feature_table(
        sources=sources,
        start_date="2017-01-04",
        end_date="2021-12-03",
    )

    labels = build_30d_labels(sources["stock_prices"])
    full_df = full_df.merge(labels, on=["Date", "SecuritiesCode"], how="left")
    full_df = full_df.sort_values(["Date", "SecuritiesCode"]).reset_index(drop=True)
    log(f"Loaded: {len(full_df)} rows")

    return full_df, feature_cols


def load_all_data_for_lstm():
    """
    Load all data sources with advanced features for LSTM.

    This function builds a comprehensive feature set including:
    1. Price momentum features (returns at various horizons)
    2. Intraday price structure features (spreads, MA ratios)
    3. Volatility features (historical volatility)
    4. Volume features (volume changes, volume ratios)
    5. Cross-sectional features (rankings within each time step)
    6. External features (trades, financials, options)

    This is different from raw OHLCV data - LSTM learns from these engineered features.
    """
    log("Loading data with advanced features for LSTM...")

    # Load all data sources
    sources = load_data_sources("train_files")

    # Build full feature table (same as load_all_data but keep all features)
    log("Building full feature table with advanced features...")
    full_df, feature_cols = build_feature_table(
        sources=sources,
        start_date="2017-01-04",
        end_date="2021-12-03",
    )

    # Build 30-day forward return labels
    labels = build_30d_labels(sources["stock_prices"])
    full_df = full_df.merge(labels, on=["Date", "SecuritiesCode"], how="left")
    full_df = full_df.sort_values(["Date", "SecuritiesCode"]).reset_index(drop=True)

    # Add cross-sectional features (rank within each time step)
    log("Adding cross-sectional features (rank within each time step)...")
    full_df = add_cross_sectional_features(full_df)

    log(f"Loaded: {len(full_df)} rows, {len(feature_cols)} features")

    return full_df, feature_cols


def add_cross_sectional_features(df):
    """
    Add cross-sectional features: rank within each time step.

    For each time step (Date), compute the rank of each stock for key features.
    This captures relative position within the market at each point in time.
    """
    # Key features for cross-sectional ranking
    rank_features = ["stk_ret_1", "stk_vol_20", "stk_volume_chg_1"]

    for feat in rank_features:
        if feat in df.columns:
            # Compute rank within each date (0-1 scale)
            df[f"{feat}_rank"] = df.groupby("Date")[feat].rank(pct=True)
            log(f"Added cross-sectional feature: {feat}_rank")

    return df


# Define advanced features for LSTM (similar to itransformer but optimized)
LSTM_FEATURES = [
    # 价格动量特征 (7)
    "stk_ret_1", "stk_ret_2", "stk_ret_3", "stk_ret_5", "stk_ret_10", "stk_ret_20",
    "stk_logret_1",

    # 日内价格结构特征 (4)
    "stk_hl_spread", "stk_oc_spread",
    "stk_close_to_ma_5", "stk_close_to_ma_20",

    # 波动率特征 (3)
    "stk_vol_5", "stk_vol_10", "stk_vol_20",

    # 成交量特征 (4)
    "stk_volume_chg_1",
    "stk_volume_to_ma_5", "stk_volume_to_ma_10", "stk_volume_to_ma_20",

    # 横截面特征 (3)
    "stk_ret_1_rank", "stk_vol_20_rank", "stk_volume_chg_1_rank",

    # 期权特征 (1)
    "iv_avg",

    # 交易流向特征 (4)
    "trd_individual", "trd_foreigners", "trd_securitiescos", "trd_investmenttrusts",

    # 财务/市值特征 (3)
    "stk_mcap", "stk_sector", "stk_market_segment",
]


def load_dataset():
    """
    Load data for LSTM with advanced features.
    Uses comprehensive feature set including price momentum, volatility, volume, and cross-sectional features.
    """
    log("Loading data with advanced features for LSTM...")

    # Load all data with advanced features (ignore feature_cols from this function)
    full_df, _ = load_all_data_for_lstm()

    # Filter to use only the specified features
    available_features = [f for f in LSTM_FEATURES if f in full_df.columns]
    missing_features = [f for f in LSTM_FEATURES if f not in full_df.columns]

    if missing_features:
        log(f"Warning: {len(missing_features)} features not found: {missing_features[:5]}...")

    log(f"Using {len(available_features)} features out of {len(LSTM_FEATURES)} requested")

    target_col = "target_30d"
    data = full_df[["Date", "SecuritiesCode"] + available_features + [target_col]].copy()

    log(f"Loaded: {len(data)} rows with {len(available_features)} advanced features")
    log(f"Features: {available_features[:5]}... (and {len(available_features)-5} more)")
    log(f"Target: {target_col}")

    return data, available_features, target_col


def build_30d_labels_raw(df):
    """Build 30-day forward return labels from raw Close prices."""
    df = df.sort_values(["SecuritiesCode", "Date"]).reset_index(drop=True)
    df["target_30d"] = df.groupby("SecuritiesCode")["Close"].shift(-30) / df["Close"] - 1.0
    return df


# ============== LSTM Model ==============

class LSTMModel(nn.Module):
    """LSTM model for stock prediction."""
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        # Use the last time step
        out = self.fc(lstm_out[:, -1, :])
        return out.squeeze(-1)


def normalize_features_per_stock(df, feature_cols):
    """Normalize features per stock using rolling window to avoid look-ahead bias."""
    log("Normalizing features per stock...")

    df = df.sort_values(["SecuritiesCode", "Date"]).reset_index(drop=True)

    # Use expanding window normalization (shift by 1 to avoid leakage)
    for col in tqdm(feature_cols, desc="Normalizing"):
        # Calculate expanding mean and std (shifted by 1 to avoid look-ahead)
        mean_vals = df.groupby("SecuritiesCode")[col].transform(
            lambda x: x.shift(1).expanding().mean()
        )
        std_vals = df.groupby("SecuritiesCode")[col].transform(
            lambda x: x.shift(1).expanding().std()
        )

        # Normalize: (x - mean) / (std + eps)
        df[f"{col}_norm"] = (df[col] - mean_vals) / (std_vals + 1e-8)

    norm_cols = [f"{col}_norm" for col in feature_cols]
    return df, norm_cols


def cross_sectional_normalize(df, feature_cols, eps=1e-8):
    """
    Cross-sectional normalization: normalize across stocks at each time step

    For each time step (Date), normalize features across all stocks.
    This removes market-wide effects and focuses on relative differences between stocks.

    Args:
        df: DataFrame with columns [Date, SecuritiesCode, features...]
        feature_cols: list of feature column names to normalize
        eps: small value to prevent division by zero

    Returns:
        df: DataFrame with normalized features (overwrites original columns)
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
        df[f'{col}_std'] = df[f'{col}_std'].replace(0, 1.0)  # Replace 0 with 1
        df[f'{col}_std'] = df[f'{col}_std'].clip(lower=eps)  # Ensure minimum value

        # Normalize: (x - mean) / std
        df[col] = (df[col] - df[f'{col}_mean']) / df[f'{col}_std']

        # Clip extreme values
        df[col] = df[col].clip(-10, 10)

        # Drop temporary columns
        df = df.drop(columns=[f'{col}_mean', f'{col}_std'])

    log("Cross-sectional normalization completed.")
    return df


def create_sequences(df, feature_cols, seq_length=60, target_col="target_30d"):
    """
    Create sequences for LSTM from raw OHLCV data.
    Each sample uses seq_length days of raw OHLCV data to predict 30-day forward return.

    Input: (batch, seq_length, 5) - 5 raw features (Open, High, Low, Close, Volume)
    """
    log(f"Creating sequences with seq_length={seq_length} from raw OHLCV data...")

    df = df.sort_values(["SecuritiesCode", "Date"]).reset_index(drop=True)

    sequences = []
    targets = []
    dates = []
    codes = []

    # Get unique stocks
    stocks = df["SecuritiesCode"].unique()

    for code in tqdm(stocks, desc="Building sequences"):
        stock_data = df[df["SecuritiesCode"] == code].sort_values("Date").reset_index(drop=True)

        if len(stock_data) < seq_length + TARGET_HORIZON + 1:
            continue

        # Get raw feature values (OHLCV)
        feature_values = stock_data[feature_cols].values
        target_values = stock_data[target_col].values
        date_values = stock_data["Date"].values

        # Create sequences
        for i in range(seq_length, len(stock_data) - TARGET_HORIZON):
            seq_features = feature_values[i - seq_length:i]

            # Target: 30-day forward return (target_30d is already a forward return, no need to add offset)
            target = target_values[i]

            # Skip if any NaN in sequence or target
            if np.isnan(seq_features).any() or np.isnan(target):
                continue

            sequences.append(seq_features)
            targets.append(target)
            dates.append(date_values[i])
            codes.append(code)

    X = np.array(sequences, dtype=np.float32)
    y = np.array(targets, dtype=np.float32)
    dates_arr = np.array(dates)
    codes_arr = np.array(codes)

    log(f"Created sequences: X={X.shape}, y={y.shape}")
    log(f"Input shape: ({seq_length}, {len(feature_cols)}) = Open, High, Low, Close, Volume")

    return X, y, dates_arr, codes_arr


def train_lstm_model(model, train_loader, epochs=10, lr=0.001):
    """Train LSTM model."""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs
    )

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            # Skip batches with NaN targets
            if torch.isnan(batch_y).any():
                continue

            optimizer.zero_grad()
            pred = model(batch_X)

            # Skip if predictions are NaN
            if torch.isnan(pred).any():
                continue

            loss = criterion(pred, batch_y)

            # Skip if loss is NaN
            if torch.isnan(loss):
                continue

            loss.backward()

            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        scheduler.step()
        if num_batches > 0 and (epoch + 1) % 2 == 0:
            log(f"  Epoch {epoch+1}/{epochs}, Loss: {total_loss/num_batches:.6f}")

    return model


def predict_lstm(model, test_loader):
    """Predict with LSTM model."""
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch_data in test_loader:
            batch_X = batch_data[0].to(device)
            pred = model(batch_X)
            predictions.extend(pred.cpu().numpy())
    return np.array(predictions)


# ============== Evaluation Functions (same as train.py) ==============

# ============== Kaggle Official Evaluation ==============

def calc_spread_return_sharpe(df: pd.DataFrame, portfolio_size: int = 200, toprank_weight_ratio: float = 2) -> float:
    """
    Kaggle official evaluation function.
    Calculates the spread return Sharpe ratio.

    Args:
        df (pd.DataFrame): Must contain 'Date', 'Rank', 'Target' columns
        portfolio_size (int): # of equities to buy/sell
        toprank_weight_ratio (float): the relative weight of the most highly ranked stock compared to the least.
    Returns:
        (float): sharpe ratio
    """
    def _calc_spread_return_per_day(df, portfolio_size, toprank_weight_ratio):
        """
        Args:
            df (pd.DataFrame): predicted results
            portfolio_size (int): # of equities to buy/sell
            toprank_weight_ratio (float): the relative weight of the most highly ranked stock compared to the least.
        Returns:
            (float): spread return
        """
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
    """
    Prepare prediction DataFrame for Kaggle official evaluation.
    Adds 'Rank' column based on prediction values (higher pred = lower rank = better).
    """
    df = pred_df.copy()
    df = df.sort_values(["Date", "pred"], ascending=[True, False]).reset_index(drop=True)

    # Add Rank: 0 is best (highest prediction), higher is worse
    df["Rank"] = df.groupby("Date").cumcount()

    # Rename y_true to Target for Kaggle format
    df = df.rename(columns={"y_true": "Target"})

    return df


def evaluate_portfolio_kaggle(pred_df, portfolio_size=200, toprank_weight_ratio=2):
    """
    Evaluate using Kaggle official method.
    """
    if pred_df.empty:
        return {"kaggle_sharpe": np.nan}

    # Prepare data with Rank column
    df = prepare_for_kaggle_eval(pred_df)

    # Calculate Kaggle Sharpe
    sharpe = calc_spread_return_sharpe(df, portfolio_size=portfolio_size, toprank_weight_ratio=toprank_weight_ratio)

    return {
        "kaggle_sharpe": float(sharpe),
    }


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

    # Annualize with sqrt(252) for daily rebalancing
    sharpe = (avg_spread / std_spread * np.sqrt(252)) if std_spread > 0 else np.nan

    return {
        "num_days": len(daily_df),
        "sharpe": float(sharpe),
        "hit_ratio": float(daily_df["hit_ratio"].mean()),
        "spread": float(daily_df["spread"].sum()),
        "daily_df": daily_df,
    }


def evaluate_predictions(pred_df):
    """Evaluate prediction accuracy - same as train.py."""
    if pred_df.empty:
        return {"rmse": np.nan, "spearman": np.nan, "hit": np.nan}

    y = pred_df["y_true"].values
    p = pred_df["pred"].values

    rmse = float(np.sqrt(mean_squared_error(y, p)))
    spearman = float(pd.Series(y).corr(pd.Series(p), method="spearman"))
    hit = float(np.mean(np.sign(y) == np.sign(p)))

    return {"rmse": rmse, "spearman": spearman, "hit": hit}


# ============== Main Training Function ==============

def predict_with_lstm(df, feature_cols, target_col):
    """Run LSTM prediction with expanding window training."""
    log(f"Running LSTM with expanding window training...")

    df = df.copy()
    df["Year"] = df["Date"].dt.year

    # Expanding window approach:
    # 1. Create ALL sequences from full data
    # 2. Filter by prediction year for train/test split
    # This avoids losing target values at year boundaries

    # Step 1: Apply cross-sectional normalization across stocks at each time step
    # This removes market-wide effects and focuses on relative differences between stocks
    log("Step 1: Applying cross-sectional normalization (normalize across stocks at each time step)...")
    df_norm = cross_sectional_normalize(df, feature_cols)

    # Step 2: Create sequences from ALL normalized data
    log("Step 2: Creating sequences from all data...")
    X, y, dates_arr, codes_arr = create_sequences(
        df_norm, feature_cols, seq_length=SEQ_LENGTH, target_col=target_col
    )

    if len(X) == 0:
        log("No sequences created!")
        return pd.DataFrame()

    log(f"Created sequences: X={X.shape}")

    # Convert dates for filtering
    dates_pd = pd.to_datetime(dates_arr)
    years = dates_pd.year

    pred_parts = []

    # ======== 2018 Prediction (Validation) ========
    log("Training for 2018 prediction...")
    train_mask = years < 2018
    test_mask = years == 2018

    X_train = X[train_mask]
    y_train = y[train_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]
    dates_test = dates_arr[test_mask]
    codes_test = codes_arr[test_mask]

    log(f"  2018: Train {len(X_train):,}, Test {len(X_test):,}")

    if len(X_train) > 0 and len(X_test) > 0:
        # Sample training data
        max_train_samples = 300000
        if len(X_train) > max_train_samples:
            step = len(X_train) // max_train_samples
            indices = np.arange(0, len(X_train), step)[:max_train_samples]
            X_train_sampled = torch.FloatTensor(X_train[indices])
            y_train_sampled = torch.FloatTensor(y_train[indices])
        else:
            X_train_sampled = torch.FloatTensor(X_train)
            y_train_sampled = torch.FloatTensor(y_train)

        # Train model
        train_dataset = TensorDataset(X_train_sampled, y_train_sampled)
        train_loader = DataLoader(train_dataset, batch_size=LSTM_BATCH_SIZE, shuffle=True, num_workers=0)

        input_size = X_train.shape[2]
        model = LSTMModel(
            input_size=input_size,
            hidden_size=LSTM_HIDDEN_SIZE,
            num_layers=LSTM_NUM_LAYERS,
            dropout=LSTM_DROPOUT
        )
        model = train_lstm_model(model, train_loader, epochs=LSTM_EPOCHS, lr=LSTM_LEARNING_RATE)

        # Predict
        test_dataset = TensorDataset(torch.FloatTensor(X_test))
        test_loader = DataLoader(test_dataset, batch_size=LSTM_BATCH_SIZE, shuffle=False, num_workers=0)
        pred = predict_lstm(model, test_loader)

        out = pd.DataFrame({
            "Date": dates_test,
            "SecuritiesCode": codes_test,
            "y_true": y_test,
            "pred": pred,
            "train_year": 2017
        })
        pred_parts.append(out)

        del model
        gc.collect()

    # ======== 2019 Prediction (Validation) ========
    log("Training for 2019 prediction...")
    train_mask = years < 2019
    test_mask = years == 2019

    X_train = X[train_mask]
    y_train = y[train_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]
    dates_test = dates_arr[test_mask]
    codes_test = codes_arr[test_mask]

    log(f"  2019: Train {len(X_train):,}, Test {len(X_test):,}")

    if len(X_train) > 0 and len(X_test) > 0:
        # Sample training data
        max_train_samples = 300000
        if len(X_train) > max_train_samples:
            step = len(X_train) // max_train_samples
            indices = np.arange(0, len(X_train), step)[:max_train_samples]
            X_train_sampled = torch.FloatTensor(X_train[indices])
            y_train_sampled = torch.FloatTensor(y_train[indices])
        else:
            X_train_sampled = torch.FloatTensor(X_train)
            y_train_sampled = torch.FloatTensor(y_train)

        # Train model
        train_dataset = TensorDataset(X_train_sampled, y_train_sampled)
        train_loader = DataLoader(train_dataset, batch_size=LSTM_BATCH_SIZE, shuffle=True, num_workers=0)

        input_size = X_train.shape[2]
        model = LSTMModel(
            input_size=input_size,
            hidden_size=LSTM_HIDDEN_SIZE,
            num_layers=LSTM_NUM_LAYERS,
            dropout=LSTM_DROPOUT
        )
        model = train_lstm_model(model, train_loader, epochs=LSTM_EPOCHS, lr=LSTM_LEARNING_RATE)

        # Predict
        test_dataset = TensorDataset(torch.FloatTensor(X_test))
        test_loader = DataLoader(test_dataset, batch_size=LSTM_BATCH_SIZE, shuffle=False, num_workers=0)
        pred = predict_lstm(model, test_loader)

        out = pd.DataFrame({
            "Date": dates_test,
            "SecuritiesCode": codes_test,
            "y_true": y_test,
            "pred": pred,
            "train_year": 2018
        })
        pred_parts.append(out)

        del model
        gc.collect()

    # ======== 2020 Prediction (Validation) ========
    log("Training for 2020 prediction...")
    train_mask = years < 2020
    test_mask = years == 2020

    X_train = X[train_mask]
    y_train = y[train_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]
    dates_test = dates_arr[test_mask]
    codes_test = codes_arr[test_mask]

    log(f"  2020: Train {len(X_train):,}, Test {len(X_test):,}")

    if len(X_train) > 0 and len(X_test) > 0:
        # Sample training data
        max_train_samples = 300000
        if len(X_train) > max_train_samples:
            step = len(X_train) // max_train_samples
            indices = np.arange(0, len(X_train), step)[:max_train_samples]
            X_train_sampled = torch.FloatTensor(X_train[indices])
            y_train_sampled = torch.FloatTensor(y_train[indices])
        else:
            X_train_sampled = torch.FloatTensor(X_train)
            y_train_sampled = torch.FloatTensor(y_train)

        # Train model
        train_dataset = TensorDataset(X_train_sampled, y_train_sampled)
        train_loader = DataLoader(train_dataset, batch_size=LSTM_BATCH_SIZE, shuffle=True, num_workers=0)

        input_size = X_train.shape[2]
        model = LSTMModel(
            input_size=input_size,
            hidden_size=LSTM_HIDDEN_SIZE,
            num_layers=LSTM_NUM_LAYERS,
            dropout=LSTM_DROPOUT
        )
        model = train_lstm_model(model, train_loader, epochs=LSTM_EPOCHS, lr=LSTM_LEARNING_RATE)

        # Predict
        test_dataset = TensorDataset(torch.FloatTensor(X_test))
        test_loader = DataLoader(test_dataset, batch_size=LSTM_BATCH_SIZE, shuffle=False, num_workers=0)
        pred = predict_lstm(model, test_loader)

        out = pd.DataFrame({
            "Date": dates_test,
            "SecuritiesCode": codes_test,
            "y_true": y_test,
            "pred": pred,
            "train_year": 2019
        })
        pred_parts.append(out)

        del model
        gc.collect()

    # ======== 2021 Prediction (Test) ========
    log("Training for 2021 prediction...")
    train_mask = years < 2021
    test_mask = years == 2021

    X_train = X[train_mask]
    y_train = y[train_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]
    dates_test = dates_arr[test_mask]
    codes_test = codes_arr[test_mask]

    log(f"  2021: Train {len(X_train):,}, Test {len(X_test):,}")

    if len(X_train) > 0 and len(X_test) > 0:
        # Sample training data
        max_train_samples = 300000
        if len(X_train) > max_train_samples:
            step = len(X_train) // max_train_samples
            indices = np.arange(0, len(X_train), step)[:max_train_samples]
            X_train_sampled = torch.FloatTensor(X_train[indices])
            y_train_sampled = torch.FloatTensor(y_train[indices])
        else:
            X_train_sampled = torch.FloatTensor(X_train)
            y_train_sampled = torch.FloatTensor(y_train)

        # Train model
        train_dataset = TensorDataset(X_train_sampled, y_train_sampled)
        train_loader = DataLoader(train_dataset, batch_size=LSTM_BATCH_SIZE, shuffle=True, num_workers=0)

        input_size = X_train.shape[2]
        model = LSTMModel(
            input_size=input_size,
            hidden_size=LSTM_HIDDEN_SIZE,
            num_layers=LSTM_NUM_LAYERS,
            dropout=LSTM_DROPOUT
        )
        model = train_lstm_model(model, train_loader, epochs=LSTM_EPOCHS, lr=LSTM_LEARNING_RATE)

        # Predict
        test_dataset = TensorDataset(torch.FloatTensor(X_test))
        test_loader = DataLoader(test_dataset, batch_size=LSTM_BATCH_SIZE, shuffle=False, num_workers=0)
        pred = predict_lstm(model, test_loader)

        out = pd.DataFrame({
            "Date": dates_test,
            "SecuritiesCode": codes_test,
            "y_true": y_test,
            "pred": pred,
            "train_year": 2020
        })
        pred_parts.append(out)

        del model
        gc.collect()

    if pred_parts:
        out = pd.concat(pred_parts, ignore_index=True).sort_values(["Date", "SecuritiesCode"]).reset_index(drop=True)
    else:
        out = pd.DataFrame()

    log(f"Total predictions: {len(out):,}")
    return out


def main():
    import time
    start_time = time.time()

    log("=" * 60)
    log("JPX 30-Day Horizon - LSTM Model (41 Engineered Features)")
    log(f"Configuration: seq_length={SEQ_LENGTH}, horizon={TARGET_HORIZON}")
    log(f"Input: 60-day window of {len(USE_FEATURES)} features")
    log("=" * 60)

    # Load dataset using load_dataset() - consistent with transformer.py
    log("Loading dataset with engineered features...")
    data, feature_cols, target_col = load_dataset()

    # Run LSTM prediction
    pred = predict_with_lstm(data, feature_cols, target_col)

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

    # Kaggle official evaluation
    kaggle_metrics = evaluate_portfolio_kaggle(pred, portfolio_size=200, toprank_weight_ratio=2)

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

    # Print monthly breakdown
    # if "daily_df" in port_metrics and not port_metrics["daily_df"].empty:
    #     daily_df = port_metrics["daily_df"]
    #     log("\n" + "=" * 40)
    #     log("MONTHLY SPREAD")
    #     log("=" * 40)
    #     daily_df["month"] = pd.to_datetime(daily_df["date"]).dt.strftime("%Y-%m")
    #     for _, row in daily_df.iterrows():
    #         log(f"  {row['month']}: Spread={row['spread']:+.4%}, Hit={row['hit_ratio']:.2%}")
    #     positive_months = (daily_df["spread"] > 0).sum()
    #     log(f"\nPositive months: {positive_months}/{len(daily_df)}")

    # Save metrics
    metrics = {**pred_metrics, **port_metrics, **kaggle_metrics}
    metrics_df = pd.DataFrame([metrics])
    metrics_path = os.path.join(OUTPUT_DIR, "metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)

    total_time = time.time() - start_time
    log(f"\nDone! Time: {total_time/60:.1f} min")


if __name__ == "__main__":
    main()
