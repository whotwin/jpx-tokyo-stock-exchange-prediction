# LSTM Data Loading Functions

import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import log


def to_num(df, cols):
    """Convert columns to numeric."""
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_data_sources(data_dir="train_files"):
    """Load data sources."""
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


def options_features(opts):
    """Extract options features."""
    if opts is None or opts.empty:
        return pd.DataFrame()

    opts = opts.sort_values("Date").reset_index(drop=True)

    if "ImpliedVolatility" in opts.columns:
        iv = opts.groupby("Date")["ImpliedVolatility"].mean().reset_index()
        iv.columns = ["Date", "iv_avg"]
        return iv

    return pd.DataFrame()


def trades_features(trades_df):
    """Extract trades features."""
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


def build_feature_table(sources, start_date=None, end_date=None):
    """Build full feature table."""
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
    """Build 30-day forward return labels."""
    px = stock_prices[["Date", "SecuritiesCode", "Close"]].copy()
    px["Close"] = pd.to_numeric(px["Close"], errors="coerce")
    px = px.sort_values(["SecuritiesCode", "Date"]).reset_index(drop=True)

    px["target_30d"] = px.groupby("SecuritiesCode")["Close"].shift(-30) / px["Close"] - 1.0

    return px[["Date", "SecuritiesCode", "target_30d"]]


def load_all_data():
    """Load all data sources."""
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


def add_cross_sectional_features(df):
    """Add cross-sectional features: rank within each time step."""
    rank_features = ["stk_ret_1", "stk_vol_20", "stk_volume_chg_1"]

    for feat in rank_features:
        if feat in df.columns:
            df[f"{feat}_rank"] = df.groupby("Date")[feat].rank(pct=True)
            log(f"Added cross-sectional feature: {feat}_rank")

    return df


def load_all_data_for_lstm():
    """Load all data sources with advanced features for LSTM."""
    log("Loading data with advanced features for LSTM...")

    sources = load_data_sources("train_files")

    log("Building full feature table with advanced features...")
    full_df, feature_cols = build_feature_table(
        sources=sources,
        start_date="2017-01-04",
        end_date="2021-12-03",
    )

    labels = build_30d_labels(sources["stock_prices"])
    full_df = full_df.merge(labels, on=["Date", "SecuritiesCode"], how="left")
    full_df = full_df.sort_values(["Date", "SecuritiesCode"]).reset_index(drop=True)

    log("Adding cross-sectional features (rank within each time step)...")
    full_df = add_cross_sectional_features(full_df)

    log(f"Loaded: {len(full_df)} rows, {len(feature_cols)} features")

    return full_df, feature_cols


def build_30d_labels_raw(df):
    """Build 30-day forward return labels from raw Close prices."""
    df = df.sort_values(["SecuritiesCode", "Date"]).reset_index(drop=True)
    df["target_30d"] = df.groupby("SecuritiesCode")["Close"].shift(-30) / df["Close"] - 1.0
    return df


def cross_sectional_normalize(df, feature_cols, eps=1e-8):
    """Cross-sectional normalization: normalize across stocks at each time step."""
    log("Applying cross-sectional normalization across stocks at each time step...")

    df = df.sort_values(["Date", "SecuritiesCode"]).reset_index(drop=True)

    for col in tqdm(feature_cols, desc="Cross-sectional normalizing"):
        date_stats = df.groupby("Date")[col].agg(['mean', 'std']).reset_index()
        date_stats.columns = ['Date', f'{col}_mean', f'{col}_std']

        df = df.merge(date_stats, on='Date', how='left')

        df[f'{col}_std'] = df[f'{col}_std'].fillna(1.0)
        df[f'{col}_std'] = df[f'{col}_std'].replace(0, 1.0)
        df[f'{col}_std'] = df[f'{col}_std'].clip(lower=eps)

        df[col] = (df[col] - df[f'{col}_mean']) / df[f'{col}_std']
        df[col] = df[col].clip(-10, 10)

        df = df.drop(columns=[f'{col}_mean', f'{col}_std'])

    log("Cross-sectional normalization completed.")
    return df


def create_sequences(df, feature_cols, seq_length=60, target_col="target_30d"):
    """Create sequences for LSTM."""
    log(f"Creating sequences with seq_length={seq_length}...")

    from .config import TARGET_HORIZON

    df = df.sort_values(["SecuritiesCode", "Date"]).reset_index(drop=True)

    sequences = []
    targets = []
    dates = []
    codes = []

    stocks = df["SecuritiesCode"].unique()

    for code in tqdm(stocks, desc="Building sequences"):
        stock_data = df[df["SecuritiesCode"] == code].sort_values("Date").reset_index(drop=True)

        if len(stock_data) < seq_length + TARGET_HORIZON + 1:
            continue

        feature_values = stock_data[feature_cols].values
        target_values = stock_data[target_col].values
        date_values = stock_data["Date"].values

        for i in range(seq_length, len(stock_data) - TARGET_HORIZON):
            seq_features = feature_values[i - seq_length:i]
            target = target_values[i]

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

    return X, y, dates_arr, codes_arr


def load_dataset():
    """Load data for LSTM with advanced features."""
    from .config import LSTM_FEATURES

    log("Loading data with advanced features for LSTM...")

    full_df, _ = load_all_data_for_lstm()

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
