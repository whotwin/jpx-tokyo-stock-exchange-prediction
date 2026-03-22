# iTransformer Data Loading Functions

import os
import numpy as np
import pandas as pd
from tqdm import tqdm

# Import config
from .config import log, TARGET_HORIZON


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
    all_feature_cols = [c for c in df.columns if c not in non_feature_cols]

    # Shift features by 1 to avoid leakage
    for col in all_feature_cols:
        df[col] = df.groupby("SecuritiesCode", sort=False)[col].shift(1)

    df = df.dropna(subset=["Date", "SecuritiesCode", "Close"])
    df[all_feature_cols] = df[all_feature_cols].fillna(0)

    # Add target labels
    labels = build_30d_labels(sources["stock_prices"])
    df = df.merge(labels, on=["Date", "SecuritiesCode"], how="left")
    df = df.sort_values(["Date", "SecuritiesCode"]).reset_index(drop=True)

    log(f"Loaded: {len(df)} rows with {len(all_feature_cols)} features")

    return df, all_feature_cols


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
        feature_means = []  # Store means for each feature
        for col in use_features:
            pivot = feature_pivots[col]
            values = pivot.loc[lookback_dates].values.copy()
            # Compute mean for this feature (across stocks and time in lookback window)
            # Shape: (seq_length, num_stocks) -> mean across both
            feature_mean = np.nanmean(values)  # Use global mean across lookback window
            feature_means.append(feature_mean)
            # Fill NaN with mean instead of 0
            values = np.where(np.isnan(values), feature_mean, values)
            # Clip extreme values
            values = np.clip(values, -10, 10)
            seq_features_list.append(values)

        # Stack: (seq_length, num_stocks, num_features)
        # Keep 4D tensor: (seq_length, num_stocks, num_features)
        # NOT reshaping to 2D - attention will learn stock ↔ stock relationships
        seq_features = np.stack(seq_features_list, axis=-1)

        # Get targets: (num_stocks,)
        target_values = target_pivot.loc[target_date].values

        # Compute mean of valid targets for filling NaN
        target_mean = np.nanmean(target_values)

        # Skip if too many NaN in targets
        valid_mask = ~np.isnan(target_values)
        if valid_mask.sum() < len(stock_codes) * 0.5:
            continue

        # Fill NaN in targets with mean
        target_values = np.where(np.isnan(target_values), target_mean, target_values)

        sequences.append(seq_features)
        targets.append(target_values)
        valid_dates.append(target_date)

    X = np.array(sequences, dtype=np.float32)
    y = np.array(targets, dtype=np.float32)
    dates_arr = np.array(valid_dates)

    log(f"Created sequences: X={X.shape}, y={y.shape}")
    log(f"Shape: (batch, seq_len, num_stocks, num_features) = {X.shape}")

    # Return num_features for model projection
    return X, y, dates_arr, stock_codes, num_features
