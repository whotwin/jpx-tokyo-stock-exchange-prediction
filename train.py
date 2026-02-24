"""
JPX Stock Prediction - 30-Day Horizon with Expanding Window Training

Training Strategy:
- Use all available data from 2017 onwards
- Expanding window: train on past data, predict next year
- 2017 train -> predict 2018
- 2018 train -> predict 2019
- ...
- 2020 train -> predict 2021

Also includes hyperparameter tuning.
"""

import os
import warnings
import gc
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")

OUTPUT_DIR = "output_train_30d"
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

TEST_YEAR = 2021
ROLL_TRAIN_YEARS = 2  # Use 2-year rolling window
TARGET_HORIZON = 30
TOP_K = 200
BOTTOM_K = 200

TRADING_COST_RATE = 0.0004
SLIPPAGE_RATE = 0.0002

PARAM_GRID = {
    "n_estimators": [300, 500],
    "learning_rate": [0.01, 0.02],
    "max_depth": [6, 8],
    "num_leaves": [15, 31],
}


def log(msg):
    print(f"[INFO] {msg}")


def to_num(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_data_sources(data_dir="train_files"):
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
    """Extract stock-level features from price data - enhanced version."""
    px = px.sort_values(["SecuritiesCode", "Date"]).reset_index(drop=True)
    # Group each stock, analyze respective stock.
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
        # Log market cap (in 100M yen)
        px["stk_mcap"] = np.log(px["MarketCapitalization"].fillna(1e8) / 1e8 + 1)
        # Sector code (numeric)
        px["stk_sector"] = pd.Categorical(px["33SectorName"]).codes
        # Market segment (Prime=0, First=1, etc.)
        px["stk_market_segment"] = pd.Categorical(px["NewMarketSegment"]).codes
        # Drop original columns
        px = px.drop(columns=["MarketCapitalization", "33SectorName", "NewMarketSegment"], errors="ignore")

    return px


def options_features(opts):
    """Extract options features."""
    if opts is None or opts.empty:
        return pd.DataFrame()

    opts = opts.sort_values("Date").reset_index(drop=True)

    # Simple IV average by date
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

    # Simple average by date
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

    # Forward fill within each security
    fn[available_cols] = fn.groupby("SecuritiesCode")[available_cols].ffill()

    # Keep latest financial data by SecuritiesCode
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

    # Stock features (with market cap and sector from stock_list)
    stock_list = sources.get("stock_list", None)
    px_with_features = stock_features(prices, stock_list=stock_list)

    # Extract stock feature columns (new naming: stk_*)
    stock_cols = [c for c in px_with_features.columns if c.startswith("stk_")]
    df = df.merge(px_with_features[["Date", "SecuritiesCode"] + stock_cols],
                  on=["Date", "SecuritiesCode"], how="left")

    # Options features
    if "options" in sources:
        opt_feat = options_features(sources["options"])
        if not opt_feat.empty:
            df = df.merge(opt_feat, on="Date", how="left")

    # Trades features
    if "trades" in sources:
        trd_feat = trades_features(sources["trades"])
        if not trd_feat.empty:
            df = df.merge(trd_feat, on="Date", how="left")

    # Financials features
    if "financials" in sources:
        fin_feat = financials_features(sources["financials"])
        if not fin_feat.empty:
            df = df.merge(fin_feat, on="SecuritiesCode", how="left")

    # Get all feature columns (exclude non-features)
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


def load_dataset():
    full_df, feature_cols = load_all_data()
    target_col = "target_30d"
    data = full_df[["Date", "SecuritiesCode"] + feature_cols + [target_col]].copy()
    log(f"Features: {len(feature_cols)}")
    log(f"Target: {target_col}")
    return data, feature_cols, target_col


def tune_hyperparameters(df, feature_cols, target_col):
    log("Starting hyperparameter tuning...")

    df = df.copy()
    df["Year"] = df["Date"].dt.year

    tuning_pairs = [(2019, 2020)]

    best_params = None
    best_score = -np.inf

    param_combinations = []
    for n_est in PARAM_GRID["n_estimators"]:
        for lr in PARAM_GRID["learning_rate"]:
            for depth in PARAM_GRID["max_depth"]:
                for leaves in PARAM_GRID["num_leaves"]:
                    param_combinations.append({
                        "n_estimators": n_est,
                        "learning_rate": lr,
                        "max_depth": depth,
                        "num_leaves": leaves,
                    })

    log(f"Testing {min(12, len(param_combinations))} param combinations...")

    sample_size = min(30000, len(df[df[target_col].notna()]))
    df_sample = df[df[target_col].notna()].sample(n=sample_size, random_state=42)

    for params in param_combinations[:12]:
        scores = []

        for train_year, test_year in tuning_pairs:
            train_df = df_sample[(df_sample["Year"] >= train_year) & (df_sample["Year"] < test_year)]
            test_df = df_sample[df_sample["Year"] == test_year]

            if len(train_df) < 500 or len(test_df) < 50:
                continue

            try:
                model = LGBMRegressor(
                    n_estimators=params["n_estimators"],
                    learning_rate=params["learning_rate"],
                    max_depth=params["max_depth"],
                    num_leaves=params["num_leaves"],
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1,
                )
                model.fit(train_df[feature_cols], train_df[target_col].values)

                pred = model.predict(test_df[feature_cols])
                spearman = pd.Series(test_df[target_col].values).corr(pd.Series(pred), method="spearman")

                if not np.isnan(spearman):
                    scores.append(spearman)

                del model
            except:
                continue

        if scores:
            avg_score = np.mean(scores)
            if avg_score > best_score:
                best_score = avg_score
                best_params = params

    if best_params:
        log(f"Best params: {best_params}, Score: {best_score:.4f}")
    else:
        log("Using default parameters")
        best_params = {"n_estimators": 500, "learning_rate": 0.02, "max_depth": 8, "num_leaves": 31}

    return best_params


def fit_lgbm_regressor(train_df, feature_cols, target_col, params):
    model = LGBMRegressor(
        n_estimators=params.get("n_estimators", 500),
        learning_rate=params.get("learning_rate", 0.02),
        max_depth=params.get("max_depth", 8),
        num_leaves=params.get("num_leaves", 31),
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(train_df[feature_cols], train_df[target_col].values)
    return model


def fit_lgbm_classifier(train_df, feature_cols, target_col):
    y_binary = (train_df[target_col] > 0).astype(int)

    model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.02,
        max_depth=6,
        num_leaves=15,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(train_df[feature_cols], y_binary)
    return model


def predict_with_expanding_window(df, feature_cols, target_col, tuned_params):
    log(f"Running all-data-for-training, 2021-only prediction...")

    df = df.copy()
    df["Year"] = df["Date"].dt.year

    pred_parts = []

    # Only test on 2021, use all prior data for training
    test_year = 2021
    train_df = df[(df["Year"] < test_year) & df[target_col].notna()].copy()
    test_df = df[(df["Year"] == test_year) & df[target_col].notna()].copy()

    if train_df.empty or test_df.empty:
        log(f"  Year {test_year}: No data, skipping")
    else:
        log(f"  Year {test_year}: Train {len(train_df):,} (all prior years), Test {len(test_df):,}")

        model_reg = fit_lgbm_regressor(train_df, feature_cols, target_col, tuned_params)
        model_cls = fit_lgbm_classifier(train_df, feature_cols, target_col)

        reg_pred = model_reg.predict(test_df[feature_cols])
        cls_prob = model_cls.predict_proba(test_df[feature_cols])[:, 1]

        hybrid_pred = reg_pred * (2 * cls_prob - 1)

        out = test_df[["Date", "SecuritiesCode", target_col]].copy()
        out = out.rename(columns={target_col: "y_true"})
        out["pred_reg"] = reg_pred
        out["pred_prob"] = cls_prob
        out["pred"] = hybrid_pred
        out["train_year"] = test_year - 1
        pred_parts.append(out)

        del model_reg, model_cls
        gc.collect()

    if pred_parts:
        pred = pd.concat(pred_parts, ignore_index=True).sort_values(["Date", "SecuritiesCode"]).reset_index(drop=True)
    else:
        pred = pd.DataFrame(columns=["Date", "SecuritiesCode", "y_true", "pred", "train_year"])

    log(f"Total predictions: {len(pred):,}")
    return pred


def evaluate_portfolio(pred_df):
    if pred_df.empty:
        return {"num_days": 0, "sharpe": np.nan, "hit_ratio": np.nan, "spread": np.nan}

    pred_df = pred_df.sort_values("Date").reset_index(drop=True)
    dates = sorted(pred_df["Date"].unique())

    # Monthly rebalancing dates (first trading day of each month)
    monthly_dates = []
    current_year_month = None
    for d in dates:
        dt = pd.to_datetime(d)
        year_month = (dt.year, dt.month)
        if year_month != current_year_month:
            monthly_dates.append(d)
            current_year_month = year_month

    daily_results = []
    prev_top = set()
    prev_bottom = set()

    for rebal_date in monthly_dates:
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
    sharpe = (avg_spread / std_spread * np.sqrt(4)) if std_spread > 0 else np.nan

    return {
        "num_days": len(daily_df),
        "sharpe": float(sharpe),
        "hit_ratio": float(daily_df["hit_ratio"].mean()),
        "spread": float(daily_df["spread"].sum()),
        "daily_df": daily_df,
    }


def evaluate_predictions(pred_df):
    if pred_df.empty:
        return {"rmse": np.nan, "spearman": np.nan, "hit": np.nan}

    y = pred_df["y_true"].values
    p = pred_df["pred"].values

    rmse = float(np.sqrt(mean_squared_error(y, p)))
    spearman = float(pd.Series(y).corr(pd.Series(p), method="spearman"))
    hit = float(np.mean(np.sign(y) == np.sign(p)))

    return {"rmse": rmse, "spearman": spearman, "hit": hit}


def main():
    import time
    start_time = time.time()

    log("=" * 60)
    log("JPX 30-Day Horizon - Expanding Window + Tuning")
    log("=" * 60)

    data, feature_cols, target_col = load_dataset()

    tuned_params = tune_hyperparameters(data, feature_cols, target_col)
    pred = predict_with_expanding_window(data, feature_cols, target_col, tuned_params)

    pred_path = os.path.join(OUTPUT_DIR, "predictions.csv")
    pred.to_csv(pred_path, index=False)
    log(f"Saved: {pred_path}")

    port_metrics = evaluate_portfolio(pred)
    pred_metrics = evaluate_predictions(pred)

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

    # Print monthly breakdown
    if "daily_df" in port_metrics and not port_metrics["daily_df"].empty:
        daily_df = port_metrics["daily_df"]
        log("\n" + "=" * 40)
        log("MONTHLY SPREAD")
        log("=" * 40)
        daily_df["month"] = pd.to_datetime(daily_df["date"]).dt.strftime("%Y-%m")
        for _, row in daily_df.iterrows():
            log(f"  {row['month']}: Spread={row['spread']:+.4%}, Hit={row['hit_ratio']:.2%}")
        positive_months = (daily_df["spread"] > 0).sum()
        log(f"\nPositive months: {positive_months}/{len(daily_df)}")

    metrics = {**pred_metrics, **port_metrics}
    metrics_df = pd.DataFrame([metrics])
    metrics_path = os.path.join(OUTPUT_DIR, "metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)

    total_time = time.time() - start_time
    log(f"\nDone! Time: {total_time/60:.1f} min")


if __name__ == "__main__":
    main()
