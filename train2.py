"""
JPX Stock Prediction - Multi-Model Ensemble with Extended Grid Search

Improvements over train.py:
1. Extended Grid Search with more parameters and multiple validation pairs
2. Multi-source model training (stock, options, trades, financials)
3. Weighted ensemble based on validation performance

Training Strategy:
- Expanding window: train on past data, predict next year
- Each data source trained with separate model
- Weighted average ensemble based on validation Spearman correlation
"""

import os
import warnings
import gc
import itertools
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")

OUTPUT_DIR = "output_train2_ensemble"
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

TEST_YEAR = 2021
ROLL_TRAIN_YEARS = 2
TARGET_HORIZON = 30
TOP_K = 200
BOTTOM_K = 200

TRADING_COST_RATE = 0.0004
SLIPPAGE_RATE = 0.0002

# Extended Parameter Grid for Grid Search
PARAM_GRID_EXTENDED = {
    "n_estimators": [100, 300, 500],
    "learning_rate": [0.01, 0.02, 0.05],
    "max_depth": [4, 6, 8],
    "num_leaves": [15, 31, 63],
    "min_child_samples": [10, 20],
    "feature_fraction": [0.6, 0.8],
    "bagging_fraction": [0.6, 0.8],
    "bagging_freq": [1, 3],
    "reg_alpha": [0.1, 0.4],
    "reg_lambda": [0.1, 0.4],
}

# Multiple validation pairs for more robust tuning
TUNING_PAIRS = [
    (2017, 2018),
    (2018, 2019),
    (2019, 2020),
]

# Define feature groups by data source
SOURCE_FEATURE_GROUPS = {
    "stock": [
        "stk_ret_1", "stk_ret_2", "stk_ret_3", "stk_ret_5", "stk_ret_10", "stk_ret_20",
        "stk_logret_1", "stk_hl_spread", "stk_oc_spread", "stk_volume_chg_1",
        "stk_vol_5", "stk_vol_10", "stk_vol_20",
        "stk_ret_mean_5", "stk_ret_mean_10", "stk_ret_mean_20",
        "stk_close_to_ma_5", "stk_close_to_ma_10", "stk_close_to_ma_20",
        "stk_volume_to_ma_5", "stk_volume_to_ma_10", "stk_volume_to_ma_20",
        "stk_skew_20", "stk_dayofweek", "stk_month",
        "stk_expected_dividend", "stk_supervision_flag",
        "stk_mcap", "stk_sector", "stk_market_segment",
    ],
    "options": ["iv_avg"],
    "trades": ["trd_individual", "trd_foreigners", "trd_securitiescos", "trd_investmenttrusts"],
    "financials": ["fin_netsales", "fin_operatingprofit", "fin_ordinaryprofit", "fin_profit",
                   "fin_totalassets", "fin_equity"],
}


def log(msg):
    print(f"[INFO] {msg}")


def to_num(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_data_sources(data_dir="train_files"):
    """Load all data sources."""
    sources = {}

    stock_prices = pd.read_csv(os.path.join(data_dir, "stock_prices.csv"))
    stock_prices = to_num(stock_prices, ["Open", "High", "Low", "Close", "Volume", "AdjustmentFactor", "ExpectedDividend", "Target", "SupervisionFlag"])
    stock_prices["Date"] = pd.to_datetime(stock_prices["Date"])
    sources["stock_prices"] = stock_prices

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
    """Extract stock-level features from price data."""
    px = px.sort_values(["SecuritiesCode", "Date"]).reset_index(drop=True)
    g = px.groupby("SecuritiesCode", sort=False)
    c = px["Close"].replace(0, np.nan)

    for w in [1, 2, 3, 5, 10, 20]:
        px[f"stk_ret_{w}"] = g["Close"].pct_change(w)

    px["stk_logret_1"] = np.log(c).groupby(px["SecuritiesCode"]).diff(1)

    px["stk_hl_spread"] = (px["High"] - px["Low"]) / c
    px["stk_oc_spread"] = (px["Close"] - px["Open"]) / px["Open"].replace(0, np.nan)

    px["stk_volume_chg_1"] = g["Volume"].pct_change(1)

    for w in [5, 10, 20]:
        px[f"stk_vol_{w}"] = g["stk_logret_1"].transform(lambda x: x.rolling(w, min_periods=w).std())

    for w in [5, 10, 20]:
        px[f"stk_ret_mean_{w}"] = g["stk_ret_1"].transform(lambda x: x.rolling(w, min_periods=w).mean())

    for w in [5, 10, 20]:
        ma = g["Close"].transform(lambda x: x.rolling(w, min_periods=w).mean())
        px[f"stk_close_to_ma_{w}"] = px["Close"] / ma - 1

    for w in [5, 10, 20]:
        vma = g["Volume"].transform(lambda x: x.rolling(w, min_periods=w).mean())
        px[f"stk_volume_to_ma_{w}"] = px["Volume"] / vma - 1

    px["stk_skew_20"] = g["stk_logret_1"].transform(lambda x: x.rolling(20, min_periods=20).skew())

    px["stk_dayofweek"] = px["Date"].dt.dayofweek
    px["stk_month"] = px["Date"].dt.month

    px["stk_expected_dividend"] = px["ExpectedDividend"].fillna(0)
    px["stk_supervision_flag"] = px["SupervisionFlag"].astype(str).str.lower().eq("true").astype(int)

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
    """Load all data."""
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
    """Load dataset for training."""
    full_df, feature_cols = load_all_data()
    target_col = "target_30d"
    data = full_df[["Date", "SecuritiesCode"] + feature_cols + [target_col]].copy()
    log(f"Features: {len(feature_cols)}")
    log(f"Target: {target_col}")
    return data, feature_cols, target_col


def get_source_features(all_features, source_name):
    """Get features for a specific data source."""
    group = SOURCE_FEATURE_GROUPS.get(source_name, [])
    available = [f for f in group if f in all_features]
    return available


def generate_param_combinations(param_grid, max_combinations=50):
    """Generate parameter combinations from grid."""
    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]

    combinations = list(itertools.product(*values))

    if len(combinations) > max_combinations:
        np.random.seed(42)
        indices = np.random.choice(len(combinations), max_combinations, replace=False)
        combinations = [combinations[i] for i in sorted(indices)]

    return [{keys[i]: comb[i] for i in range(len(keys))} for comb in combinations]


def tune_hyperparameters_extended(df, feature_cols, target_col):
    """Extended hyperparameter tuning with multiple validation pairs."""
    log("Starting EXTENDED hyperparameter tuning...")

    df = df.copy()
    df["Year"] = df["Date"].dt.year

    best_params = None
    best_score = -np.inf

    param_combinations = generate_param_combinations(PARAM_GRID_EXTENDED, max_combinations=50)

    log(f"Testing {len(param_combinations)} param combinations across {len(TUNING_PAIRS)} validation pairs...")

    sample_size = min(30000, len(df[df[target_col].notna()]))
    df_sample = df[df[target_col].notna()].sample(n=sample_size, random_state=42)

    for params in param_combinations:
        scores = []

        for train_year, test_year in TUNING_PAIRS:
            train_df = df_sample[(df_sample["Year"] >= train_year) & (df_sample["Year"] < test_year)]
            test_df = df_sample[df_sample["Year"] == test_year]

            if len(train_df) < 500 or len(test_df) < 50:
                continue

            try:
                model = LGBMRegressor(
                    n_estimators=params.get("n_estimators", 300),
                    learning_rate=params.get("learning_rate", 0.02),
                    max_depth=params.get("max_depth", 6),
                    num_leaves=params.get("num_leaves", 31),
                    min_child_samples=params.get("min_child_samples", 20),
                    subsample=params.get("bagging_fraction", 0.8),
                    colsample_bytree=params.get("feature_fraction", 0.8),
                    reg_alpha=params.get("reg_alpha", 0.1),
                    reg_lambda=params.get("reg_lambda", 0.1),
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
        best_params = {
            "n_estimators": 300, "learning_rate": 0.02, "max_depth": 6,
            "num_leaves": 31, "min_child_samples": 20,
            "feature_fraction": 0.8, "bagging_fraction": 0.8,
            "reg_alpha": 0.1, "reg_lambda": 0.1,
        }

    return best_params


def fit_lgbm_regressor(train_df, feature_cols, target_col, params):
    """Fit LightGBM regressor with given parameters."""
    model = LGBMRegressor(
        n_estimators=params.get("n_estimators", 300),
        learning_rate=params.get("learning_rate", 0.02),
        max_depth=params.get("max_depth", 6),
        num_leaves=params.get("num_leaves", 31),
        min_child_samples=params.get("min_child_samples", 20),
        subsample=params.get("bagging_fraction", 0.8),
        colsample_bytree=params.get("feature_fraction", 0.8),
        reg_alpha=params.get("reg_alpha", 0.1),
        reg_lambda=params.get("reg_lambda", 0.1),
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(train_df[feature_cols], train_df[target_col].values)
    return model


def fit_lgbm_classifier(train_df, feature_cols, target_col):
    """Fit LightGBM classifier for direction prediction."""
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


def calculate_ensemble_weights(models, val_df, source_features_dict, target_col):
    """Calculate weights for each model based on validation performance."""
    weights = {}

    for source_name, (model, features) in models.items():
        if not features:
            weights[source_name] = 0.0
            continue

        try:
            pred = model.predict(val_df[features])
            spearman = pd.Series(val_df[target_col].values).corr(pd.Series(pred), method="spearman")
            weights[source_name] = max(0, spearman) if not np.isnan(spearman) else 0.0
        except:
            weights[source_name] = 0.0

    total_weight = sum(weights.values())
    if total_weight > 0:
        weights = {k: v / total_weight for k, v in weights.items()}
    else:
        weights = {k: 1.0 / len(models) for k in models.keys()}

    log(f"Ensemble weights: {weights}")
    return weights


def predict_with_ensemble(df, feature_cols, target_col, tuned_params):
    """Predict using ensemble of models trained on different data sources."""
    log(f"Running ensemble prediction with expanding window...")

    df = df.copy()
    df["Year"] = df["Date"].dt.year

    test_year = 2021
    train_df = df[(df["Year"] < test_year) & df[target_col].notna()].copy()
    test_df = df[(df["Year"] == test_year) & df[target_col].notna()].copy()

    if train_df.empty or test_df.empty:
        log("No data available for training/prediction")
        return pd.DataFrame(columns=["Date", "SecuritiesCode", "y_true", "pred", "train_year"])

    log(f"Train: {len(train_df):,}, Test: {len(test_df):,}")

    # Get validation set for weight calculation
    val_df = df[(df["Year"] == 2020) & df[target_col].notna()].copy()
    if val_df.empty:
        val_df = train_df.tail(min(10000, len(train_df)))

    # Train models for each data source
    models = {}
    source_features_dict = {}

    for source_name in SOURCE_FEATURE_GROUPS.keys():
        features = get_source_features(feature_cols, source_name)
        if features:
            source_features_dict[source_name] = features
            log(f"Training model for {source_name} with {len(features)} features")
            models[source_name] = (fit_lgbm_regressor(train_df, features, target_col, tuned_params), features)

    # Calculate ensemble weights based on validation performance
    weights = calculate_ensemble_weights(models, val_df, source_features_dict, target_col)

    # Also train a full model for comparison
    log("Training full model with all features...")
    full_model = fit_lgbm_regressor(train_df, feature_cols, target_col, tuned_params)

    # Generate predictions
    log("Generating ensemble predictions...")

    # Ensemble prediction
    ensemble_pred = np.zeros(len(test_df))
    for source_name, (model, features) in models.items():
        pred = model.predict(test_df[features])
        ensemble_pred += pred * weights.get(source_name, 0)

    # Full model prediction
    full_pred = full_model.predict(test_df[feature_cols])

    # Hybrid adjustment using classifier
    model_cls = fit_lgbm_classifier(train_df, feature_cols, target_col)
    cls_prob = model_cls.predict_proba(test_df[feature_cols])[:, 1]

    # Combine ensemble with hybrid adjustment
    hybrid_pred = ensemble_pred * (2 * cls_prob - 1)

    # Also compute weighted combination of ensemble and full model
    final_pred = 0.5 * hybrid_pred + 0.5 * full_pred * (2 * cls_prob - 1)

    out = test_df[["Date", "SecuritiesCode", target_col]].copy()
    out = out.rename(columns={target_col: "y_true"})
    out["pred_ensemble"] = ensemble_pred
    out["pred_full"] = full_pred
    out["pred_prob"] = cls_prob
    out["pred"] = final_pred
    out["train_year"] = test_year - 1

    # Clean up
    for model, _ in models.values():
        del model
    del full_model, model_cls
    gc.collect()

    log(f"Total predictions: {len(out):,}")
    return out


def evaluate_portfolio(pred_df):
    """Evaluate portfolio performance."""
    if pred_df.empty:
        return {"num_days": 0, "sharpe": np.nan, "hit_ratio": np.nan, "spread": np.nan}

    pred_df = pred_df.sort_values("Date").reset_index(drop=True)
    dates = sorted(pred_df["Date"].unique())

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
    """Evaluate prediction quality."""
    if pred_df.empty:
        return {"rmse": np.nan, "spearman": np.nan, "hit": np.nan}

    y = pred_df["y_true"].values
    p = pred_df["pred"].values

    rmse = float(np.sqrt(mean_squared_error(y, p)))
    spearman = float(pd.Series(y).corr(pd.Series(p), method="spearman"))
    hit = float(np.mean(np.sign(y) == np.sign(p)))

    return {"rmse": rmse, "spearman": spearman, "hit": hit}


def evaluate_daily(pred_df, top_k=TOP_K, bottom_k=BOTTOM_K):
    """
    Daily evaluation - evaluate every day instead of monthly rebalancing.
    This provides more data points and reduces noise from monthly sampling.
    """
    if pred_df.empty:
        return {"num_days": 0, "sharpe": np.nan, "hit_ratio": np.nan, "spread": np.nan, "daily_df": None}

    pred_df = pred_df.sort_values("Date").reset_index(drop=True)
    dates = sorted(pred_df["Date"].unique())

    daily_results = []

    for eval_date in dates:
        day_pred = pred_df[pred_df["Date"] == eval_date].copy()

        if len(day_pred) < top_k + bottom_k:
            continue

        sorted_pred = day_pred.sort_values("pred", ascending=False).reset_index(drop=True)

        top_stocks = sorted_pred.head(top_k)
        bottom_stocks = sorted_pred.tail(bottom_k)

        top_ret = top_stocks["y_true"].mean()
        bottom_ret = bottom_stocks["y_true"].mean()
        spread = top_ret - bottom_ret

        top_correct = (top_stocks["y_true"] > 0).sum()
        bottom_correct = (bottom_stocks["y_true"] < 0).sum()
        hit = (top_correct + bottom_correct) / (top_k + bottom_k)

        daily_results.append({
            "date": eval_date,
            "top_ret": top_ret,
            "bottom_ret": bottom_ret,
            "spread": spread,
            "hit_ratio": hit,
        })

    if not daily_results:
        return {"num_days": 0, "sharpe": np.nan, "hit_ratio": np.nan, "spread": np.nan, "daily_df": None}

    daily_df = pd.DataFrame(daily_results)

    avg_spread = daily_df["spread"].mean()
    std_spread = daily_df["spread"].std()
    sharpe = (avg_spread / std_spread * np.sqrt(252)) if std_spread > 0 else np.nan

    return {
        "num_days": len(daily_df),
        "sharpe": float(sharpe),
        "hit_ratio": float(daily_df["hit_ratio"].mean()),
        "spread": float(daily_df["spread"].sum()),
        "avg_spread": float(avg_spread),
        "std_spread": float(std_spread),
        "daily_df": daily_df,
    }


def main():
    import time
    start_time = time.time()

    log("=" * 60)
    log("JPX 30-Day Horizon - Multi-Model Ensemble + Extended Grid Search")
    log("=" * 60)

    data, feature_cols, target_col = load_dataset()

    # Extended hyperparameter tuning
    tuned_params = tune_hyperparameters_extended(data, feature_cols, target_col)

    # Ensemble prediction
    pred = predict_with_ensemble(data, feature_cols, target_col, tuned_params)

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

    # ===== NEW: Daily Evaluation =====
    log("\n" + "=" * 50)
    log("DAILY EVALUATION (Every trading day)")
    log("=" * 50)
    daily_metrics = evaluate_daily(pred)

    log(f"Number of trading days: {daily_metrics['num_days']}")
    log(f"Average Daily Spread: {daily_metrics['avg_spread']:.6f}")
    log(f"Std Daily Spread: {daily_metrics['std_spread']:.6f}")
    log(f"Total Cumulative Spread: {daily_metrics['spread']:.4%}")
    log(f"Sharpe Ratio (annualized): {daily_metrics['sharpe']:.4f}")
    log(f"Hit Ratio: {daily_metrics['hit_ratio']:.2%}")

    if daily_metrics["daily_df"] is not None and not daily_metrics["daily_df"].empty:
        log("\n" + "=" * 40)
        log("MONTHLY BREAKDOWN (Daily Eval)")
        log("=" * 40)
        daily_df = daily_metrics["daily_df"]
        daily_df["month"] = pd.to_datetime(daily_df["date"]).dt.strftime("%Y-%m")
        monthly_spread = daily_df.groupby("month")["spread"].sum()
        for month, spread in monthly_spread.items():
            log(f"  {month}: Spread={spread:+.4%}")
        positive_months = (monthly_spread > 0).sum()
        log(f"\nPositive months: {positive_months}/{len(monthly_spread)}")

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
