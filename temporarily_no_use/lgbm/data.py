# LightGBM Model Package

import os
import gc

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Import from base module
import demo_v2 as base

from .config import (
    OUTPUT_DIR,
    PLOT_DIR,
    TEST_YEAR,
    ROLL_TRAIN_YEARS,
    ROLL_RETRAIN_FREQ,
    TARGET_HORIZON,
    DATA_CONFIG,
    LGBM_NUM_LEAVES,
    LGBM_MAX_DEPTH,
    LGBM_LEARNING_RATE,
    LGBM_N_ESTIMATORS,
    LGBM_SUBSAMPLE,
    LGBM_COLSAMPLE_BYTREE,
    LGBM_MIN_CHILD_SAMPLES,
    LGBM_REG_ALPHA,
    LGBM_REG_LAMBDA,
    TOP_K,
    BOTTOM_K,
    log,
)


def build_forward_return_labels(stock_prices, horizons):
    """Build forward return labels."""
    px = stock_prices[["Date", "SecuritiesCode", "Close"]].copy()
    px["Close"] = pd.to_numeric(px["Close"], errors="coerce")
    px = px.sort_values(["SecuritiesCode", "Date"]).reset_index(drop=True)
    grp = px.groupby("SecuritiesCode", sort=False)["Close"]
    for h in horizons:
        px[f"target_{h}d"] = grp.shift(-h) / px["Close"].replace(0, np.nan) - 1.0
    return px[["Date", "SecuritiesCode"] + [f"target_{h}d" for h in horizons]]


def load_all_data():
    """Load all data sources."""
    log("Loading all data sources...")
    sources = base.load_data_sources("train_files")
    log("Building full feature table...")
    full_df, groups = base.build_feature_table(
        sources=sources,
        start_date="2017-01-04",
        end_date="2021-12-03",
    )
    # Add forward return labels
    labels = build_forward_return_labels(sources["stock_prices"], [TARGET_HORIZON])
    full_df = full_df.merge(labels, on=["Date", "SecuritiesCode"], how="left")
    full_df = full_df.sort_values(["Date", "SecuritiesCode"]).reset_index(drop=True)
    log(f"Loaded: {len(full_df)} rows")

    return full_df, groups, sources


def load_dataset():
    """Load dataset with stock+all configuration."""
    full_df, groups, sources = load_all_data()

    # Use all features
    feature_cols = sorted(set(sum(groups.values(), [])))
    target_col = f"target_{TARGET_HORIZON}d"

    data = full_df[["Date", "SecuritiesCode"] + feature_cols + [target_col]].copy()
    log(f"Features: {len(feature_cols)}")
    log(f"Target: {target_col}")

    return data, feature_cols, target_col


def iter_periods(dates, freq=ROLL_RETRAIN_FREQ):
    """Iterate over time periods."""
    ds = pd.Series(sorted(pd.unique(pd.to_datetime(dates))))
    per = ds.dt.to_period(freq)
    for p in per.drop_duplicates():
        m = per == p
        dd = ds[m]
        yield dd.min(), dd.max()


def fit_lgbm(train_df, feature_cols, target_col):
    """Fit LightGBM model."""
    params = base.params()
    model = LGBMRegressor(**params)
    model.fit(train_df[feature_cols], train_df[target_col].values)
    return model


def predict_timeseries_lgbm(df, feature_cols, target_col):
    """Walk-forward prediction with monthly retrain."""
    log(f"Running LGBM walkforward prediction for {target_col}...")
    test_df = df[df["Date"].dt.year == TEST_YEAR].copy()
    pred_parts = []

    periods = list(iter_periods(test_df["Date"], freq=ROLL_RETRAIN_FREQ))
    log(f"Number of periods: {len(periods)}")

    for i, (period_start, period_end) in enumerate(periods):
        train_end = period_start - pd.Timedelta(days=1)
        train_start = train_end - pd.DateOffset(years=ROLL_TRAIN_YEARS) + pd.Timedelta(days=1)

        train_win = df[(df["Date"] >= train_start) & (df["Date"] <= train_end) & df[target_col].notna()].copy()
        infer_win = test_df[(test_df["Date"] >= period_start) & (test_df["Date"] <= period_end) & test_df[target_col].notna()].copy()

        if train_win.empty or infer_win.empty:
            log(f"  Period {i+1}/{len(periods)}: No data, skipping")
            continue

        log(f"  Period {i+1}/{len(periods)}: Train {len(train_win):,} rows, Test {len(infer_win):,} rows")

        model = fit_lgbm(train_win, feature_cols, target_col)
        out = infer_win[["Date", "SecuritiesCode", target_col]].copy()
        out = out.rename(columns={target_col: "y_true"})
        out["pred"] = model.predict(infer_win[feature_cols])
        out["retrain_period_start"] = period_start
        pred_parts.append(out)

        del model
        gc.collect()

    if pred_parts:
        pred = pd.concat(pred_parts, ignore_index=True).sort_values(["Date", "SecuritiesCode"]).reset_index(drop=True)
    else:
        pred = pd.DataFrame(columns=["Date", "SecuritiesCode", "y_true", "pred", "retrain_period_start"])

    log(f"Total predictions: {len(pred):,}")
    return pred


def evaluate_predictions(pred_df):
    """Evaluate predictions."""
    if pred_df.empty:
        return {
            "rows": 0, "days": 0, "rmse": np.nan, "mae": np.nan,
            "pearson_corr": np.nan, "spearman_corr": np.nan, "hit_ratio": np.nan,
            "mean_daily_rankic": np.nan, "rankic_ir": np.nan,
        }

    y = pred_df["y_true"].to_numpy(dtype=float)
    p = pred_df["pred"].to_numpy(dtype=float)

    rmse = float(np.sqrt(mean_squared_error(y, p)))
    mae = float(mean_absolute_error(y, p))
    pearson = float(pd.Series(y).corr(pd.Series(p), method="pearson"))
    spearman = float(pd.Series(y).corr(pd.Series(p), method="spearman"))
    hit = float(np.mean(np.sign(y) == np.sign(p)))

    daily_rankic = []
    for _, g in pred_df.groupby("Date"):
        if g["y_true"].nunique() > 1 and g["pred"].nunique() > 1:
            ic = g["y_true"].corr(g["pred"], method="spearman")
            if pd.notna(ic):
                daily_rankic.append(float(ic))

    mean_daily_rankic = float(np.mean(daily_rankic)) if daily_rankic else np.nan
    std_daily_rankic = float(np.std(daily_rankic)) if daily_rankic else np.nan
    rankic_ir = float(mean_daily_rankic / std_daily_rankic) if std_daily_rankic and std_daily_rankic > 0 else np.nan

    return {
        "rows": int(len(pred_df)), "days": int(pred_df["Date"].nunique()),
        "rmse": rmse, "mae": mae, "pearson_corr": pearson, "spearman_corr": spearman,
        "hit_ratio": hit, "mean_daily_rankic": mean_daily_rankic, "rankic_ir": rankic_ir,
    }


def evaluate_portfolio_from_predictions(pred_df):
    """Evaluate portfolio from predictions."""
    if pred_df.empty:
        return {
            "portfolio_total_return": np.nan, "portfolio_sharpe": np.nan,
            "portfolio_max_drawdown": np.nan, "portfolio_avg_turnover": np.nan,
        }

    tmp = pred_df[["Date", "SecuritiesCode", "pred", "y_true"]].copy()
    tmp = tmp.rename(columns={"y_true": "Target"})

    _, daily_perf, m = base.construct_rank_band_portfolio(
        tmp, pred_col="pred", target_col="Target",
        long_k=200, short_k=200, band=50, rebalance_freq="M",
        trading_cost_rate=base.TRADING_COST_RATE, slippage_rate=base.SLIPPAGE_RATE,
    )

    return {
        "portfolio_total_return": float(m["total_return"]),
        "portfolio_sharpe": float(m["sharpe"]),
        "portfolio_max_drawdown": float(m["max_drawdown"]),
        "portfolio_avg_turnover": float(m["avg_turnover"]),
    }
