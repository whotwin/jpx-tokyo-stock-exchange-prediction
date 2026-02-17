"""
LGBM Walkforward Prediction - 20d Horizon with All Data Sources

Best performing model from horizon_model_comparison.py:
- Model: timeseries_lgbm_walkforward
- Horizon: 20d
- Data source: stock+all (stock_prices + secondary + options + trades + financials)
- Test Year: 2021

Output:
- output_lgbm_20d_all/predictions.csv
- output_lgbm_20d_all/metrics.csv
"""

import os
import warnings
import gc

import matplotlib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

import demo_v2 as base

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

OUTPUT_DIR = "output_lgbm_20d_all"
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

TEST_YEAR = 2021
ROLL_TRAIN_YEARS = 2
ROLL_RETRAIN_FREQ = "M"
TARGET_HORIZON = 20
DATA_CONFIG = "stock+all"


def log(msg):
    print(f"[INFO] {msg}")


def build_forward_return_labels(stock_prices, horizons):
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
    ds = pd.Series(sorted(pd.unique(pd.to_datetime(dates))))
    per = ds.dt.to_period(freq)
    for p in per.drop_duplicates():
        m = per == p
        dd = ds[m]
        yield dd.min(), dd.max()


def fit_lgbm(train_df, feature_cols, target_col):
    model = LGBMRegressor(**base.params())
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
    if pred_df.empty:
        return {
            "portfolio_total_return": np.nan, "portfolio_sharpe": np.nan,
            "portfolio_max_drawdown": np.nan, "portfolio_avg_turnover": np.nan,
        }

    tmp = pred_df[["Date", "SecuritiesCode", "pred", "y_true"]].copy()
    tmp = tmp.rename(columns={"y_true": "Target"})

    _, daily_perf, m = base.construct_rank_band_portfolio(
        tmp, pred_col="pred", target_col="Target",
        long_k=200, short_k=200, band=50, rebalance_freq="M",  # Monthly for 20d horizon
        trading_cost_rate=base.TRADING_COST_RATE, slippage_rate=base.SLIPPAGE_RATE,
    )

    return {
        "portfolio_total_return": float(m["total_return"]),
        "portfolio_sharpe": float(m["sharpe"]),
        "portfolio_max_drawdown": float(m["max_drawdown"]),
        "portfolio_avg_turnover": float(m["avg_turnover"]),
    }


def plot_predictions(pred_df):
    """Plot prediction analysis."""
    # Daily RankIC over time
    daily_ic = pred_df.groupby("Date").apply(
        lambda g: g["y_true"].corr(g["pred"], method="spearman") if len(g) > 1 else np.nan
    ).dropna()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Daily RankIC
    ax1 = axes[0, 0]
    ax1.plot(daily_ic.index, daily_ic.values, alpha=0.7)
    ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax1.set_title("Daily RankIC Over Time")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Spearman Correlation")
    ax1.grid(True, alpha=0.3)

    # 2. Prediction distribution
    ax2 = axes[0, 1]
    ax2.hist(pred_df["pred"], bins=50, alpha=0.7, label="Predicted")
    ax2.hist(pred_df["y_true"], bins=50, alpha=0.7, label="Actual")
    ax2.set_title("Prediction vs Actual Distribution")
    ax2.set_xlabel("Return")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Cumulative return
    tmp = pred_df[["Date", "SecuritiesCode", "pred", "y_true"]].copy()
    tmp = tmp.rename(columns={"y_true": "Target"})
    _, daily_perf, _ = base.construct_rank_band_portfolio(
        tmp, pred_col="pred", target_col="Target",
        long_k=200, short_k=200, band=50, rebalance_freq="M",  # Monthly for 20d horizon
        trading_cost_rate=base.TRADING_COST_RATE, slippage_rate=base.SLIPPAGE_RATE,
    )
    if not daily_perf.empty:
        ax3 = axes[1, 0]
        ax3.plot(daily_perf["Date"], daily_perf["cumulative_return"])
        ax3.set_title("Portfolio Cumulative Return")
        ax3.set_xlabel("Date")
        ax3.set_ylabel("Cumulative Return")
        ax3.grid(True, alpha=0.3)

    # 4. Hit ratio by month
    pred_df["month"] = pd.to_datetime(pred_df["Date"]).dt.month
    monthly_hit = pred_df.groupby("month").apply(
        lambda g: np.mean(np.sign(g["y_true"]) == np.sign(g["pred"]))
    )
    ax4 = axes[1, 1]
    ax4.bar(monthly_hit.index, monthly_hit.values)
    ax4.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
    ax4.set_title("Hit Ratio by Month")
    ax4.set_xlabel("Month")
    ax4.set_ylabel("Hit Ratio")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "prediction_analysis.png"), dpi=150, bbox_inches="tight")
    plt.close()
    log(f"Saved: {PLOT_DIR}/prediction_analysis.png")


def main():
    import time
    start_time = time.time()

    log("=" * 60)
    log("LGBM Walkforward - 20d Horizon - Stock+All")
    log("=" * 60)

    # Load dataset
    data, feature_cols, target_col = load_dataset()

    # Run prediction
    pred = predict_timeseries_lgbm(data, feature_cols, target_col)

    # Save predictions
    pred_path = os.path.join(OUTPUT_DIR, "predictions.csv")
    pred.to_csv(pred_path, index=False)
    log(f"Saved predictions: {pred_path}")

    # Evaluate
    stat = evaluate_predictions(pred)
    port = evaluate_portfolio_from_predictions(pred)

    # Print metrics
    log("\n" + "=" * 40)
    log("PREDICTION METRICS")
    log("=" * 40)
    log(f"Rows: {stat['rows']:,}")
    log(f"Days: {stat['days']}")
    log(f"RMSE: {stat['rmse']:.6f}")
    log(f"MAE: {stat['mae']:.6f}")
    log(f"Pearson Corr: {stat['pearson_corr']:.4f}")
    log(f"Spearman Corr: {stat['spearman_corr']:.4f}")
    log(f"Hit Ratio: {stat['hit_ratio']:.2%}")
    log(f"Mean Daily RankIC: {stat['mean_daily_rankic']:.4f}")
    log(f"RankIC IR: {stat['rankic_ir']:.4f}")

    log("\n" + "=" * 40)
    log("PORTFOLIO METRICS")
    log("=" * 40)
    log(f"Total Return: {port['portfolio_total_return']:.2%}")
    log(f"Sharpe Ratio: {port['portfolio_sharpe']:.2f}")
    log(f"Max Drawdown: {port['portfolio_max_drawdown']:.2%}")
    log(f"Avg Turnover: {port['portfolio_avg_turnover']:.2%}")

    # Save metrics
    metrics = {**stat, **port}
    metrics_df = pd.DataFrame([metrics])
    metrics_path = os.path.join(OUTPUT_DIR, "metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    log(f"\nSaved metrics: {metrics_path}")

    # Plot analysis
    plot_predictions(pred)

    total_time = time.time() - start_time
    log(f"\n{'=' * 60}")
    log(f"Done! Total time: {total_time/60:.1f} minutes")
    log("=" * 60)


if __name__ == "__main__":
    main()
