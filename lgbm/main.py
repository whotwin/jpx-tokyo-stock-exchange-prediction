# LightGBM Main Entry Point

import os
import sys
import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lgbm.config import OUTPUT_DIR
from lgbm.data import load_dataset, predict_timeseries_lgbm, evaluate_predictions, evaluate_portfolio_from_predictions


def main():
    """Main function to run LightGBM model."""
    start_time = time.time()

    print("=" * 60)
    print("LGBM Walkforward - 20d Horizon - Stock+All")
    print("=" * 60)

    # Load dataset
    print("\n[1/3] Loading data...")
    data, feature_cols, target_col = load_dataset()

    # Run prediction
    print("\n[2/3] Running LGBM prediction...")
    pred = predict_timeseries_lgbm(data, feature_cols, target_col)

    # Save predictions
    print("\n[3/3] Saving results...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pred_path = os.path.join(OUTPUT_DIR, "predictions.csv")
    pred.to_csv(pred_path, index=False)
    print(f"Saved predictions: {pred_path}")

    # Evaluate
    stat = evaluate_predictions(pred)
    port = evaluate_portfolio_from_predictions(pred)

    # Print metrics
    print("\n" + "=" * 40)
    print("PREDICTION METRICS")
    print("=" * 40)
    print(f"Rows: {stat['rows']:,}")
    print(f"Days: {stat['days']}")
    print(f"RMSE: {stat['rmse']:.6f}")
    print(f"MAE: {stat['mae']:.6f}")
    print(f"Pearson Corr: {stat['pearson_corr']:.4f}")
    print(f"Spearman Corr: {stat['spearman_corr']:.4f}")
    print(f"Hit Ratio: {stat['hit_ratio']:.2%}")
    print(f"Mean Daily RankIC: {stat['mean_daily_rankic']:.4f}")
    print(f"RankIC IR: {stat['rankic_ir']:.4f}")

    print("\n" + "=" * 40)
    print("PORTFOLIO METRICS")
    print("=" * 40)
    print(f"Total Return: {port['portfolio_total_return']:.2%}")
    print(f"Sharpe Ratio: {port['portfolio_sharpe']:.2f}")
    print(f"Max Drawdown: {port['portfolio_max_drawdown']:.2%}")
    print(f"Avg Turnover: {port['portfolio_avg_turnover']:.2%}")

    # Save metrics
    metrics = {**stat, **port}
    import pandas as pd
    metrics_df = pd.DataFrame([metrics])
    metrics_path = os.path.join(OUTPUT_DIR, "metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\nSaved metrics: {metrics_path}")

    total_time = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"Done! Total time: {total_time/60:.1f} minutes")
    print("=" * 60)


if __name__ == "__main__":
    main()
