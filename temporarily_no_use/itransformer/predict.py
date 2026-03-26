# iTransformer Prediction and Evaluation Functions

import gc
import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import torch
from torch.utils.data import DataLoader, TensorDataset

from .config import (
    log, device, SEQ_LENGTH, TARGET_HORIZON, TOP_K, BOTTOM_K,
    TRADING_COST_RATE, SLIPPAGE_RATE,
    ITRANSFORMER_BATCH_SIZE, ITRANSFORMER_EPOCHS, ITRANSFORMER_LEARNING_RATE,
    ITRANSFORMER_D_MODEL, ITRANSFORMER_NUM_LAYERS, ITRANSFORMER_NUM_HEADS,
    OUTPUT_DIR,
)

from .train import train_itransformer_model, cross_sectional_normalize, FeatureProjector

# Import the iTransformer model class
try:
    from iTransformer import iTransformer
except ImportError:
    # Fallback: try importing from new_transformer_embedding
    try:
        from new_transformer_embedding import StockEmbeddingTransformer as iTransformer
    except ImportError:
        iTransformer = None


def predict_itransformer(model, test_loader, projector=None):
    """Predict with iTransformer model."""
    model.eval()
    if projector is not None:
        projector.eval()
    predictions = []
    with torch.no_grad():
        for batch_data in test_loader:
            batch_X = batch_data[0].to(device)
            # batch_X: (batch, seq_length, num_stocks, num_features)

            # Apply feature projection if provided
            if projector is not None:
                batch_X = projector(batch_X)
                # Now batch_X is (batch, seq_len, num_stocks, dim)

            # Reshape to 3D for iTransformer: (batch, seq_len, num_stocks * dim)
            batch_size, seq_len, num_stocks_val, dim = batch_X.shape
            batch_X = batch_X.reshape(batch_size, seq_len, num_stocks_val * dim)

            pred = model(batch_X)
            pred = pred[:, 0, :].cpu().numpy()
            predictions.append(pred)
    return np.concatenate(predictions, axis=0)


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

    pred_df = pred_df.dropna(subset=["y_true", "pred"])

    y = pred_df["y_true"].values
    p = pred_df["pred"].values

    valid_mask = np.isfinite(y) & np.isfinite(p)
    y = y[valid_mask]
    p = p[valid_mask]

    if len(y) == 0:
        return {"rmse": np.nan, "spearman": np.nan, "hit": np.nan}

    rmse = float(np.sqrt(mean_squared_error(y, p)))
    spearman = float(pd.Series(y).corr(pd.Series(p), method="spearman"))
    hit = float(np.mean(np.sign(y) == np.sign(p)))

    return {"rmse": rmse, "spearman": spearman, "hit": hit}


def predict_with_itransformer(df, feature_cols, stock_codes):
    """Run iTransformer prediction with expanding window training and multiple features."""
    if iTransformer is None:
        log("ERROR: iTransformer model class not found!")
        log("Please make sure iTransformer is installed or importable")
        return pd.DataFrame()

    log(f"Running iTransformer with expanding window training...")
    log(f"Using {len(stock_codes)} stocks")

    df = df.copy()
    df["Year"] = df["Date"].dt.year

    # Import data preparation functions
    from .data import create_itransformer_sequences

    # Prepare sequences - returns (X, y, dates_arr, stock_codes, num_features)
    X, y, dates_arr, _, num_features = create_itransformer_sequences(
        df, feature_cols, stock_codes,
        seq_length=SEQ_LENGTH, target_col="target_30d"
    )

    if len(X) == 0:
        log("No sequences created!")
        return pd.DataFrame()

    # Get num_stocks from X shape
    # X shape: (samples, seq_length, num_stocks, num_features)
    num_stocks = X.shape[2]

    log(f"Created sequences: X={X.shape}, y={y.shape}")
    log(f"Shape: (batch, seq_len, num_stocks, num_features) = {X.shape}")
    log(f"num_stocks={num_stocks}, num_features={num_features}")

    # Define expanding window training rounds
    # Format: ((train_years tuple), test_year)
    train_years_list = [
        ((2017,), 2018),
        ((2017, 2018), 2019),
        ((2017, 2018, 2019), 2020),
        ((2017, 2018, 2019, 2020), 2021),
    ]

    all_predictions = []
    all_metrics = []

    # Run each expanding window round
    for train_years, test_year in train_years_list:
        log("\n" + "="*50)
        log(f"Round: Train {train_years} -> Test {test_year}")
        log("="*50)

        # Filter data
        dates_pd = pd.to_datetime(dates_arr)
        years = dates_pd.year

        train_mask = years.isin(train_years)
        test_mask = years == test_year

        X_train = X[train_mask]
        y_train = y[train_mask]
        X_test = X[test_mask]
        y_test = y[test_mask]
        dates_test = dates_arr[test_mask]

        log(f"  Train: {len(X_train):,}, Test: {len(X_test):,}")

        if len(X_train) == 0 or len(X_test) == 0:
            log(f"  No data for this round, skipping...")
            continue

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

        # Apply cross-sectional normalization (handles 4D tensor)
        X_train_norm = cross_sectional_normalize(X_train_norm, num_features=num_features)
        X_test_norm = cross_sectional_normalize(X_test_norm, num_features=num_features)

        log(f"  Feature stats: mean={np.nanmean(x_mean):.6f}, std={np.nanmean(x_std):.6f}")
        log(f"  Target stats: mean={np.nanmean(y_mean):.4f}, std={np.nanmean(y_std):.4f}")

        # Create feature projector: projects num_features -> d_model
        projector = FeatureProjector(num_features=num_features, dim=ITRANSFORMER_D_MODEL)

        # Train model
        train_dataset = TensorDataset(torch.FloatTensor(X_train_norm), torch.FloatTensor(y_train_norm))
        train_loader = DataLoader(train_dataset, batch_size=ITRANSFORMER_BATCH_SIZE, shuffle=True, num_workers=0)

        # iTransformer expects: (batch, seq_len, num_stocks * d_model) after projection
        model = iTransformer(
            num_variates=num_stocks * ITRANSFORMER_D_MODEL,
            lookback_len=SEQ_LENGTH,
            depth=ITRANSFORMER_NUM_LAYERS,
            dim=ITRANSFORMER_D_MODEL,
            heads=ITRANSFORMER_NUM_HEADS,
            pred_length=1,
            flash_attn=True
        )
        model = train_itransformer_model(model, train_loader, epochs=ITRANSFORMER_EPOCHS, lr=ITRANSFORMER_LEARNING_RATE, projector=projector)

        # Predict
        test_dataset = TensorDataset(torch.FloatTensor(X_test_norm))
        test_loader = DataLoader(test_dataset, batch_size=ITRANSFORMER_BATCH_SIZE, shuffle=False, num_workers=0)
        pred_norm = predict_itransformer(model, test_loader, projector=projector, num_stocks=num_stocks)

        # Extract predictions - now shape is (samples, num_stocks * d_model)
        pred_reshaped = pred_norm.reshape(pred_norm.shape[0], num_stocks, ITRANSFORMER_D_MODEL)
        pred_per_stock = pred_reshaped[:, :, 0]  # Take first dimension as prediction

        # Denormalize
        pred = pred_per_stock * y_std[0, :] + y_mean[0, :]

        # Flatten predictions and targets
        pred_parts = []
        for i, date in enumerate(dates_test):
            for j, code in enumerate(stock_codes):
                if j < pred.shape[1] and not np.isnan(y_test[i, j]):
                    pred_parts.append({
                        "Date": date,
                        "SecuritiesCode": code,
                        "y_true": y_test[i, j],
                        "pred": pred[i, j],
                        "train_year": max(train_years)
                    })

        pred_df = pd.DataFrame(pred_parts)
        all_predictions.append(pred_df)

        # Evaluate
        if len(pred_df) > 0:
            # Prediction accuracy metrics
            pred_metrics = evaluate_predictions(pred_df)
            log(f"  Prediction Metrics: RMSE={pred_metrics['rmse']:.6f}, Spearman={pred_metrics['spearman']:.4f}, Hit={pred_metrics['hit']:.4f}")

            # Portfolio metrics
            portfolio_metrics = evaluate_portfolio(pred_df)
            log(f"  Portfolio Metrics: Sharpe={portfolio_metrics['sharpe']:.4f}, Hit Ratio={portfolio_metrics['hit_ratio']:.4f}, Spread={portfolio_metrics['spread']:.6f}")

            # Kaggle metrics
            kaggle_metrics = evaluate_portfolio_kaggle(pred_df)
            log(f"  Kaggle Sharpe: {kaggle_metrics['kaggle_sharpe']:.4f}")

            # Save metrics
            round_metrics = {
                "train_years": str(train_years),
                "test_year": test_year,
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "pred_samples": len(pred_df),
                "rmse": pred_metrics['rmse'],
                "spearman": pred_metrics['spearman'],
                "hit": pred_metrics['hit'],
                "sharpe": portfolio_metrics['sharpe'],
                "hit_ratio": portfolio_metrics['hit_ratio'],
                "spread": portfolio_metrics['spread'],
                "kaggle_sharpe": kaggle_metrics['kaggle_sharpe'],
            }
            all_metrics.append(round_metrics)

        del model
        gc.collect()

    # Combine all predictions
    if all_predictions:
        out = pd.concat(all_predictions, ignore_index=True)
        out = out.sort_values(["Date", "SecuritiesCode"]).reset_index(drop=True)
    else:
        out = pd.DataFrame()

    # Save metrics to CSV
    if all_metrics:
        metrics_df = pd.DataFrame(all_metrics)
        metrics_path = os.path.join(OUTPUT_DIR, "itransformer_metrics.csv")
        metrics_df.to_csv(metrics_path, index=False)
        log(f"\nMetrics saved to: {metrics_path}")
        log("\n" + "="*50)
        log("Summary of all rounds:")
        log("="*50)
        log(metrics_df.to_string(index=False))

    log(f"\nTotal predictions: {len(out):,}")
    return out
