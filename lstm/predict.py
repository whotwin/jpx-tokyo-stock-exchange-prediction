# LSTM Prediction and Evaluation Functions

import gc
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from .config import (
    device, SEQ_LENGTH, LSTM_HIDDEN_SIZE, LSTM_NUM_LAYERS,
    LSTM_DROPOUT, LSTM_EPOCHS, LSTM_LEARNING_RATE, TARGET_HORIZON,
    TOP_K, BOTTOM_K
)
from .model import LSTMModel
from .train import train_lstm_model, create_train_loader, create_test_loader


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
        # Sort by date and rank (ascending)
        df = df.sort_values(["Date", "Rank"]).reset_index(drop=True)

        # Calculate actual number of stocks (handle case when stocks < portfolio_size)
        actual_size = min(len(df), portfolio_size)
        if actual_size < 10:
            return pd.Series([0.0] * len(df), index=df.index)

        # Weight: linear from toprank_weight_ratio to 1
        weights = np.linspace(start=toprank_weight_ratio, stop=1, num=actual_size)

        # Long positions (top ranks)
        long_weights = weights[:actual_size // 2]
        long_returns = df.groupby("Date").apply(
            lambda x: (x["Target"].iloc[:actual_size // 2] * long_weights).sum()
        )

        # Short positions (bottom ranks)
        short_weights = weights[actual_size // 2:][::-1]
        short_returns = df.groupby("Date").apply(
            lambda x: (x["Target"].iloc[actual_size // 2:actual_size] * short_weights).sum()
        )

        # Spread return
        spread_returns = long_returns - short_returns

        return spread_returns

    # Calculate daily spread returns
    spread_returns = _calc_spread_return_per_day(df, portfolio_size, toprank_weight_ratio)

    # Annualize (sqrt(252) for daily data)
    daily_sharpe = spread_returns.mean() / (spread_returns.std() + 1e-8)
    annualized_sharpe = daily_sharpe * np.sqrt(252)

    return annualized_sharpe


def evaluate_predictions(pred_df, year):
    """Evaluate predictions for a specific year."""
    # Add rank based on predictions
    pred_df = pred_df.copy()
    pred_df["Rank"] = pred_df.groupby("Date")["pred"].rank(method="first", ascending=False)

    # Calculate Sharpe
    sharpe = calc_spread_return_sharpe(pred_df, portfolio_size=200)

    # Calculate RMSE
    rmse = np.sqrt(np.mean((pred_df["y_true"] - pred_df["pred"]) ** 2))

    # Calculate Spearman correlation
    spearman_corr = pred_df.groupby("Date").apply(
        lambda x: x["y_true"].corr(x["pred"], method="spearman")
    ).mean()

    # Calculate Hit Ratio (direction accuracy)
    pred_df["pred_direction"] = (pred_df["pred"] > 0).astype(int)
    pred_df["true_direction"] = (pred_df["y_true"] > 0).astype(int)
    hit_ratio = (pred_df["pred_direction"] == pred_df["true_direction"]).mean()

    return {
        "year": year,
        "rmse": rmse,
        "spearman": spearman_corr,
        "hit_ratio": hit_ratio,
        "sharpe": sharpe,
        "num_samples": len(pred_df)
    }


def predict_with_lstm(df, feature_cols, target_col):
    """Run LSTM prediction with expanding window training."""
    print(f"[INFO] Running LSTM with expanding window training...")

    df = df.copy()
    df["Year"] = df["Date"].dt.year

    # Step 1: Apply cross-sectional normalization across stocks at each time step
    from .data import cross_sectional_normalize
    print("[INFO] Step 1: Applying cross-sectional normalization...")
    df_norm = cross_sectional_normalize(df, feature_cols)

    # Step 2: Create sequences from ALL normalized data
    print("[INFO] Step 2: Creating sequences from all data...")
    from .data import create_sequences
    X, y, dates_arr, codes_arr = create_sequences(
        df_norm, feature_cols, seq_length=SEQ_LENGTH, target_col=target_col
    )

    if len(X) == 0:
        print("[INFO] No sequences created!")
        return pd.DataFrame()

    print(f"[INFO] Created sequences: X={X.shape}")

    # Convert dates for filtering
    dates_pd = pd.to_datetime(dates_arr)
    years = dates_pd.year

    pred_parts = []

    # ======== 2018 Prediction (Validation) ========
    print("Training for 2018 prediction...")
    train_mask = years < 2018
    test_mask = years == 2018

    X_train = X[train_mask]
    y_train = y[train_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]
    dates_test = dates_arr[test_mask]
    codes_test = codes_arr[test_mask]

    print(f"  2018: Train {len(X_train):,}, Test {len(X_test):,}")

    if len(X_train) > 0 and len(X_test) > 0:
        train_loader = create_train_loader(X_train, y_train)
        input_size = X_train.shape[2]

        model = LSTMModel(
            input_size=input_size,
            hidden_size=LSTM_HIDDEN_SIZE,
            num_layers=LSTM_NUM_LAYERS,
            dropout=LSTM_DROPOUT
        )
        model = train_lstm_model(model, train_loader, epochs=LSTM_EPOCHS, lr=LSTM_LEARNING_RATE)

        test_loader = create_test_loader(X_test)
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
    print("Training for 2019 prediction...")
    train_mask = years < 2019
    test_mask = years == 2019

    X_train = X[train_mask]
    y_train = y[train_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]
    dates_test = dates_arr[test_mask]
    codes_test = codes_arr[test_mask]

    print(f"  2019: Train {len(X_train):,}, Test {len(X_test):,}")

    if len(X_train) > 0 and len(X_test) > 0:
        train_loader = create_train_loader(X_train, y_train)
        input_size = X_train.shape[2]

        model = LSTMModel(
            input_size=input_size,
            hidden_size=LSTM_HIDDEN_SIZE,
            num_layers=LSTM_NUM_LAYERS,
            dropout=LSTM_DROPOUT
        )
        model = train_lstm_model(model, train_loader, epochs=LSTM_EPOCHS, lr=LSTM_LEARNING_RATE)

        test_loader = create_test_loader(X_test)
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
    print("Training for 2020 prediction...")
    train_mask = years < 2020
    test_mask = years == 2020

    X_train = X[train_mask]
    y_train = y[train_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]
    dates_test = dates_arr[test_mask]
    codes_test = codes_arr[test_mask]

    print(f"  2020: Train {len(X_train):,}, Test {len(X_test):,}")

    if len(X_train) > 0 and len(X_test) > 0:
        train_loader = create_train_loader(X_train, y_train)
        input_size = X_train.shape[2]

        model = LSTMModel(
            input_size=input_size,
            hidden_size=LSTM_HIDDEN_SIZE,
            num_layers=LSTM_NUM_LAYERS,
            dropout=LSTM_DROPOUT
        )
        model = train_lstm_model(model, train_loader, epochs=LSTM_EPOCHS, lr=LSTM_LEARNING_RATE)

        test_loader = create_test_loader(X_test)
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
    print("Training for 2021 prediction (Final)...")
    train_mask = years < 2021
    test_mask = years == 2021

    X_train = X[train_mask]
    y_train = y[train_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]
    dates_test = dates_arr[test_mask]
    codes_test = codes_arr[test_mask]

    print(f"  2021: Train {len(X_train):,}, Test {len(X_test):,}")

    if len(X_train) > 0 and len(X_test) > 0:
        train_loader = create_train_loader(X_train, y_train)
        input_size = X_train.shape[2]

        model = LSTMModel(
            input_size=input_size,
            hidden_size=LSTM_HIDDEN_SIZE,
            num_layers=LSTM_NUM_LAYERS,
            dropout=LSTM_DROPOUT
        )
        model = train_lstm_model(model, train_loader, epochs=LSTM_EPOCHS, lr=LSTM_LEARNING_RATE)

        test_loader = create_test_loader(X_test)
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

    if not pred_parts:
        return pd.DataFrame()

    result = pd.concat(pred_parts, ignore_index=True)

    # Evaluate each year
    metrics = []
    for year in result["train_year"].unique():
        year_df = result[result["train_year"] == year]
        metrics.append(evaluate_predictions(year_df, year))

    metrics_df = pd.DataFrame(metrics)
    print(f"\n[INFO] Evaluation Results:")
    print(metrics_df.to_string(index=False))

    return result
