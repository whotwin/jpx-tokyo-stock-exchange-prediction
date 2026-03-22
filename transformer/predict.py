# Transformer Prediction and Evaluation Functions

import gc
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from .config import (
    device, SEQ_LENGTH, TRANSFORMER_D_MODEL, TRANSFORMER_NUM_HEADS,
    TRANSFORMER_NUM_LAYERS, TRANSFORMER_DROPOUT, TRANSFORMER_EPOCHS,
    TRANSFORMER_LEARNING_RATE, TARGET_HORIZON, TOP_K, BOTTOM_K,
    TRADING_COST_RATE, SLIPPAGE_RATE
)
from .model import TransformerModel
from .train import train_transformer_model, create_train_loader, create_test_loader


def predict_transformer(model, test_loader):
    """Predict with Transformer model."""
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch_data in test_loader:
            batch_X = batch_data[0].to(device)
            pred = model(batch_X)
            predictions.extend(pred.cpu().numpy())
    return np.array(predictions)


# ============== Evaluation Functions ==============

def evaluate_portfolio(pred_df):
    """Evaluate portfolio performance."""
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

    sharpe = avg_spread / (std_spread + 1e-8) * np.sqrt(252)
    avg_hit = daily_df["hit_ratio"].mean()

    return {
        "num_days": len(daily_df),
        "sharpe": sharpe,
        "hit_ratio": avg_hit,
        "spread": avg_spread
    }


def predict_with_transformer(df, feature_cols, target_col):
    """Run Transformer prediction with expanding window training."""
    print(f"[INFO] Running Transformer with expanding window training...")

    df = df.copy()
    df["Year"] = df["Date"].dt.year

    # Step 1: Normalize features per stock
    from .data import create_sequences
    print("[INFO] Step 1: Creating sequences from all data...")
    X, y, dates_arr, codes_arr = create_sequences(
        df, feature_cols, seq_length=SEQ_LENGTH, target_col=target_col
    )

    if len(X) == 0:
        print("[INFO] No sequences created!")
        return pd.DataFrame()

    print(f"[INFO] Created sequences: X={X.shape}")

    # Convert dates for filtering
    dates_pd = pd.to_datetime(dates_arr)
    years = dates_pd.year

    pred_parts = []

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

        model = TransformerModel(
            input_size=input_size,
            d_model=TRANSFORMER_D_MODEL,
            num_heads=TRANSFORMER_NUM_HEADS,
            num_layers=TRANSFORMER_NUM_LAYERS,
            dropout=TRANSFORMER_DROPOUT
        )
        model = train_transformer_model(model, train_loader, epochs=TRANSFORMER_EPOCHS, lr=TRANSFORMER_LEARNING_RATE)

        test_loader = create_test_loader(X_test)
        pred = predict_transformer(model, test_loader)

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

        model = TransformerModel(
            input_size=input_size,
            d_model=TRANSFORMER_D_MODEL,
            num_heads=TRANSFORMER_NUM_HEADS,
            num_layers=TRANSFORMER_NUM_LAYERS,
            dropout=TRANSFORMER_DROPOUT
        )
        model = train_transformer_model(model, train_loader, epochs=TRANSFORMER_EPOCHS, lr=TRANSFORMER_LEARNING_RATE)

        test_loader = create_test_loader(X_test)
        pred = predict_transformer(model, test_loader)

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


def calc_spread_return_sharpe(df: pd.DataFrame, portfolio_size: int = 200, toprank_weight_ratio: float = 2) -> float:
    """Kaggle official evaluation function."""
    def _calc_spread_return_per_day(df, portfolio_size, toprank_weight_ratio):
        df = df.sort_values(["Date", "Rank"]).reset_index(drop=True)

        actual_size = min(len(df), portfolio_size)
        if actual_size < 10:
            return pd.Series([0.0] * len(df), index=df.index)

        weights = np.linspace(start=toprank_weight_ratio, stop=1, num=actual_size)

        long_weights = weights[:actual_size // 2]
        long_returns = df.groupby("Date").apply(
            lambda x: (x["Target"].iloc[:actual_size // 2] * long_weights).sum()
        )

        short_weights = weights[actual_size // 2:][::-1]
        short_returns = df.groupby("Date").apply(
            lambda x: (x["Target"].iloc[actual_size // 2:actual_size] * short_weights).sum()
        )

        spread_returns = long_returns - short_returns

        return spread_returns

    spread_returns = _calc_spread_return_per_day(df, portfolio_size, toprank_weight_ratio)

    daily_sharpe = spread_returns.mean() / (spread_returns.std() + 1e-8)
    annualized_sharpe = daily_sharpe * np.sqrt(252)

    return annualized_sharpe


def evaluate_predictions(pred_df, year):
    """Evaluate predictions for a specific year."""
    pred_df = pred_df.copy()
    pred_df["Rank"] = pred_df.groupby("Date")["pred"].rank(method="first", ascending=False)

    sharpe = calc_spread_return_sharpe(pred_df, portfolio_size=200)

    rmse = np.sqrt(np.mean((pred_df["y_true"] - pred_df["pred"]) ** 2))

    spearman_corr = pred_df.groupby("Date").apply(
        lambda x: x["y_true"].corr(x["pred"], method="spearman")
    ).mean()

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
