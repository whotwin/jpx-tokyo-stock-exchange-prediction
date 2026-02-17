"""
Compare prediction horizons and model types on JPX data - OPTIMIZED VERSION

Matrix:
- Horizons: 1d, 5d, 20d forward return per stock
- Data Sources: stock_only, stock+all (reduced for speed)
- Models:
  1) timeseries_lgbm_walkforward  (monthly retrain, rolling 2-year train window)
  2) ordinary_ridge_static        (single fit on all pre-2021 data)
  3) lstm_walkforward            (PyTorch LSTM, monthly retrain) - OPTIMIZED
  4) transformer_walkforward     (PyTorch Transformer, monthly retrain) - OPTIMIZED

Outputs:
- output_horizon_compare/horizon_model_metrics.csv
- output_horizon_compare/pred_{model}_{datasource}_{horizon}d.csv
- output_horizon_compare/plots/*.png
"""

import os
import warnings
import gc

import matplotlib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

import demo_v2 as base

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# Check PyTorch availability
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. LSTM and Transformer models will be skipped.")

OUTPUT_DIR = "output_horizon_compare"
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

TEST_YEAR = 2021
ROLL_TRAIN_YEARS = 2
ROLL_RETRAIN_FREQ = "M"
HORIZONS = [1, 5, 20]

# Reduced data sources for speed
DATA_SOURCES = [
    "stock_only",
    "stock+all",
]

# Model types - only include available models
MODELS = ["timeseries_lgbm_walkforward"]
if TORCH_AVAILABLE:
    MODELS.append("lstm_walkforward")

# OPTIMIZED Deep learning hyperparameters
SEQ_LENGTH = 10  # Reduced from 20
LSTM_HIDDEN_DIM = 32  # Reduced from 64
LSTM_NUM_LAYERS = 1  # Reduced from 2
LSTM_DROPOUT = 0.1
TRANSFORMER_D_MODEL = 32  # Reduced from 64
TRANSFORMER_NHEAD = 2  # Reduced from 4
TRANSFORMER_NUM_LAYERS = 1  # Reduced from 2
TRANSFORMER_DROPOUT = 0.1
BATCH_SIZE = 512  # Increased
NUM_EPOCHS = 10  # Reduced from 30
LEARNING_RATE = 0.002
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cache for loaded data
_CACHED_DATA = {}
_CACHED_SOURCES = None


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
    """Load all data once and cache it."""
    global _CACHED_SOURCES
    if _CACHED_SOURCES is None:
        log("Loading all data sources (cached)...")
        _CACHED_SOURCES = base.load_data_sources("train_files")
        log("Building full feature table (cached)...")
        full_df, groups = base.build_feature_table(
            sources=_CACHED_SOURCES,
            start_date="2017-01-04",
            end_date="2021-12-03",
        )
        # Add forward return labels
        labels = build_forward_return_labels(_CACHED_SOURCES["stock_prices"], HORIZONS)
        full_df = full_df.merge(labels, on=["Date", "SecuritiesCode"], how="left")
        full_df = full_df.sort_values(["Date", "SecuritiesCode"]).reset_index(drop=True)

        _CACHED_DATA["full_df"] = full_df
        _CACHED_DATA["groups"] = groups
        log(f"Cached: {len(full_df)} rows")

    return _CACHED_DATA["full_df"], _CACHED_DATA["groups"], _CACHED_SOURCES


def load_dataset(data_config="stock_only"):
    """Load data with different source configurations - uses cached data."""
    full_df, groups, sources = load_all_data()

    if data_config == "stock_only":
        feature_cols = sorted(groups["stock"])
    elif data_config == "stock+all":
        feature_cols = sorted(set(sum(groups.values(), [])))
    else:
        raise ValueError(f"Unknown data_config: {data_config}")

    data = full_df[["Date", "SecuritiesCode"] + feature_cols + [f"target_{h}d" for h in HORIZONS]].copy()
    return data, feature_cols, sources


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
    test_df = df[df["Date"].dt.year == TEST_YEAR].copy()
    pred_parts = []

    for period_start, period_end in iter_periods(test_df["Date"], freq=ROLL_RETRAIN_FREQ):
        train_end = period_start - pd.Timedelta(days=1)
        train_start = train_end - pd.DateOffset(years=ROLL_TRAIN_YEARS) + pd.Timedelta(days=1)

        train_win = df[(df["Date"] >= train_start) & (df["Date"] <= train_end) & df[target_col].notna()].copy()
        infer_win = test_df[(test_df["Date"] >= period_start) & (test_df["Date"] <= period_end) & test_df[target_col].notna()].copy()

        if train_win.empty or infer_win.empty:
            continue

        model = fit_lgbm(train_win, feature_cols, target_col)
        out = infer_win[["Date", "SecuritiesCode", target_col]].copy()
        out = out.rename(columns={target_col: "y_true"})
        out["pred"] = model.predict(infer_win[feature_cols])
        out["retrain_period_start"] = period_start
        pred_parts.append(out)

    if pred_parts:
        pred = pd.concat(pred_parts, ignore_index=True).sort_values(["Date", "SecuritiesCode"]).reset_index(drop=True)
    else:
        pred = pd.DataFrame(columns=["Date", "SecuritiesCode", "y_true", "pred", "retrain_period_start"])
    pred["model_type"] = "timeseries_lgbm_walkforward"
    return pred


def predict_ordinary_ridge(df, feature_cols, target_col):
    train_df = df[(df["Date"].dt.year < TEST_YEAR) & df[target_col].notna()].copy()
    test_df = df[(df["Date"].dt.year == TEST_YEAR) & df[target_col].notna()].copy()
    if train_df.empty or test_df.empty:
        return pd.DataFrame(columns=["Date", "SecuritiesCode", "y_true", "pred", "model_type"])

    model = Ridge(alpha=1.0)
    model.fit(train_df[feature_cols], train_df[target_col].values)

    out = test_df[["Date", "SecuritiesCode", target_col]].copy()
    out = out.rename(columns={target_col: "y_true"})
    out["pred"] = model.predict(test_df[feature_cols])
    out["model_type"] = "ordinary_ridge_static"
    return out


# ============== OPTIMIZED PyTorch Deep Learning Models ==============

class SequenceDataset(Dataset):
    """Dataset for sequence models."""
    def __init__(self, sequences, targets=None):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets) if targets is not None else None

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        if self.targets is not None:
            return self.sequences[idx], self.targets[idx]
        return self.sequences[idx]


class LSTMModel(nn.Module):
    """Optimized LSTM model."""
    def __init__(self, input_dim, hidden_dim=LSTM_HIDDEN_DIM, num_layers=LSTM_NUM_LAYERS, dropout=LSTM_DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        return self.fc(last_output).squeeze(-1)


class TransformerModel(nn.Module):
    """Optimized Transformer encoder model."""
    def __init__(self, input_dim, d_model=TRANSFORMER_D_MODEL, nhead=TRANSFORMER_NHEAD,
                 num_layers=TRANSFORMER_NUM_LAYERS, dropout=TRANSFORMER_DROPOUT):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dropout=dropout, batch_first=True, activation="gelu")
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.input_proj(x)
        transformer_out = self.transformer(x)
        last_output = transformer_out[:, -1, :]
        return self.fc(last_output).squeeze(-1)


def create_sequences_fast(df, feature_cols, target_col, seq_length=SEQ_LENGTH):
    """Optimized sequence creation - vectorized."""
    df = df.sort_values(["Date", "SecuritiesCode"]).reset_index(drop=True)

    sequences = []
    targets = []
    indices = []

    # Group by securities code
    for code, group in df.groupby("SecuritiesCode"):
        group = group.sort_values("Date").reset_index(drop=True)
        X = group[feature_cols].values
        y = group[target_col].values if target_col else None

        if y is None:
            continue

        # Create sequences for this stock
        for i in range(len(X) - seq_length):
            seq = X[i:i+seq_length]
            tgt = y[i + seq_length]

            if np.isnan(seq).any() or np.isnan(tgt):
                continue

            sequences.append(seq)
            targets.append(tgt)
            indices.append({
                "Date": group.iloc[i + seq_length]["Date"],
                "SecuritiesCode": code
            })

    return np.array(sequences), np.array(targets), indices


def train_deep_learning_model(model, train_loader, val_loader, num_epochs=NUM_EPOCHS, lr=LEARNING_RATE):
    """Train deep learning model with early stopping."""
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    best_val_loss = float('inf')
    best_state = None

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()

        val_loss /= len(val_loader) if len(val_loader) > 0 else 1
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(DEVICE)
    return model


def predict_deep_learning(df, feature_cols, target_col, is_lstm=True):
    """Optimized deep learning prediction with efficient batching."""
    test_df = df[df["Date"].dt.year == TEST_YEAR].copy()

    # Pre-compute sequences for test data (only need to do this once per target_col)
    log("    Creating test sequences...")
    X_test, y_test, test_indices = create_sequences_fast(test_df, feature_cols, target_col, SEQ_LENGTH)

    if len(X_test) == 0:
        return pd.DataFrame(columns=["Date", "SecuritiesCode", "y_true", "pred", "retrain_period_start"])

    # Normalize test data - fit on training data only, then transform test
    # CRITICAL: Use the same scaler for train and test to avoid distribution mismatch
    scaler = StandardScaler()
    X_test_2d = X_test.reshape(-1, X_test.shape[-1])
    X_test_scaled = scaler.fit_transform(X_test_2d).reshape(X_test.shape)

    # Get test dates for grouping
    test_dates = pd.to_datetime([idx["Date"] for idx in test_indices])
    unique_test_dates = sorted(test_dates.unique())

    pred_parts = []

    # Process by period
    for period_start, period_end in iter_periods(test_df["Date"], freq=ROLL_RETRAIN_FREQ):
        train_end = period_start - pd.Timedelta(days=1)
        train_start = train_end - pd.DateOffset(years=ROLL_TRAIN_YEARS) + pd.Timedelta(days=1)

        train_win = df[(df["Date"] >= train_start) & (df["Date"] <= train_end) & df[target_col].notna()].copy()

        if train_win.empty:
            continue

        # Get test samples for this period
        period_mask = (test_dates >= period_start) & (test_dates <= period_end)
        if not period_mask.any():
            continue

        X_period = X_test_scaled[period_mask]
        y_period = y_test[period_mask]
        indices_period = [test_indices[i] for i in range(len(test_indices)) if period_mask[i]]

        if len(X_period) < 100:
            continue

        # Sample training data for efficiency
        train_sample = train_win.sample(n=min(50000, len(train_win)), random_state=42) if len(train_win) > 50000 else train_win

        # Create training sequences
        X_train, y_train, _ = create_sequences_fast(train_sample, feature_cols, target_col, SEQ_LENGTH)

        if len(X_train) < 1000:
            continue

        # Normalize training data - fit scaler on training data only
        train_scaler = StandardScaler()
        X_train_2d = X_train.reshape(-1, X_train.shape[-1])
        X_train_scaled = train_scaler.fit_transform(X_train_2d).reshape(X_train.shape)

        # Transform test data using the same scaler fitted on training data
        X_period_2d = X_period.reshape(-1, X_period.shape[-1])
        X_period_scaled = train_scaler.transform(X_period_2d).reshape(X_period.shape)

        # Split train/val
        val_size = min(500, int(len(X_train_scaled) * 0.1))
        X_val = X_train_scaled[-val_size:]
        y_val = y_train[-val_size:]
        X_train_final = X_train_scaled[:-val_size]
        y_train_final = y_train[:-val_size]

        # Create data loaders
        train_loader = DataLoader(SequenceDataset(X_train_final, y_train_final), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        val_loader = DataLoader(SequenceDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        # Initialize and train model
        input_dim = X_train.shape[-1]
        if is_lstm:
            model = LSTMModel(input_dim)
        else:
            model = TransformerModel(input_dim)

        model = train_deep_learning_model(model, train_loader, val_loader)

        # Make predictions using scaled test data
        model.eval()
        test_loader = DataLoader(SequenceDataset(X_period_scaled), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        predictions = []
        with torch.no_grad():
            for batch_x in test_loader:
                batch_x = batch_x.to(DEVICE)
                preds = model(batch_x)
                predictions.extend(preds.cpu().numpy())

        # Create output
        if len(predictions) > 0:
            out = pd.DataFrame(indices_period)
            out["y_true"] = y_period
            out["pred"] = predictions
            out["retrain_period_start"] = period_start
            pred_parts.append(out)

        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()

    if pred_parts:
        pred = pd.concat(pred_parts, ignore_index=True).sort_values(["Date", "SecuritiesCode"]).reset_index(drop=True)
    else:
        pred = pd.DataFrame(columns=["Date", "SecuritiesCode", "y_true", "pred", "retrain_period_start"])

    return pred


def predict_lstm(df, feature_cols, target_col):
    pred = predict_deep_learning(df, feature_cols, target_col, is_lstm=True)
    pred["model_type"] = "lstm_walkforward"
    return pred


def predict_transformer(df, feature_cols, target_col):
    pred = predict_deep_learning(df, feature_cols, target_col, is_lstm=False)
    pred["model_type"] = "transformer_walkforward"
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


def evaluate_portfolio_from_predictions(pred_df, horizon=1):
    """
    Evaluate portfolio performance with horizon-aware rebalance frequency.

    Args:
        pred_df: DataFrame with predictions
        horizon: Forecast horizon in days (1, 5, 20). Adjusts rebalance frequency to avoid
                 overlapping returns - for longer horizons, we rebalance less frequently
                 to match the holding period.
    """
    if pred_df.empty:
        return {
            "portfolio_total_return": np.nan, "portfolio_sharpe": np.nan,
            "portfolio_max_drawdown": np.nan, "portfolio_avg_turnover": np.nan,
        }

    # Adjust rebalance frequency based on horizon to avoid overlapping returns
    # 1d: weekly (standard), 5d: bi-weekly, 20d: monthly
    if horizon >= 20:
        rebalance_freq = "M"  # Monthly for 20d horizon
    elif horizon >= 5:
        rebalance_freq = "2W-FRI"  # Bi-weekly for 5d horizon
    else:
        rebalance_freq = "W-FRI"  # Weekly for 1d horizon

    tmp = pred_df[["Date", "SecuritiesCode", "pred", "y_true"]].copy()
    tmp = tmp.rename(columns={"y_true": "Target"})

    _, daily_perf, m = base.construct_rank_band_portfolio(
        tmp, pred_col="pred", target_col="Target",
        long_k=200, short_k=200, band=50, rebalance_freq=rebalance_freq,
        trading_cost_rate=base.TRADING_COST_RATE, slippage_rate=base.SLIPPAGE_RATE,
    )

    return {
        "portfolio_total_return": float(m["total_return"]),
        "portfolio_sharpe": float(m["sharpe"]),
        "portfolio_max_drawdown": float(m["max_drawdown"]),
        "portfolio_avg_turnover": float(m["avg_turnover"]),
    }


def plot_heatmap(metrics_df, value_col, title, filename):
    # Filter to single datasource for clean heatmap, or aggregate if multiple
    if "datasource" in metrics_df.columns:
        # If multiple datasources, pick the first one for visualization
        if metrics_df["datasource"].nunique() > 1:
            # Aggregate by taking mean across datasources
            pivot_data = metrics_df.groupby(["horizon", "model_type"])[value_col].mean().reset_index()
            pivot = pivot_data.pivot(index="horizon", columns="model_type", values=value_col)
        else:
            pivot = metrics_df.pivot(index="horizon", columns="model_type", values=value_col)
    else:
        pivot = metrics_df.pivot(index="horizon", columns="model_type", values=value_col)

    # Sort horizon order
    horizon_order = ["1d", "5d", "20d"]
    pivot = pivot.reindex(horizon_order)

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(pivot.values, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=25, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title(title)

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.iloc[i, j]
            txt = "nan" if pd.isna(v) else f"{v:.4f}"
            ax.text(j, i, txt, ha="center", va="center", color="white")

    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, filename), dpi=150, bbox_inches="tight")
    plt.close()


def plot_datasource_comparison(metrics_df, filename):
    pivot = metrics_df.pivot(index="datasource", columns="horizon", values="mean_daily_rankic")
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(pivot.values, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Mean Daily RankIC by Data Source x Horizon")

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.iloc[i, j]
            txt = "nan" if pd.isna(v) else f"{v:.4f}"
            ax.text(j, i, txt, ha="center", va="center", color="white")

    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, filename), dpi=150, bbox_inches="tight")
    plt.close()


def main():
    import time
    start_time = time.time()

    log("=" * 60)
    log("OPTIMIZED Horizon Model Comparison")
    log("=" * 60)
    log(f"Data sources: {DATA_SOURCES}")
    log(f"Models: {MODELS}")
    log(f"Horizons: {HORIZONS}")
    log(f"PyTorch available: {TORCH_AVAILABLE}")
    log(f"Device: {DEVICE}")
    log("=" * 60)

    # Pre-load all data once
    log("\n=== Loading and caching data ===")
    load_all_data()

    all_metrics = []
    datasource_metrics = []

    # Run experiments
    for ds in DATA_SOURCES:
        log(f"\n=== Data source: {ds} ===")
        data, feature_cols, _ = load_dataset(ds)
        log(f"Features: {len(feature_cols)}")

        for h in HORIZONS:
            target_col = f"target_{h}d"
            log(f"  Horizon={h}d")

            h_df = data[["Date", "SecuritiesCode"] + feature_cols + [target_col]].copy()

            for model_type in MODELS:
                iter_start = time.time()
                log(f"    Model={model_type}")

                if model_type == "timeseries_lgbm_walkforward":
                    pred = predict_timeseries_lgbm(h_df, feature_cols, target_col)
                elif model_type == "ordinary_ridge_static":
                    pred = predict_ordinary_ridge(h_df, feature_cols, target_col)
                elif model_type == "lstm_walkforward":
                    if not TORCH_AVAILABLE:
                        continue
                    pred = predict_lstm(h_df, feature_cols, target_col)
                elif model_type == "transformer_walkforward":
                    if not TORCH_AVAILABLE:
                        continue
                    pred = predict_transformer(h_df, feature_cols, target_col)
                else:
                    continue

                iter_time = time.time() - iter_start
                log(f"      Completed in {iter_time:.1f}s, {len(pred)} predictions")

                if pred.empty:
                    continue

                pred["datasource"] = ds
                pred["horizon"] = f"{h}d"

                # Save predictions
                pred_path = os.path.join(OUTPUT_DIR, f"pred_{model_type}_{ds}_{h}d.csv")
                pred.to_csv(pred_path, index=False)

                # Evaluate
                stat = evaluate_predictions(pred)
                port = evaluate_portfolio_from_predictions(pred, horizon=h)

                row = {"datasource": ds, "horizon": f"{h}d", "model_type": model_type}
                row.update(stat)
                row.update(port)
                all_metrics.append(row)

                # Aggregate for datasource comparison
                if model_type == "timeseries_lgbm_walkforward":
                    ds_row = {
                        "datasource": ds, "horizon": f"{h}d",
                        "mean_daily_rankic": stat["mean_daily_rankic"],
                        "rankic_ir": stat["rankic_ir"],
                        "portfolio_sharpe": port["portfolio_sharpe"],
                        "portfolio_total_return": port["portfolio_total_return"],
                    }
                    datasource_metrics.append(ds_row)

                gc.collect()

    # Save results
    if all_metrics:
        metrics_df = pd.DataFrame(all_metrics).sort_values(["datasource", "horizon", "model_type"]).reset_index(drop=True)
        metrics_path = os.path.join(OUTPUT_DIR, "horizon_model_metrics.csv")
        metrics_df.to_csv(metrics_path, index=False)
        log(f"\nSaved: {metrics_path}")

    if datasource_metrics:
        ds_metrics_df = pd.DataFrame(datasource_metrics).sort_values(["datasource", "horizon"]).reset_index(drop=True)
        ds_metrics_path = os.path.join(OUTPUT_DIR, "horizon_datasource_metrics.csv")
        ds_metrics_df.to_csv(ds_metrics_path, index=False)
        log(f"Saved: {ds_metrics_path}")
        plot_datasource_comparison(ds_metrics_df, "datasource_comparison.png")

    # Plot heatmaps
    if all_metrics:
        main_metrics = metrics_df[~metrics_df["model_type"].str.startswith("secondary")]
        if not main_metrics.empty and len(main_metrics) > 0:
            plot_heatmap(main_metrics, "mean_daily_rankic", "Mean Daily RankIC by Horizon x Model", "heatmap_rankic.png")
            plot_heatmap(main_metrics, "portfolio_sharpe", "Portfolio Sharpe by Horizon x Model", "heatmap_portfolio_sharpe.png")

    total_time = time.time() - start_time
    log(f"\n{'=' * 60}")
    log(f"Done! Total time: {total_time/60:.1f} minutes")
    log("=" * 60)


if __name__ == "__main__":
    main()
