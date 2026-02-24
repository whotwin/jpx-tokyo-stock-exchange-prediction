"""
JPX Stock Prediction - Model Comparison
Compare LightGBM (cross-sectional) vs LSTM/Transformer (time-series)
Using PyTorch for deep learning models
"""

import os
import warnings
import gc
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

OUTPUT_DIR = "output_comparison"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Configuration
TEST_YEAR = 2021
TARGET_HORIZON = 30
TOP_K = 200
BOTTOM_K = 200
SEQ_LENGTH = 20

# LightGBM parameters
LGBM_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.02,
    "max_depth": 6,
    "num_leaves": 31,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}


def log(msg):
    print(f"[INFO] {msg}")


def to_num(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_data_sources(data_dir="train_files"):
    """Load data sources."""
    sources = {}

    stock_prices = pd.read_csv(os.path.join(data_dir, "stock_prices.csv"))
    stock_prices = to_num(stock_prices, ["Open", "High", "Low", "Close", "Volume", "AdjustmentFactor"])
    stock_prices["Date"] = pd.to_datetime(stock_prices["Date"])
    sources["stock_prices"] = stock_prices

    if os.path.exists(os.path.join(data_dir, "stock_list.csv")):
        stock_list = pd.read_csv(os.path.join(data_dir, "stock_list.csv"),
                                 usecols=["SecuritiesCode", "MarketCapitalization", "33SectorName"])
        sources["stock_list"] = stock_list

    return sources


def build_stock_features(px, stock_list=None):
    """Build cross-sectional features for LightGBM."""
    px = px.sort_values(["SecuritiesCode", "Date"]).reset_index(drop=True)
    g = px.groupby("SecuritiesCode", sort=False)
    c = px["Close"].replace(0, np.nan)

    for w in [1, 2, 3, 5, 10, 20]:
        px[f"ret_{w}"] = g["Close"].pct_change(w)

    px["logret_1"] = np.log(c).groupby(px["SecuritiesCode"]).diff(1)

    for w in [5, 10, 20]:
        px[f"vol_{w}"] = g["logret_1"].transform(lambda x: x.rolling(w, min_periods=w).std())

    for w in [5, 10, 20]:
        ma = g["Close"].transform(lambda x: x.rolling(w, min_periods=w).mean())
        px[f"close_to_ma_{w}"] = px["Close"] / ma - 1

    px["volume_chg"] = g["Volume"].pct_change(1)

    if stock_list is not None:
        px = px.merge(stock_list[["SecuritiesCode", "MarketCapitalization"]], on="SecuritiesCode", how="left")
        px["mcap"] = np.log(px["MarketCapitalization"].fillna(1e8) / 1e8 + 1)
        px = px.drop(columns=["MarketCapitalization"], errors="ignore")

    return px


def prepare_lgbm_data(sources):
    """Prepare data for LightGBM model."""
    prices = sources["stock_prices"].copy()
    prices = prices[(prices["Date"] >= "2017-01-04") & (prices["Date"] <= "2021-12-03")]

    stock_list = sources.get("stock_list", None)
    px = build_stock_features(prices, stock_list)

    feature_cols = [c for c in px.columns if c.startswith("ret_") or c.startswith("vol_")
                    or c.startswith("close_to_ma") or c == "volume_chg" or c == "mcap"]

    df = prices[["Date", "SecuritiesCode", "Close"]].copy()
    df = df.merge(px[["Date", "SecuritiesCode"] + feature_cols], on=["Date", "SecuritiesCode"], how="left")

    df = df.sort_values(["SecuritiesCode", "Date"]).reset_index(drop=True)
    df["target_30d"] = df.groupby("SecuritiesCode")["Close"].shift(-30) / df["Close"] - 1.0

    for col in feature_cols:
        df[col] = df.groupby("SecuritiesCode", sort=False)[col].shift(1)

    df = df.dropna(subset=["Date", "SecuritiesCode", "Close", "target_30d"])
    df[feature_cols] = df[feature_cols].fillna(0)

    return df, feature_cols


def prepare_sequence_data(sources, seq_length=20):
    """
    Prepare sequential data for LSTM/Transformer - FULL DATA version.
    Use efficient batching instead of sampling.
    """
    prices = sources["stock_prices"].copy()
    prices = prices[(prices["Date"] >= "2017-01-04") & (prices["Date"] <= "2021-12-03")]
    prices = prices.sort_values(["SecuritiesCode", "Date"]).reset_index(drop=True)

    # Calculate features
    prices["target_30d"] = prices.groupby("SecuritiesCode")["Close"].shift(-30) / prices["Close"] - 1.0
    prices["log_close"] = np.log(prices["Close"].replace(0, np.nan))
    prices["log_volume"] = np.log(prices["Volume"].replace(0, 1))

    # Normalize per stock - use expanding window to avoid look-ahead bias
    for col in ["log_close", "log_volume"]:
        # Use past 60 days for normalization (no future data)
        mean_vals = prices.groupby("SecuritiesCode")[col].transform(lambda x: x.shift(1).rolling(60, min_periods=20).mean())
        std_vals = prices.groupby("SecuritiesCode")[col].transform(lambda x: x.shift(1).rolling(60, min_periods=20).std())
        prices[f"{col}_norm"] = (prices[col] - mean_vals) / (std_vals + 1e-8)

    # Use all stocks but filter to those with enough data
    stock_counts = prices.groupby("SecuritiesCode")["Close"].count()
    valid_stocks = stock_counts[stock_counts > seq_length + TARGET_HORIZON + 30].index.tolist()
    prices = prices[prices["SecuritiesCode"].isin(valid_stocks)]

    # Create sequences efficiently using vectorized approach
    sequences = []
    targets = []
    dates = []
    codes = []

    for code in tqdm(valid_stocks, desc="Building sequences"):
        stock_data = prices[prices["SecuritiesCode"] == code].sort_values("Date")

        # Get normalized columns
        close_norm = stock_data["log_close_norm"].values
        vol_norm = stock_data["log_volume_norm"].values
        target_vals = stock_data["target_30d"].values
        date_vals = stock_data["Date"].values

        # Create sequences
        for i in range(seq_length, len(stock_data) - TARGET_HORIZON):
            seq_close = close_norm[i-seq_length:i]
            seq_vol = vol_norm[i-seq_length:i]
            target = target_vals[i + TARGET_HORIZON]

            # Skip if any NaN
            if np.isnan(seq_close).any() or np.isnan(seq_vol).any() or np.isnan(target):
                continue

            seq_features = np.stack([seq_close, seq_vol], axis=1)
            sequences.append(seq_features)
            targets.append(target)
            dates.append(date_vals[i])
            codes.append(code)

    X = np.array(sequences, dtype=np.float32)
    y = np.array(targets, dtype=np.float32)

    log(f"Sequence data: X={X.shape}, y={y.shape}")

    return X, y, dates, codes


# ============== PyTorch Models with Proper Positional Encoding ==============

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class LSTMModel(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out.squeeze()


class TransformerModel(nn.Module):
    def __init__(self, input_size=2, d_model=64, num_heads=4, num_layers=2, dropout=0.1, max_len=500):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        out = self.fc(x[:, -1, :])
        return out.squeeze()


def train_pytorch_model(model, train_loader, epochs=5, lr=0.001):
    """Train PyTorch model efficiently."""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            pred = model(batch_X)
            loss = criterion(pred, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1

        if (epoch + 1) % 1 == 0:
            log(f"  Epoch {epoch+1}/{epochs}, Loss: {total_loss/num_batches:.6f}")

    return model


def train_pytorch_model_with_validation(model, train_loader, val_loader, epochs=5, lr=0.001, model_name="Model"):
    """Train PyTorch model with validation for early stopping."""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    best_model_state = None
    patience = 2
    no_improve = 0

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        num_batches = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            pred = model(batch_X)
            loss = criterion(pred, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            num_batches += 1

        train_loss /= num_batches

        # Validation
        model.eval()
        val_loss = 0
        val_batches = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                pred = model(batch_X)
                loss = criterion(pred, batch_y)
                val_loss += loss.item()
                val_batches += 1

        val_loss /= val_batches

        log(f"  {model_name} Epoch {epoch+1}/{epochs}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                log(f"  Early stopping at epoch {epoch+1}")
                break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model


def predict_pytorch_model(model, test_loader):
    """Predict with PyTorch model."""
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch_X, in test_loader:
            batch_X = batch_X.to(device)
            pred = model(batch_X)
            predictions.extend(pred.cpu().numpy())
    return np.array(predictions)


# ============== Evaluation ==============

def evaluate_daily(df_eval):
    """Evaluate portfolio performance with daily rebalancing."""
    daily_results = []
    for date in df_eval["Date"].unique():
        day_data = df_eval[df_eval["Date"] == date].dropna()
        if len(day_data) < TOP_K + BOTTOM_K:
            continue

        sorted_data = day_data.sort_values("y_pred", ascending=False)
        top_k = sorted_data.head(TOP_K)
        bottom_k = sorted_data.tail(BOTTOM_K)

        spread = top_k["y_true"].mean() - bottom_k["y_true"].mean()
        daily_results.append({"date": date, "spread": spread})

    if not daily_results:
        return {}

    daily_df = pd.DataFrame(daily_results)
    return {
        "avg_spread": daily_df["spread"].mean(),
        "sharpe": daily_df["spread"].mean() / daily_df["spread"].std() * np.sqrt(252) if daily_df["spread"].std() > 0 else 0,
        "daily_df": daily_df,
    }


def run_comparison():
    """Main comparison function."""
    import time
    start_time = time.time()

    log("=" * 60)
    log("Model Comparison: LightGBM vs LSTM vs Transformer")
    log("=" * 60)

    # Load data
    log("Loading data...")
    sources = load_data_sources("train_files")

    results = []

    # ====== LightGBM ======
    log("\n" + "=" * 40)
    log("Training LightGBM...")
    lgbm_df, lgbm_features = prepare_lgbm_data(sources)

    lgbm_df["Year"] = lgbm_df["Date"].dt.year

    train_df = lgbm_df[(lgbm_df["Year"] < TEST_YEAR) & lgbm_df["target_30d"].notna()]
    test_df = lgbm_df[(lgbm_df["Year"] == TEST_YEAR) & lgbm_df["target_30d"].notna()]

    log(f"LightGBM - Train: {len(train_df):,}, Test: {len(test_df):,}")

    # Train LightGBM
    lgbm_model = LGBMRegressor(**LGBM_PARAMS)
    lgbm_model.fit(train_df[lgbm_features], train_df["target_30d"])

    # Predict
    lgbm_pred = lgbm_model.predict(test_df[lgbm_features])

    # Evaluate
    lgbm_eval_df = pd.DataFrame({
        "Date": test_df["Date"].values,
        "SecuritiesCode": test_df["SecuritiesCode"].values,
        "y_true": test_df["target_30d"].values,
        "y_pred": lgbm_pred,
    })

    lgbm_metrics = evaluate_daily(lgbm_eval_df)

    lgbm_result = {
        "model": "LightGBM",
        "rmse": np.sqrt(mean_squared_error(test_df["target_30d"], lgbm_pred)),
        "spearman": pd.Series(test_df["target_30d"]).corr(pd.Series(lgbm_pred), method="spearman"),
        "hit": np.mean(np.sign(test_df["target_30d"]) == np.sign(lgbm_pred)),
    }
    lgbm_result.update(lgbm_metrics)
    results.append(lgbm_result)

    log(f"LightGBM - Spearman: {lgbm_result['spearman']:.4f}, Sharpe: {lgbm_result.get('sharpe', 0):.4f}")

    # ====== LSTM ======
    log("\n" + "=" * 40)
    log("Training LSTM...")
    X_seq, y_seq, dates_seq, codes_seq = prepare_sequence_data(sources, seq_length=SEQ_LENGTH)

    dates_arr = np.array(dates_seq)

    # Time-series split: train on 2017-2019, validate on 2020, test on 2021
    train_mask = dates_arr < pd.Timestamp("2020-01-01")
    val_mask = (dates_arr >= pd.Timestamp("2020-01-01")) & (dates_arr < pd.Timestamp("2021-01-01"))
    test_mask = dates_arr >= pd.Timestamp("2021-01-01")

    # For efficiency, use a subset for training if data is too large
    X_train_full = X_seq[train_mask]
    y_train_full = y_seq[train_mask]
    dates_train_full = dates_arr[train_mask]

    # Sample training data if too large (keep temporal order)
    max_train_samples = 200000
    if len(X_train_full) > max_train_samples:
        step = len(X_train_full) // max_train_samples
        indices = np.arange(0, len(X_train_full), step)[:max_train_samples]
        X_train_seq = torch.FloatTensor(X_train_full[indices])
        y_train_seq = torch.FloatTensor(y_train_full[indices])
    else:
        X_train_seq = torch.FloatTensor(X_train_full)
        y_train_seq = torch.FloatTensor(y_train_full)

    X_val_seq = torch.FloatTensor(X_seq[val_mask])
    y_val_seq = y_seq[val_mask]
    X_test_seq = torch.FloatTensor(X_seq[test_mask])
    y_test_seq = y_seq[test_mask]
    dates_test_seq = dates_arr[test_mask]

    log(f"LSTM - Train: {len(X_train_seq):,}, Val: {len(X_val_seq):,}, Test: {len(X_test_seq):,}")

    # Create data loaders
    train_dataset = TensorDataset(X_train_seq, y_train_seq)
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True, num_workers=0)

    val_dataset = TensorDataset(X_val_seq, y_val_seq)
    val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False, num_workers=0)

    test_dataset = TensorDataset(X_test_seq)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False, num_workers=0)

    # Train LSTM with validation-based early stopping
    lstm_model = LSTMModel(input_size=2, hidden_size=32, num_layers=1, dropout=0.1)
    lstm_model = train_pytorch_model_with_validation(
        lstm_model, train_loader, val_loader, epochs=5, lr=0.005, model_name="LSTM")

    # Predict
    lstm_pred = predict_pytorch_model(lstm_model, test_loader)

    # Evaluate
    lstm_eval_df = pd.DataFrame({
        "Date": dates_test_seq,
        "y_true": y_test_seq,
        "y_pred": lstm_pred,
    })

    lstm_metrics = evaluate_daily(lstm_eval_df)

    lstm_result = {
        "model": "LSTM",
        "rmse": np.sqrt(mean_squared_error(y_test_seq, lstm_pred)),
        "spearman": pd.Series(y_test_seq).corr(pd.Series(lstm_pred), method="spearman"),
        "hit": np.mean(np.sign(y_test_seq) == np.sign(lstm_pred)),
    }
    lstm_result.update(lstm_metrics)
    results.append(lstm_result)

    log(f"LSTM - Spearman: {lstm_result['spearman']:.4f}, Sharpe: {lstm_result.get('sharpe', 0):.4f}")

    # ====== Transformer ======
    log("\n" + "=" * 40)
    log("Training Transformer...")

    # Train Transformer
    trans_model = TransformerModel(input_size=2, d_model=32, num_heads=2, num_layers=1, dropout=0.1)
    trans_model = train_pytorch_model(trans_model, train_loader, epochs=3, lr=0.003)

    # Predict
    trans_pred = predict_pytorch_model(trans_model, test_loader)

    # Evaluate
    trans_eval_df = pd.DataFrame({
        "Date": dates_test_seq,
        "y_true": y_test_seq,
        "y_pred": trans_pred,
    })

    trans_metrics = evaluate_daily(trans_eval_df)

    trans_result = {
        "model": "Transformer",
        "rmse": np.sqrt(mean_squared_error(y_test_seq, trans_pred)),
        "spearman": pd.Series(y_test_seq).corr(pd.Series(trans_pred), method="spearman"),
        "hit": np.mean(np.sign(y_test_seq) == np.sign(trans_pred)),
    }
    trans_result.update(trans_metrics)
    results.append(trans_result)

    log(f"Transformer - Spearman: {trans_result['spearman']:.4f}, Sharpe: {trans_result.get('sharpe', 0):.4f}")

    # ====== Summary ======
    log("\n" + "=" * 60)
    log("COMPARISON SUMMARY")
    log("=" * 60)

    for r in results:
        log(f"\n{r['model']}:")
        log(f"  RMSE: {r['rmse']:.6f}")
        log(f"  Spearman: {r['spearman']:.4f}")
        log(f"  Hit Ratio: {r['hit']:.2%}")
        if 'avg_spread' in r:
            log(f"  Avg Daily Spread: {r['avg_spread']:.6f}")
            log(f"  Sharpe (annualized): {r['sharpe']:.4f}")

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(OUTPUT_DIR, "model_comparison.csv"), index=False)
    log(f"\nResults saved to {OUTPUT_DIR}/model_comparison.csv")

    total_time = time.time() - start_time
    log(f"\nDone! Time: {total_time/60:.1f} min")

    return results


if __name__ == "__main__":
    run_comparison()
