import os
import random
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from itertools import product
from scipy.stats import spearmanr

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# =============================
# 1. 基本设置
# =============================
SEED = 42
WINDOW_SIZE = 3
TARGET_COL = "Target"
ID_COL = "SecuritiesCode"
TIME_COL = "Quarter"

TRAIN_FILE = r"cleaned_data\train_dataset_clean.csv"
TEST_FILE  = r"cleaned_data\test_dataset.csv"

SAVE_DIR = r"lstm_quant_outputs"
os.makedirs(SAVE_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================
# 2. 固定随机种子
# =============================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)


# =============================
# 3. Dataset
# =============================
class SeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# =============================
# 4. LSTM 模型
# =============================
class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden1, hidden2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden1,
            batch_first=True
        )
        self.fc1 = nn.Linear(hidden1, hidden2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden2, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out


# =============================
# 5. 计算 rankIC
# =============================
def calc_rankic(y_pred, y_true):
    if len(y_pred) < 2:
        return np.nan
    if np.std(y_pred) == 0 or np.std(y_true) == 0:
        return np.nan
    return spearmanr(y_pred, y_true)[0]


# =============================
# 6. 构造序列
# window=2: 前2个季度 -> 下1个季度
# LabelYear / LabelQuarter 都按 y 来定
# =============================
def build_sequences(df, feature_cols, target_col, id_col, time_col, window_size):
    X_list = []
    y_list = []
    meta_list = []

    for stock, group in df.groupby(id_col):
        group = group.sort_values(time_col).reset_index(drop=True)

        if len(group) <= window_size:
            continue

        for i in range(len(group) - window_size):
            X_seq = group.loc[i:i + window_size - 1, feature_cols].values.astype(np.float32)
            y_value = group.loc[i + window_size, target_col]
            y_time = group.loc[i + window_size, time_col]

            X_list.append(X_seq)
            y_list.append(y_value)
            meta_list.append([
                stock,
                y_time,
                y_time.year,
                f"{y_time.year}Q{y_time.quarter}"
            ])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32).reshape(-1, 1)
    meta = pd.DataFrame(meta_list, columns=[id_col, time_col, "LabelYear", "LabelQuarter"])

    return X, y, meta


# =============================
# 7. 单个 fold 的训练与验证
# 每个季度算一个 rankIC
# =============================
def train_one_fold(X_train, y_train, X_val, y_val, meta_val, params, input_size):
    train_dataset = SeqDataset(X_train, y_train)
    val_dataset = SeqDataset(X_val, y_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=params["batch_size"],
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=params["batch_size"],
        shuffle=False
    )

    model = SimpleLSTM(
        input_size=input_size,
        hidden1=params["hidden1"],
        hidden2=params["hidden2"]
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
    criterion = nn.MSELoss()

    # 训练
    for epoch in range(params["epochs"]):
        model.train()
        for Xb, yb in train_loader:
            Xb = Xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad()
            pred = model(Xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

    # 验证预测
    model.eval()
    pred_list = []

    with torch.no_grad():
        for Xb, yb in val_loader:
            Xb = Xb.to(DEVICE)
            pred = model(Xb).cpu().numpy().reshape(-1)
            pred_list.extend(pred)

    val_result = meta_val.copy().reset_index(drop=True)
    val_result["y_true"] = y_val.reshape(-1)
    val_result["y_pred"] = np.array(pred_list)

    quarter_result = []
    quarter_ic_list = []

    quarter_names = sorted(val_result["LabelQuarter"].unique())

    for q in quarter_names:
        temp = val_result[val_result["LabelQuarter"] == q]
        ic = calc_rankic(temp["y_pred"].values, temp["y_true"].values)
        quarter_ic_list.append(ic)

        quarter_result.append({
            "LabelQuarter": q,
            "n_stocks": len(temp),
            "rankIC": ic
        })

    fold_mean_ic = np.nanmean(quarter_ic_list)

    return fold_mean_ic, quarter_result


# =============================
# 8. 主程序
# =============================
def main():
    print("========== Start ==========")
    print("Device:", DEVICE)
    print("Reading train file...")
    train_df = pd.read_csv(TRAIN_FILE)
    print("Reading test file...")
    test_df = pd.read_csv(TEST_FILE)

    print("Converting Quarter to datetime...")
    train_df[TIME_COL] = pd.to_datetime(train_df[TIME_COL], errors="coerce")
    test_df[TIME_COL] = pd.to_datetime(test_df[TIME_COL], errors="coerce")

    train_df = train_df.sort_values([ID_COL, TIME_COL]).reset_index(drop=True)
    test_df = test_df.sort_values([ID_COL, TIME_COL]).reset_index(drop=True)

    feature_cols = [c for c in train_df.columns if c not in [ID_COL, TIME_COL, TARGET_COL]]

    print("Feature count:", len(feature_cols))
    print("Filling missing values...")

    train_df[feature_cols] = train_df.groupby(ID_COL)[feature_cols].transform(lambda x: x.ffill().bfill())
    train_df[feature_cols] = train_df[feature_cols].fillna(0)

    test_df[feature_cols] = test_df.groupby(ID_COL)[feature_cols].transform(lambda x: x.ffill().bfill())
    test_df[feature_cols] = test_df[feature_cols].fillna(0)

    print("Building sequences with WINDOW_SIZE =", WINDOW_SIZE)
    X_all, y_all, meta_all = build_sequences(
        train_df, feature_cols, TARGET_COL, ID_COL, TIME_COL, WINDOW_SIZE
    )

    print("Total training sequences:", len(X_all))
    print("Sequence counts by LabelYear:")
    print(meta_all["LabelYear"].value_counts().sort_index())
    print()

    # 8组参数
    param_grid = {
        "hidden1": [32, 64],
        "hidden2": [16, 32],
        "learning_rate": [0.001, 0.0005],
        "batch_size": [32],
        "epochs": [20]
    }

    param_combinations = list(product(
        param_grid["hidden1"],
        param_grid["hidden2"],
        param_grid["learning_rate"],
        param_grid["batch_size"],
        param_grid["epochs"]
    ))

    folds = [
        ([2017], 2018),
        ([2017, 2018], 2019),
        ([2017, 2018, 2019], 2020)
    ]

    results = []
    quarter_results = []

    print("========== Start Grid Search ==========")
    print("Total parameter combinations:", len(param_combinations))
    print()

    for combo_idx, combo in enumerate(param_combinations, start=1):
        params = {
            "hidden1": combo[0],
            "hidden2": combo[1],
            "learning_rate": combo[2],
            "batch_size": combo[3],
            "epochs": combo[4]
        }

        print(f"Running combo {combo_idx}/{len(param_combinations)}: {params}")

        fold_mean_list = []

        for fold_idx, (train_years, val_year) in enumerate(folds, start=1):
            print(f"  Fold {fold_idx}: train={train_years}, val={val_year}")

            train_mask = meta_all["LabelYear"].isin(train_years).values
            val_mask = (meta_all["LabelYear"] == val_year).values

            X_train = X_all[train_mask]
            y_train = y_all[train_mask]
            X_val = X_all[val_mask]
            y_val = y_all[val_mask]
            meta_val = meta_all[val_mask]

            print("    Train samples:", len(X_train))
            print("    Val samples:", len(X_val))
            print("    Val quarters:", sorted(meta_val["LabelQuarter"].unique()))

            if len(X_train) == 0 or len(X_val) == 0:
                print("    Skip this fold because train or val is empty.")
                fold_mean_list.append(np.nan)
                continue

            fold_mean_ic, quarter_result = train_one_fold(
                X_train, y_train, X_val, y_val, meta_val, params, len(feature_cols)
            )

            print("    Fold mean rankIC:", fold_mean_ic)

            fold_mean_list.append(fold_mean_ic)

            for item in quarter_result:
                quarter_results.append({
                    "combo_idx": combo_idx,
                    "fold_idx": fold_idx,
                    "train_years": str(train_years),
                    "val_year": val_year,
                    "hidden1": params["hidden1"],
                    "hidden2": params["hidden2"],
                    "learning_rate": params["learning_rate"],
                    "batch_size": params["batch_size"],
                    "epochs": params["epochs"],
                    "LabelQuarter": item["LabelQuarter"],
                    "n_stocks": item["n_stocks"],
                    "rankIC": item["rankIC"]
                })

        avg_rankic = np.nanmean(fold_mean_list)

        results.append({
            "combo_idx": combo_idx,
            "hidden1": params["hidden1"],
            "hidden2": params["hidden2"],
            "learning_rate": params["learning_rate"],
            "batch_size": params["batch_size"],
            "epochs": params["epochs"],
            "fold1_rankic": fold_mean_list[0] if len(fold_mean_list) > 0 else np.nan,
            "fold2_rankic": fold_mean_list[1] if len(fold_mean_list) > 1 else np.nan,
            "fold3_rankic": fold_mean_list[2] if len(fold_mean_list) > 2 else np.nan,
            "avg_rankic": avg_rankic
        })

        print("  Combo avg rankIC:", avg_rankic)
        print()

    print("========== Saving Results ==========")

    results_df = pd.DataFrame(results).sort_values("avg_rankic", ascending=False).reset_index(drop=True)
    quarter_results_df = pd.DataFrame(quarter_results)

    result_file = os.path.join(SAVE_DIR, "gridsearch_results.csv")
    quarter_file = os.path.join(SAVE_DIR, "quarter_rankic_results.csv")

    results_df.to_csv(result_file, index=False)
    quarter_results_df.to_csv(quarter_file, index=False)

    print("Parameter combinations total:", len(results_df))
    print("Quarter-level rankIC total:", len(quarter_results_df))
    print("Expected quarter-level rankIC total: 96 (if every val year has 4 quarters)")
    print()

    print("Top results:")
    print(results_df.head())
    print()

    print("Quarter-level rankIC sample:")
    print(quarter_results_df.head(20))
    print()

    if len(results_df) > 0:
        print("Best Params:")
        print(results_df.iloc[0].to_dict())

    print()
    print("Saved files:")
    print(result_file)
    print(quarter_file)
    print("========== Finished ==========")


if __name__ == "__main__":
    main()
    input("Press Enter to exit...")