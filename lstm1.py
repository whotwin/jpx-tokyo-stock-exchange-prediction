import os
import random
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# =============================
# 1. 基本设置
# =============================
SEED = 42
WINDOW_SIZE = 2
TARGET_COL = "Target"
ID_COL = "SecuritiesCode"
TIME_COL = "Quarter"

TRAIN_FILE = r"C:\Users\DELL\Desktop\SemB\6423\GP\train_dataset_clean.csv"
TEST_FILE = r"C:\Users\DELL\Desktop\SemB\6423\GP\test_dataset.csv"

SAVE_DIR = r"C:\Users\DELL\Desktop\SemB\6423\GP\lstm_final_outputs"
os.makedirs(SAVE_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

best_params = {
    "hidden1": 64,
    "hidden2": 32,
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 20
}

# =============================
# 2. 随机种子
# =============================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)

print("Using device:", DEVICE)
print("Best params:", best_params)

# =============================
# 3. 读取数据
# =============================
train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

train_df[TIME_COL] = pd.to_datetime(train_df[TIME_COL], errors="coerce")
test_df[TIME_COL] = pd.to_datetime(test_df[TIME_COL], errors="coerce")

train_df = train_df.sort_values([ID_COL, TIME_COL]).reset_index(drop=True)
test_df = test_df.sort_values([ID_COL, TIME_COL]).reset_index(drop=True)

feature_cols = [c for c in train_df.columns if c not in [ID_COL, TIME_COL, TARGET_COL]]
print("Number of features:", len(feature_cols))

# =============================
# 4. 缺失值处理
# =============================
def fill_missing(df):
    df = df.copy()
    df[feature_cols] = df.groupby(ID_COL)[feature_cols].transform(lambda x: x.ffill().bfill())
    df[feature_cols] = df[feature_cols].fillna(0)
    return df

train_df = fill_missing(train_df)
test_df = fill_missing(test_df)

# =============================
# 5. 构造序列
# =============================
def build_sequences(df, window_size):
    X_list = []
    y_list = []
    meta_list = []

    for stock, group in df.groupby(ID_COL):
        group = group.sort_values(TIME_COL).reset_index(drop=True)

        if len(group) <= window_size:
            continue

        for i in range(len(group) - window_size):
            if pd.isna(group.loc[i + window_size, TARGET_COL]):
                continue

            X_seq = group.loc[i:i + window_size - 1, feature_cols].values.astype(np.float32)
            y_val = np.float32(group.loc[i + window_size, TARGET_COL])
            y_time = group.loc[i + window_size, TIME_COL]

            X_list.append(X_seq)
            y_list.append(y_val)
            meta_list.append([stock, y_time, y_time.year])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32).reshape(-1, 1)
    meta = pd.DataFrame(meta_list, columns=[ID_COL, TIME_COL, "LabelYear"])
    return X, y, meta

X_all, y_all, meta_all = build_sequences(train_df, WINDOW_SIZE)

print("X_all shape:", X_all.shape)
print("y_all shape:", y_all.shape)

full_df = pd.concat([train_df, test_df], ignore_index=True)
full_df = full_df.sort_values([ID_COL, TIME_COL]).reset_index(drop=True)

X_full, y_full, meta_full = build_sequences(full_df, WINDOW_SIZE)

test_mask = (meta_full["LabelYear"] == 2021).values
X_test = X_full[test_mask]
y_test = y_full[test_mask]
meta_test = meta_full.loc[test_mask].reset_index(drop=True)

print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

# =============================
# 6. 标准化
# =============================
def fit_standardizer(X):
    X2 = X.reshape(-1, X.shape[-1])
    mean_ = X2.mean(axis=0)
    std_ = X2.std(axis=0)
    std_[std_ < 1e-8] = 1.0
    return mean_, std_

def transform_standardizer(X, mean_, std_):
    return (X - mean_) / std_

mean_, std_ = fit_standardizer(X_all)
X_all = transform_standardizer(X_all, mean_, std_)
X_test = transform_standardizer(X_test, mean_, std_)

# =============================
# 7. Dataset
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
# 8. 模型（加入 Dropout）
# =============================
class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden1, hidden2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden1,
            batch_first=True
        )
        self.fc1 = nn.Linear(hidden1, hidden2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden2, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

# =============================
# 9. 训练最终模型
# 损失函数改为 SmoothL1Loss
# =============================
train_dataset = SeqDataset(X_all, y_all)
train_loader = DataLoader(
    train_dataset,
    batch_size=best_params["batch_size"],
    shuffle=True
)

model = SimpleLSTM(
    input_size=len(feature_cols),
    hidden1=best_params["hidden1"],
    hidden2=best_params["hidden2"],
    dropout=0.2
).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=best_params["learning_rate"])
criterion = nn.SmoothL1Loss()

print("\nStart training...")
loss_history = []

for epoch in range(best_params["epochs"]):
    model.train()
    total_loss = 0.0

    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        optimizer.zero_grad()
        pred = model(X_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    loss_history.append(avg_loss)
    print(f"Epoch {epoch+1}/{best_params['epochs']}  Loss={avg_loss:.6f}")

print("Training finished.")

loss_df = pd.DataFrame({
    "Epoch": range(1, len(loss_history) + 1),
    "TrainLoss": loss_history
})
loss_df.to_csv(os.path.join(SAVE_DIR, "final_training_loss_optimized.csv"), index=False)

# =============================
# 10. 测试集预测
# =============================
test_dataset = SeqDataset(X_test, y_test)
test_loader = DataLoader(
    test_dataset,
    batch_size=best_params["batch_size"],
    shuffle=False
)

model.eval()
pred_list = []
true_list = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(DEVICE)
        pred = model(X_batch)

        pred_list.append(pred.cpu().numpy())
        true_list.append(y_batch.numpy())

preds = np.vstack(pred_list).reshape(-1)
trues = np.vstack(true_list).reshape(-1)

# =============================
# 11. 结果表
# 同时保留 signal_pos 和 signal_neg
# =============================
ranking_df = meta_test.copy()
ranking_df["y_pred"] = preds
ranking_df["y_true"] = trues
ranking_df["signal_pos"] = ranking_df["y_pred"]
ranking_df["signal_neg"] = -ranking_df["y_pred"]

ranking_df.to_csv(os.path.join(SAVE_DIR, "test_2021_ranking_detail_optimized.csv"), index=False)

print("\nPrediction finished.")
print(ranking_df.head())

# =============================
# 12. 评估函数
# =============================
def evaluate_signal(df, signal_col, save_prefix):
    df = df.copy()

    df["PredRank"] = df.groupby(TIME_COL)[signal_col].rank(method="first", ascending=False)
    df["TrueRank"] = df.groupby(TIME_COL)["y_true"].rank(method="first", ascending=False)

    rankic_rows = []
    top_list = []
    bottom_list = []

    print(f"\n========== Evaluating {signal_col} ==========")

    for quarter, group in df.groupby(TIME_COL):
        group = group.sort_values("PredRank").reset_index(drop=True)

        top200 = group.head(200).copy()
        bottom200 = group.tail(200).copy()

        top_list.append(top200)
        bottom_list.append(bottom200)

        if group[signal_col].nunique() > 1 and group["y_true"].nunique() > 1:
            rank_ic = spearmanr(group[signal_col], group["y_true"])[0]
        else:
            rank_ic = np.nan

        top_avg = top200["y_true"].mean()
        bottom_avg = bottom200["y_true"].mean()
        long_short = top_avg - bottom_avg

        rankic_rows.append({
            "Quarter": quarter,
            "NumStocks": len(group),
            "RankIC": rank_ic,
            "Top200AvgTrue": top_avg,
            "Bottom200AvgTrue": bottom_avg,
            "LongShortSpread": long_short
        })

        print(
            str(quarter.date()),
            "| RankIC =", round(rank_ic, 6),
            "| Top200Avg =", round(top_avg, 6),
            "| Bottom200Avg =", round(bottom_avg, 6),
            "| LongShort =", round(long_short, 6)
        )

    rankic_df = pd.DataFrame(rankic_rows)
    top200_df = pd.concat(top_list, ignore_index=True)
    bottom200_df = pd.concat(bottom_list, ignore_index=True)

    df.to_csv(os.path.join(SAVE_DIR, f"{save_prefix}_ranking.csv"), index=False)
    rankic_df.to_csv(os.path.join(SAVE_DIR, f"{save_prefix}_rankic.csv"), index=False)
    top200_df.to_csv(os.path.join(SAVE_DIR, f"{save_prefix}_top200.csv"), index=False)
    bottom200_df.to_csv(os.path.join(SAVE_DIR, f"{save_prefix}_bottom200.csv"), index=False)

    if df[signal_col].nunique() > 1 and df["y_true"].nunique() > 1:
        overall_rankic = spearmanr(df[signal_col], df["y_true"])[0]
    else:
        overall_rankic = np.nan

    avg_quarterly_rankic = rankic_df["RankIC"].mean()
    avg_long_short = rankic_df["LongShortSpread"].mean()

    print("\nSummary for", signal_col)
    print("Overall RankIC =", round(overall_rankic, 6))
    print("Average Quarterly RankIC =", round(avg_quarterly_rankic, 6))
    print("Average LongShort Spread =", round(avg_long_short, 6))

    return {
        "signal": signal_col,
        "overall_rankic": overall_rankic,
        "avg_quarterly_rankic": avg_quarterly_rankic,
        "avg_long_short": avg_long_short
    }

# =============================
# 13. 同时评估正向和取反
# =============================
result_pos = evaluate_signal(ranking_df, "signal_pos", "signal_pos")
result_neg = evaluate_signal(ranking_df, "signal_neg", "signal_neg")

summary_df = pd.DataFrame([result_pos, result_neg])
summary_df.to_csv(os.path.join(SAVE_DIR, "signal_compare_summary.csv"), index=False)

print("\n================ FINAL COMPARISON ================")
print(summary_df)
print("\nSaved in:", SAVE_DIR)
input("Press Enter to exit...")