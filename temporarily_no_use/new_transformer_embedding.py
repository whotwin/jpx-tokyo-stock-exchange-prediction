"""
JPX Stock Prediction - Stock Embedding Transformer Model

=======================================================================
修改要点 (Key Modifications from iTransformer):
=======================================================================
1. 显存优化: 注意力从 O(N²) 降到 O(T²)，支持 2000+ 股票训练
   - iTransformer: 2000股票×41特征=82,000变元，注意力矩阵 82,000²≈67亿元素
   - StockEmbeddingTransformer: 注意力矩阵仅 20²=400元素 (seq_len=20)

2. 输入格式改变:
   - 旧: (batch, seq_len, stocks × features) = (batch, 20, 82000)
   - 新: (batch, seq_len, features) = (batch, 20, 41)，每行是一只股票

3. 股票嵌入: 添加 nn.Embedding(num_stocks, embed_dim)
   - 将 Stock ID 映射为向量，拼接到特征维度
   - 让模型能区分不同股票的特性

4. 注意力机制: 仅在时间步 (seq_len) 上计算
   - 使用 nn.TransformerEncoder, batch_first=True
   - 复杂度与股票数量无关

5. 输出: 每只股票独立预测 30 日收益率

训练策略: Expanding Window (保持不变)
- Round 1: Train 2017 → Validate 2018
- Round 2: Train 2017-2018 → Validate 2019
- Round 3: Train 2017-2019 → Validate 2020
- Final: Train 2017-2020 → Pred 2021

评估指标: 复用原有实现
- RMSE, Spearman, Hit Ratio
- Portfolio Sharpe, Kaggle Official Sharpe
=======================================================================
"""

import os
import warnings
import gc
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

OUTPUT_DIR = "output_stock_embedding_transformer"
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# Configuration
TEST_YEAR = 2021
TARGET_HORIZON = 30
TOP_K = 200
BOTTOM_K = 200
SEQ_LENGTH = 20  # 20-day lookback window

# Trading costs - same as original
TRADING_COST_RATE = 0.0004
SLIPPAGE_RATE = 0.0002

# Stock Embedding Transformer hyperparameters
# 显存优化: batch_size 可以大幅提高 (不再受股票数限制)
EMBED_DIM = 64               # Stock embedding 维度
TRANSFORMER_DIM = 64        # Transformer 隐层维度
TRANSFORMER_HEADS = 4       # 注意力头数
TRANSFORMER_LAYERS = 2      # Transformer 层数
DROPOUT = 0.1
TRANSFORMER_EPOCHS = 10
TRANSFORMER_BATCH_SIZE = 256  # 可大幅提高! 相比 iTransformer 的 4
TRANSFORMER_LEARNING_RATE = 0.001

# 股票过滤
MIN_DATA_POINTS = 300  # Minimum data points per stock


def log(msg):
    print(f"[INFO] {msg}")


def to_num(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# ============== 数据加载函数 (复用) ==============

def load_data_sources(data_dir="train_files"):
    """Load data sources - same as train.py."""
    sources = {}

    stock_prices = pd.read_csv(os.path.join(data_dir, "stock_prices.csv"))
    stock_prices = to_num(stock_prices, ["Open", "High", "Low", "Close", "Volume", "AdjustmentFactor", "ExpectedDividend", "Target", "SupervisionFlag"])
    stock_prices["Date"] = pd.to_datetime(stock_prices["Date"])
    sources["stock_prices"] = stock_prices

    if os.path.exists(os.path.join(data_dir, "stock_list.csv")):
        stock_list = pd.read_csv(os.path.join(data_dir, "stock_list.csv"),
                                 usecols=["SecuritiesCode", "MarketCapitalization", "33SectorName", "NewMarketSegment"])
        sources["stock_list"] = stock_list
    elif os.path.exists("stock_list.csv"):
        stock_list = pd.read_csv("stock_list.csv",
                                 usecols=["SecuritiesCode", "MarketCapitalization", "33SectorName", "NewMarketSegment"])
        sources["stock_list"] = stock_list

    if os.path.exists(os.path.join(data_dir, "secondary_stock_prices.csv")):
        secondary = pd.read_csv(os.path.join(data_dir, "secondary_stock_prices.csv"))
        secondary = to_num(secondary, ["Open", "High", "Low", "Close", "Volume", "AdjustmentFactor"])
        secondary["Date"] = pd.to_datetime(secondary["Date"])
        sources["secondary_stock_prices"] = secondary

    if os.path.exists(os.path.join(data_dir, "options.csv")):
        opts = pd.read_csv(os.path.join(data_dir, "options.csv"))
        opts = to_num(opts, ["ImpliedVolatility", "TradingVolume", "OpenInterest", "SettlementPrice", "BaseVolatility"])
        opts["Date"] = pd.to_datetime(opts["Date"])
        sources["options"] = opts

    if os.path.exists(os.path.join(data_dir, "trades.csv")):
        trades = pd.read_csv(os.path.join(data_dir, "trades.csv"))
        trades = to_num(trades, ["Individual", "Foreigners", "SecuritiesCos", "InvestmentTrusts", "InsuranceCos", "CityBKs", "RegionalBKs", "TrustBanks"])
        trades["Date"] = pd.to_datetime(trades["Date"])
        sources["trades"] = trades

    if os.path.exists(os.path.join(data_dir, "financials.csv")):
        financials = pd.read_csv(os.path.join(data_dir, "financials.csv"), low_memory=False)
        financials = to_num(financials, ["NetSales", "OperatingProfit", "OrdinaryProfit", "Profit", "TotalAssets", "Equity", "EquityToAssetRatio", "EarningsPerShare", "ForecastedEarningsPerShare"])
        financials["Date"] = pd.to_datetime(financials["Date"])
        sources["financials"] = financials

    log(f"Loaded data sources: {list(sources.keys())}")
    return sources


def stock_features(px, stock_list=None):
    """Extract stock-level features from price data."""
    px = px.sort_values(["SecuritiesCode", "Date"]).reset_index(drop=True)
    g = px.groupby("SecuritiesCode", sort=False)
    c = px["Close"].replace(0, np.nan)

    # Returns
    for w in [1, 2, 3, 5, 10, 20]:
        px[f"stk_ret_{w}"] = g["Close"].pct_change(w)

    # Log returns
    px["stk_logret_1"] = np.log(c).groupby(px["SecuritiesCode"]).diff(1)

    # Spread features
    px["stk_hl_spread"] = (px["High"] - px["Low"]) / c
    px["stk_oc_spread"] = (px["Close"] - px["Open"]) / px["Open"].replace(0, np.nan)

    # Volume change
    px["stk_volume_chg_1"] = g["Volume"].pct_change(1)

    # Volatility (rolling std of log returns)
    for w in [5, 10, 20]:
        px[f"stk_vol_{w}"] = g["stk_logret_1"].transform(lambda x: x.rolling(w, min_periods=w).std())

    # Rolling mean returns
    for w in [5, 10, 20]:
        px[f"stk_ret_mean_{w}"] = g["stk_ret_1"].transform(lambda x: x.rolling(w, min_periods=w).mean())

    # Moving averages
    for w in [5, 10, 20]:
        ma = g["Close"].transform(lambda x: x.rolling(w, min_periods=w).mean())
        px[f"stk_close_to_ma_{w}"] = px["Close"] / ma - 1

    # Volume to MA
    for w in [5, 10, 20]:
        vma = g["Volume"].transform(lambda x: x.rolling(w, min_periods=w).mean())
        px[f"stk_volume_to_ma_{w}"] = px["Volume"] / vma - 1

    # Skewness
    px["stk_skew_20"] = g["stk_logret_1"].transform(lambda x: x.rolling(20, min_periods=20).skew())

    # Day of week and month
    px["stk_dayofweek"] = px["Date"].dt.dayofweek
    px["stk_month"] = px["Date"].dt.month

    # Expected dividend
    px["stk_expected_dividend"] = px["ExpectedDividend"].fillna(0)

    # Supervision flag
    px["stk_supervision_flag"] = px["SupervisionFlag"].astype(str).str.lower().eq("true").astype(int)

    # Add market cap and sector features from stock_list
    if stock_list is not None:
        stock_list = stock_list.copy()
        px = px.merge(
            stock_list[["SecuritiesCode", "MarketCapitalization", "33SectorName", "NewMarketSegment"]],
            on="SecuritiesCode",
            how="left"
        )
        px["stk_mcap"] = np.log(px["MarketCapitalization"].fillna(1e8) / 1e8 + 1)
        px["stk_sector"] = pd.Categorical(px["33SectorName"]).codes
        px["stk_market_segment"] = pd.Categorical(px["NewMarketSegment"]).codes
        px = px.drop(columns=["MarketCapitalization", "33SectorName", "NewMarketSegment"], errors="ignore")

    return px


def build_30d_labels(stock_prices):
    """Build 30-day forward return labels."""
    px = stock_prices[["Date", "SecuritiesCode", "Close"]].copy()
    px["Close"] = pd.to_numeric(px["Close"], errors="coerce")
    px = px.sort_values(["SecuritiesCode", "Date"]).reset_index(drop=True)

    px["target_30d"] = px.groupby("SecuritiesCode")["Close"].shift(-30) / px["Close"] - 1.0

    return px[["Date", "SecuritiesCode", "target_30d"]]


def options_features(opts):
    """Extract options features - implied volatility."""
    if opts is None or opts.empty:
        return pd.DataFrame()

    opts = opts.sort_values("Date").reset_index(drop=True)

    if "ImpliedVolatility" in opts.columns:
        iv = opts.groupby("Date")["ImpliedVolatility"].mean().reset_index()
        iv.columns = ["Date", "iv_avg"]
        return iv

    return pd.DataFrame()


def trades_features(trades_df):
    """Extract trades features - investor trading data."""
    if trades_df is None or trades_df.empty:
        return pd.DataFrame()

    trades_df = trades_df.sort_values("Date").reset_index(drop=True)

    investor_cols = ["Individual", "Foreigners", "SecuritiesCos", "InvestmentTrusts"]
    available_cols = [c for c in investor_cols if c in trades_df.columns]

    if not available_cols:
        return pd.DataFrame()

    result = trades_df.groupby("Date")[available_cols].mean().reset_index()
    result.columns = ["Date"] + [f"trd_{c.lower()}" for c in available_cols]

    return result


def financials_features(fn):
    """Extract financials features."""
    if fn is None or fn.empty:
        return pd.DataFrame()

    fn = fn.sort_values(["SecuritiesCode", "Date"]).reset_index(drop=True)

    num_cols = ["NetSales", "OperatingProfit", "OrdinaryProfit", "Profit", "TotalAssets", "Equity"]
    available_cols = [c for c in num_cols if c in fn.columns]

    if not available_cols:
        return pd.DataFrame()

    fn[available_cols] = fn.groupby("SecuritiesCode")[available_cols].ffill()

    result = fn.groupby("SecuritiesCode", as_index=False)[available_cols].last()
    result.columns = ["SecuritiesCode"] + [f"fin_{c.lower()}" for c in available_cols]

    return result


def load_all_data():
    """Load all data sources."""
    log("Loading all data sources...")
    sources = load_data_sources("train_files")

    log("Building full feature table...")
    prices = sources["stock_prices"].copy()
    prices = prices[(prices["Date"] >= pd.to_datetime("2017-01-04")) &
                    (prices["Date"] <= pd.to_datetime("2021-12-03"))]

    df = prices[["Date", "SecuritiesCode", "Close", "Volume", "High", "Low", "Open", "ExpectedDividend", "SupervisionFlag"]].copy()

    stock_list = sources.get("stock_list", None)
    px_with_features = stock_features(prices, stock_list=stock_list)

    stock_cols = [c for c in px_with_features.columns if c.startswith("stk_")]
    df = df.merge(px_with_features[["Date", "SecuritiesCode"] + stock_cols],
                  on=["Date", "SecuritiesCode"], how="left")

    # Add options features (implied volatility)
    if "options" in sources:
        opt_feat = options_features(sources["options"])
        if not opt_feat.empty:
            df = df.merge(opt_feat, on="Date", how="left")
            opt_cols = [c for c in df.columns if c.startswith("iv_")]
            log(f"Added options features: {opt_cols}")

    # Add trades features (investor trading data)
    if "trades" in sources:
        trd_feat = trades_features(sources["trades"])
        if not trd_feat.empty:
            df = df.merge(trd_feat, on="Date", how="left")
            trd_cols = [c for c in df.columns if c.startswith("trd_")]
            log(f"Added trades features: {trd_cols}")

    # Add financials features
    if "financials" in sources:
        fin_feat = financials_features(sources["financials"])
        if not fin_feat.empty:
            df = df.merge(fin_feat, on="SecuritiesCode", how="left")
            fin_cols = [c for c in df.columns if c.startswith("fin_")]
            log(f"Added financials features: {fin_cols}")

    # Shift features by 1 to avoid leakage
    all_feature_cols = [c for c in df.columns if c not in ["Date", "SecuritiesCode", "Close", "Volume", "High", "Low", "Open", "ExpectedDividend", "SupervisionFlag", "target_30d"]]
    for col in all_feature_cols:
        if col in df.columns:
            df[col] = df.groupby("SecuritiesCode", sort=False)[col].shift(1)

    df = df.dropna(subset=["Date", "SecuritiesCode", "Close"])
    df[all_feature_cols] = df[all_feature_cols].fillna(0)

    # Build 30-day forward return labels
    labels = build_30d_labels(sources["stock_prices"])
    df = df.merge(labels, on=["Date", "SecuritiesCode"], how="left")
    df = df.sort_values(["Date", "SecuritiesCode"]).reset_index(drop=True)

    log(f"Loaded: {len(df)} rows, {len(all_feature_cols)} features")

    return df, all_feature_cols


# ============== Stock Embedding Transformer 模型 ==============

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for Transformer."""
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Add batch dimension
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class StockEmbeddingTransformer(nn.Module):
    """
    Stock Embedding Transformer - 显存优化版

    核心改进:
    1. 每只股票作为独立样本处理 (batch_size = 股票数 × 序列数)
    2. 添加 Stock Embedding 区分不同股票
    3. 注意力仅在时间步上计算，复杂度 O(T²) vs 原 O(N²)

    输入:
    - x: (batch, seq_len, num_features) - 每只股票的特征序列
    - stock_ids: (batch,) - 股票索引，用于 embedding 查找

    输出:
    - pred: (batch,) - 每只股票的 30 日收益预测
    """
    def __init__(self, num_stocks, num_features, embed_dim=64, d_model=64,
                 num_heads=4, num_layers=2, dropout=0.1, max_len=500):
        super().__init__()

        self.num_stocks = num_stocks
        self.num_features = num_features
        self.embed_dim = embed_dim
        self.d_model = d_model

        # Stock Embedding: 将股票ID映射为向量
        # 显存优化: 只增加 embed_dim 维度，不增加注意力复杂度
        self.stock_embedding = nn.Embedding(num_stocks, embed_dim)

        # 特征投影: 将 num_features 投影到 d_model - embed_dim
        # 然后拼接 stock embedding -> d_model
        feature_dim = d_model - embed_dim
        self.feature_proj = nn.Linear(num_features, feature_dim)

        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)

        # Transformer 编码器 - 仅在时间步上计算注意力
        # 显存优化: 复杂度 O(T²)，与股票数量无关
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 输出投影: d_model -> 1
        self.fc = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x, stock_ids):
        """
        前向传播

        Args:
            x: (batch, seq_len, num_features) - 特征序列
            stock_ids: (batch,) - 股票索引

        Returns:
            pred: (batch,) - 预测的 30 日收益
        """
        batch_size = x.size(0)

        # 1. 特征投影: (batch, seq_len, num_features) -> (batch, seq_len, d_model - embed_dim)
        x_feat = self.feature_proj(x)

        # 2. Stock Embedding: (batch,) -> (batch, 1, embed_dim)
        stock_emb = self.stock_embedding(stock_ids).unsqueeze(1)

        # 3. 拼接: (batch, seq_len, d_model - embed_dim) + (batch, 1, embed_dim)
        # 在特征维度拼接: (batch, seq_len, d_model)
        # 广播 stock_emb 到所有时间步
        stock_emb = stock_emb.expand(-1, x.size(1), -1)  # (batch, seq_len, embed_dim)
        x = torch.cat([x_feat, stock_emb], dim=-1)      # (batch, seq_len, d_model)

        # 4. 位置编码
        x = self.pos_encoder(x)

        # 5. Transformer 编码器 - 仅在时间步上计算注意力
        # 显存优化: 注意力矩阵大小 (batch, seq_len, seq_len) = (batch, 20, 20)
        # vs iTransformer: (batch, num_stocks*features, num_stocks*features)
        x = self.transformer(x)

        # 6. 取最后一个时间步的输出
        out = self.fc(x[:, -1, :])  # (batch, 1)

        return out.squeeze(-1)  # (batch,)


# ============== 数据加载函数 ==============

def create_transformer_sequences(df, feature_cols, stock_codes, seq_length=20, target_col="target_30d"):
    """
    创建 Stock Embedding Transformer 的训练数据

    关键修改:
    - 返回 X: (total_samples, seq_len, num_features) - 每行是一只股票的一个序列
    - 返回 y: (total_samples,) - 每只股票的 30 日收益
    - 返回 stock_ids: (total_samples,) - 股票索引用于 embedding 查找

    对比 iTransformer:
    - 旧: X = (samples, seq_len, stocks × features)
    - 新: X = (samples, seq_len, features)，每行是一只股票
    """
    log(f"Creating transformer sequences with seq_length={seq_length}, stocks={len(stock_codes)}...")

    # 创建股票代码到索引的映射
    stock_to_idx = {code: idx for idx, code in enumerate(stock_codes)}

    # 过滤只包含选定股票的数据
    df = df[df["SecuritiesCode"].isin(stock_codes)].copy()
    df = df.sort_values(["SecuritiesCode", "Date"]).reset_index(drop=True)

    # 定义使用的特征 (与 iTransformer 相同)
    use_features = [
        # 收益率特征（7个）
        "stk_ret_1", "stk_ret_2", "stk_ret_3", "stk_ret_5", "stk_ret_10", "stk_ret_20",
        "stk_logret_1",

        # 波动率特征（3个）
        "stk_vol_5", "stk_vol_10", "stk_vol_20",

        # 价差特征（2个）
        "stk_hl_spread", "stk_oc_spread",

        # 成交量特征（4个）
        "stk_volume_chg_1",
        "stk_volume_to_ma_5", "stk_volume_to_ma_10", "stk_volume_to_ma_20",

        # 滚动均值收益（3个）
        "stk_ret_mean_5", "stk_ret_mean_10", "stk_ret_mean_20",

        # 移动平均比率（3个）
        "stk_close_to_ma_5", "stk_close_to_ma_10", "stk_close_to_ma_20",

        # 分布特征（1个）
        "stk_skew_20",

        # 时间特征（2个）
        "stk_dayofweek", "stk_month",

        # 公司行为（2个）
        "stk_expected_dividend", "stk_supervision_flag",

        # 基本面特征（3个）
        "stk_mcap", "stk_sector", "stk_market_segment",

        # 期权特征（1个）
        "iv_avg",

        # 交易特征（4个）
        "trd_individual", "trd_foreigners", "trd_securitiescos", "trd_investmenttrusts",

        # 财务特征（6个）
        "fin_netsales", "fin_operatingprofit", "fin_ordinaryprofit",
        "fin_profit", "fin_totalassets", "fin_equity",
    ]

    # 过滤存在的特征
    use_features = [f for f in use_features if f in df.columns]
    num_features = len(use_features)
    if num_features == 0:
        use_features = [feature_cols[0]]
        num_features = 1

    log(f"Using {num_features} features: {use_features}")

    # 获取所有日期
    dates = sorted(df["Date"].unique())

    sequences = []
    targets = []
    stock_ids_list = []
    valid_dates = []

    # 为每只股票创建序列
    for code in tqdm(stock_codes, desc="Building sequences"):
        stock_data = df[df["SecuritiesCode"] == code].sort_values("Date").reset_index(drop=True)

        if len(stock_data) < seq_length + TARGET_HORIZON + 1:
            continue

        # 获取特征值和目标值
        feature_values = stock_data[use_features].values
        target_values = stock_data[target_col].values
        date_values = stock_data["Date"].values

        stock_idx = stock_to_idx[code]

        # 创建序列
        for i in range(seq_length, len(stock_data)):
            seq_features = feature_values[i - seq_length:i]
            target = target_values[i]
            target_date = date_values[i]

            # 跳过 NaN
            if np.isnan(seq_features).any() or np.isnan(target):
                continue

            sequences.append(seq_features)
            targets.append(target)
            stock_ids_list.append(stock_idx)
            valid_dates.append(target_date)

    X = np.array(sequences, dtype=np.float32)
    y = np.array(targets, dtype=np.float32)
    stock_ids_arr = np.array(stock_ids_list, dtype=np.int64)
    dates_arr = np.array(valid_dates)

    log(f"Created sequences: X={X.shape}, y={y.shape}, stock_ids={stock_ids_arr.shape}")
    log(f"num_features={num_features}")

    return X, y, stock_ids_arr, dates_arr, num_features


# ============== 训练函数 ==============

def train_stock_transformer_model(model, train_loader, epochs=10, lr=0.001):
    """训练 Stock Embedding Transformer 模型"""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0

        for batch_X, batch_y, batch_stock_ids in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            batch_stock_ids = batch_stock_ids.to(device)

            # 跳过包含 NaN 目标的批次
            if torch.isnan(batch_y).any():
                continue

            optimizer.zero_grad()
            pred = model(batch_X, batch_stock_ids)

            # 跳过 NaN 预测
            if torch.isnan(pred).any():
                continue

            loss = criterion(pred, batch_y)

            if torch.isnan(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        if num_batches > 0 and (epoch + 1) % 2 == 0:
            log(f"  Epoch {epoch+1}/{epochs}, Loss: {total_loss/num_batches:.6f}")

    return model


def predict_stock_transformer(model, test_loader):
    """使用 Stock Embedding Transformer 模型预测"""
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch_data in test_loader:
            batch_X = batch_data[0].to(device)
            batch_stock_ids = batch_data[2].to(device)  # stock_ids 在第三个位置
            pred = model(batch_X, batch_stock_ids)
            predictions.extend(pred.cpu().numpy())
    return np.array(predictions)


# ============== 评估函数 (复用) ==============

def calc_spread_return_sharpe(df: pd.DataFrame, portfolio_size: int = 200, toprank_weight_ratio: float = 2) -> float:
    """Kaggle official evaluation function."""
    def _calc_spread_return_per_day(df, portfolio_size, toprank_weight_ratio):
        actual_size = min(len(df), portfolio_size)
        if actual_size < 10:
            return 0.0

        assert df['Rank'].min() == 0
        assert df['Rank'].max() == len(df['Rank']) - 1
        weights = np.linspace(start=toprank_weight_ratio, stop=1, num=actual_size)
        purchase = (df.sort_values(by='Rank')['Target'][:actual_size] * weights).sum() / weights.mean()
        short = (df.sort_values(by='Rank', ascending=False)['Target'][:actual_size] * weights).sum() / weights.mean()
        return purchase - short

    buf = df.groupby('Date').apply(_calc_spread_return_per_day, portfolio_size, toprank_weight_ratio)
    buf = buf[buf != 0]
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


# ============== 主训练函数 ==============

def predict_with_stock_transformer(df, feature_cols, stock_codes):
    """运行 Stock Embedding Transformer 预测，使用 expanding window 训练"""
    log(f"Running Stock Embedding Transformer with expanding window training...")
    log(f"Using {len(stock_codes)} stocks")

    df = df.copy()
    df["Year"] = df["Date"].dt.year

    # 创建序列数据
    X, y, stock_ids_arr, dates_arr, num_features = create_transformer_sequences(
        df, feature_cols, stock_codes,
        seq_length=SEQ_LENGTH, target_col="target_30d"
    )

    if len(X) == 0:
        log("No sequences created!")
        return pd.DataFrame()

    num_stocks = len(stock_codes)
    log(f"Created sequences: X={X.shape}, y={y.shape}")
    log(f"num_features={num_features}, num_stocks={num_stocks}")

    # 转换日期用于过滤
    dates_pd = pd.to_datetime(dates_arr)
    years = dates_pd.year

    pred_parts = []

    # ======== Round 1: Train 2017 → Validate 2018 ========
    log("\n" + "="*50)
    log("Round 1: Train 2017 → Validate 2018")
    log("="*50)

    train_mask = years == 2017
    test_mask = years == 2018

    X_train = X[train_mask]
    y_train = y[train_mask]
    stock_ids_train = stock_ids_arr[train_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]
    stock_ids_test = stock_ids_arr[test_mask]
    dates_test = dates_arr[test_mask]

    log(f"  Train: {len(X_train):,}, Test: {len(X_test):,}")

    if len(X_train) > 0 and len(X_test) > 0:
        # 特征归一化: 仅使用训练集统计量
        x_mean = np.nanmean(X_train, axis=(0, 1), keepdims=True)
        x_std = np.nanstd(X_train, axis=(0, 1), keepdims=True)
        x_std[x_std < 1e-8] = 1.0

        # 目标归一化: 全局归一化
        y_mean = np.nanmean(y_train)
        y_std = np.nanstd(y_train)
        if y_std < 1e-8:
            y_std = 1.0

        # 归一化
        X_train_norm = (X_train - x_mean) / x_std
        y_train_norm = (y_train - y_mean) / y_std
        X_test_norm = (X_test - x_mean) / x_std

        # 裁剪极端值
        X_train_norm = np.clip(X_train_norm, -10, 10)
        X_test_norm = np.clip(X_test_norm, -10, 10)

        log(f"  Feature stats: mean={np.nanmean(x_mean):.6f}, std={np.nanmean(x_std):.6f}")
        log(f"  Target stats: mean={y_mean:.4f}, std={y_std:.4f}")

        # 创建 DataLoader (包含 stock_ids)
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train_norm),
            torch.FloatTensor(y_train_norm),
            torch.LongTensor(stock_ids_train)
        )
        train_loader = DataLoader(train_dataset, batch_size=TRANSFORMER_BATCH_SIZE, shuffle=True, num_workers=0)

        # 创建模型
        model = StockEmbeddingTransformer(
            num_stocks=num_stocks,
            num_features=num_features,
            embed_dim=EMBED_DIM,
            d_model=TRANSFORMER_DIM,
            num_heads=TRANSFORMER_HEADS,
            num_layers=TRANSFORMER_LAYERS,
            dropout=DROPOUT,
            max_len=SEQ_LENGTH
        )

        # 训练
        model = train_stock_transformer_model(model, train_loader, epochs=TRANSFORMER_EPOCHS, lr=TRANSFORMER_LEARNING_RATE)

        # 预测
        test_dataset = TensorDataset(
            torch.FloatTensor(X_test_norm),
            torch.zeros(len(X_test_norm)),  # dummy y
            torch.LongTensor(stock_ids_test)
        )
        test_loader = DataLoader(test_dataset, batch_size=TRANSFORMER_BATCH_SIZE, shuffle=False, num_workers=0)
        pred_norm = predict_stock_transformer(model, test_loader)

        # 反归一化
        pred = pred_norm * y_std + y_mean

        # 构建结果
        for i, date in enumerate(dates_test):
            code = stock_codes[stock_ids_test[i]]
            if not np.isnan(y_test[i]):
                pred_parts.append({
                    "Date": date,
                    "SecuritiesCode": code,
                    "y_true": y_test[i],
                    "pred": pred[i],
                    "train_year": 2017
                })

        del model
        gc.collect()

    # ======== Round 2: Train 2017-2018 → Validate 2019 ========
    log("\n" + "="*50)
    log("Round 2: Train 2017-2018 → Validate 2019")
    log("="*50)

    train_mask = (years == 2017) | (years == 2018)
    test_mask = years == 2019

    X_train = X[train_mask]
    y_train = y[train_mask]
    stock_ids_train = stock_ids_arr[train_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]
    stock_ids_test = stock_ids_arr[test_mask]
    dates_test = dates_arr[test_mask]

    log(f"  Train: {len(X_train):,}, Test: {len(X_test):,}")

    if len(X_train) > 0 and len(X_test) > 0:
        x_mean = np.nanmean(X_train, axis=(0, 1), keepdims=True)
        x_std = np.nanstd(X_train, axis=(0, 1), keepdims=True)
        x_std[x_std < 1e-8] = 1.0

        y_mean = np.nanmean(y_train)
        y_std = np.nanstd(y_train)
        if y_std < 1e-8:
            y_std = 1.0

        X_train_norm = (X_train - x_mean) / x_std
        y_train_norm = (y_train - y_mean) / y_std
        X_test_norm = (X_test - x_mean) / x_std
        X_train_norm = np.clip(X_train_norm, -10, 10)
        X_test_norm = np.clip(X_test_norm, -10, 10)

        train_dataset = TensorDataset(
            torch.FloatTensor(X_train_norm),
            torch.FloatTensor(y_train_norm),
            torch.LongTensor(stock_ids_train)
        )
        train_loader = DataLoader(train_dataset, batch_size=TRANSFORMER_BATCH_SIZE, shuffle=True, num_workers=0)

        model = StockEmbeddingTransformer(
            num_stocks=num_stocks,
            num_features=num_features,
            embed_dim=EMBED_DIM,
            d_model=TRANSFORMER_DIM,
            num_heads=TRANSFORMER_HEADS,
            num_layers=TRANSFORMER_LAYERS,
            dropout=DROPOUT,
            max_len=SEQ_LENGTH
        )
        model = train_stock_transformer_model(model, train_loader, epochs=TRANSFORMER_EPOCHS, lr=TRANSFORMER_LEARNING_RATE)

        test_dataset = TensorDataset(
            torch.FloatTensor(X_test_norm),
            torch.zeros(len(X_test_norm)),
            torch.LongTensor(stock_ids_test)
        )
        test_loader = DataLoader(test_dataset, batch_size=TRANSFORMER_BATCH_SIZE, shuffle=False, num_workers=0)
        pred_norm = predict_stock_transformer(model, test_loader)

        pred = pred_norm * y_std + y_mean

        for i, date in enumerate(dates_test):
            code = stock_codes[stock_ids_test[i]]
            if not np.isnan(y_test[i]):
                pred_parts.append({
                    "Date": date,
                    "SecuritiesCode": code,
                    "y_true": y_test[i],
                    "pred": pred[i],
                    "train_year": 2018
                })

        del model
        gc.collect()

    # ======== Round 3: Train 2017-2019 → Validate 2020 ========
    log("\n" + "="*50)
    log("Round 3: Train 2017-2019 → Validate 2020")
    log("="*50)

    train_mask = (years == 2017) | (years == 2018) | (years == 2019)
    test_mask = years == 2020

    X_train = X[train_mask]
    y_train = y[train_mask]
    stock_ids_train = stock_ids_arr[train_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]
    stock_ids_test = stock_ids_arr[test_mask]
    dates_test = dates_arr[test_mask]

    log(f"  Train: {len(X_train):,}, Test: {len(X_test):,}")

    if len(X_train) > 0 and len(X_test) > 0:
        x_mean = np.nanmean(X_train, axis=(0, 1), keepdims=True)
        x_std = np.nanstd(X_train, axis=(0, 1), keepdims=True)
        x_std[x_std < 1e-8] = 1.0

        y_mean = np.nanmean(y_train)
        y_std = np.nanstd(y_train)
        if y_std < 1e-8:
            y_std = 1.0

        X_train_norm = (X_train - x_mean) / x_std
        y_train_norm = (y_train - y_mean) / y_std
        X_test_norm = (X_test - x_mean) / x_std
        X_train_norm = np.clip(X_train_norm, -10, 10)
        X_test_norm = np.clip(X_test_norm, -10, 10)

        train_dataset = TensorDataset(
            torch.FloatTensor(X_train_norm),
            torch.FloatTensor(y_train_norm),
            torch.LongTensor(stock_ids_train)
        )
        train_loader = DataLoader(train_dataset, batch_size=TRANSFORMER_BATCH_SIZE, shuffle=True, num_workers=0)

        model = StockEmbeddingTransformer(
            num_stocks=num_stocks,
            num_features=num_features,
            embed_dim=EMBED_DIM,
            d_model=TRANSFORMER_DIM,
            num_heads=TRANSFORMER_HEADS,
            num_layers=TRANSFORMER_LAYERS,
            dropout=DROPOUT,
            max_len=SEQ_LENGTH
        )
        model = train_stock_transformer_model(model, train_loader, epochs=TRANSFORMER_EPOCHS, lr=TRANSFORMER_LEARNING_RATE)

        test_dataset = TensorDataset(
            torch.FloatTensor(X_test_norm),
            torch.zeros(len(X_test_norm)),
            torch.LongTensor(stock_ids_test)
        )
        test_loader = DataLoader(test_dataset, batch_size=TRANSFORMER_BATCH_SIZE, shuffle=False, num_workers=0)
        pred_norm = predict_stock_transformer(model, test_loader)

        pred = pred_norm * y_std + y_mean

        for i, date in enumerate(dates_test):
            code = stock_codes[stock_ids_test[i]]
            if not np.isnan(y_test[i]):
                pred_parts.append({
                    "Date": date,
                    "SecuritiesCode": code,
                    "y_true": y_test[i],
                    "pred": pred[i],
                    "train_year": 2019
                })

        del model
        gc.collect()

    # ======== Final: Train 2017-2020 → Predict 2021 ========
    log("\n" + "="*50)
    log("Final: Train 2017-2020 → Predict 2021")
    log("="*50)

    train_mask = (years == 2017) | (years == 2018) | (years == 2019) | (years == 2020)
    test_mask = years == 2021

    X_train = X[train_mask]
    y_train = y[train_mask]
    stock_ids_train = stock_ids_arr[train_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]
    stock_ids_test = stock_ids_arr[test_mask]
    dates_test = dates_arr[test_mask]

    log(f"  Train: {len(X_train):,}, Test: {len(X_test):,}")

    if len(X_train) > 0 and len(X_test) > 0:
        x_mean = np.nanmean(X_train, axis=(0, 1), keepdims=True)
        x_std = np.nanstd(X_train, axis=(0, 1), keepdims=True)
        x_std[x_std < 1e-8] = 1.0

        y_mean = np.nanmean(y_train)
        y_std = np.nanstd(y_train)
        if y_std < 1e-8:
            y_std = 1.0

        X_train_norm = (X_train - x_mean) / x_std
        y_train_norm = (y_train - y_mean) / y_std
        X_test_norm = (X_test - x_mean) / x_std
        X_train_norm = np.clip(X_train_norm, -10, 10)
        X_test_norm = np.clip(X_test_norm, -10, 10)

        train_dataset = TensorDataset(
            torch.FloatTensor(X_train_norm),
            torch.FloatTensor(y_train_norm),
            torch.LongTensor(stock_ids_train)
        )
        train_loader = DataLoader(train_dataset, batch_size=TRANSFORMER_BATCH_SIZE, shuffle=True, num_workers=0)

        model = StockEmbeddingTransformer(
            num_stocks=num_stocks,
            num_features=num_features,
            embed_dim=EMBED_DIM,
            d_model=TRANSFORMER_DIM,
            num_heads=TRANSFORMER_HEADS,
            num_layers=TRANSFORMER_LAYERS,
            dropout=DROPOUT,
            max_len=SEQ_LENGTH
        )
        model = train_stock_transformer_model(model, train_loader, epochs=TRANSFORMER_EPOCHS, lr=TRANSFORMER_LEARNING_RATE)

        test_dataset = TensorDataset(
            torch.FloatTensor(X_test_norm),
            torch.zeros(len(X_test_norm)),
            torch.LongTensor(stock_ids_test)
        )
        test_loader = DataLoader(test_dataset, batch_size=TRANSFORMER_BATCH_SIZE, shuffle=False, num_workers=0)
        pred_norm = predict_stock_transformer(model, test_loader)

        pred = pred_norm * y_std + y_mean

        for i, date in enumerate(dates_test):
            code = stock_codes[stock_ids_test[i]]
            if not np.isnan(y_test[i]):
                pred_parts.append({
                    "Date": date,
                    "SecuritiesCode": code,
                    "y_true": y_test[i],
                    "pred": pred[i],
                    "train_year": 2020
                })

        del model
        gc.collect()

    if pred_parts:
        out = pd.DataFrame(pred_parts).sort_values(["Date", "SecuritiesCode"]).reset_index(drop=True)
    else:
        out = pd.DataFrame()

    log(f"Total predictions: {len(out):,}")
    return out


def main():
    import time
    start_time = time.time()

    log("=" * 60)
    log("JPX 30-Day Horizon - Stock Embedding Transformer Model")
    log(f"Configuration: seq_length={SEQ_LENGTH}, horizon={TARGET_HORIZON}")
    log(f"Model: embed_dim={EMBED_DIM}, d_model={TRANSFORMER_DIM}, heads={TRANSFORMER_HEADS}, layers={TRANSFORMER_LAYERS}")
    log(f"Training: batch_size={TRANSFORMER_BATCH_SIZE}, epochs={TRANSFORMER_EPOCHS}")
    log("=" * 60)

    # 加载数据
    data, feature_cols = load_all_data()

    # 过滤有足够数据的股票
    log("Filtering stocks with sufficient data...")
    stock_counts = data.groupby("SecuritiesCode").size()
    valid_stocks = stock_counts[stock_counts >= MIN_DATA_POINTS].index.tolist()
    log(f"Stocks with >= {MIN_DATA_POINTS} data points: {len(valid_stocks)}")

    log(f"Using {len(valid_stocks)} stocks with multi features for Stock Embedding Transformer")

    # 运行预测
    pred = predict_with_stock_transformer(data, feature_cols, valid_stocks)

    if pred.empty:
        log("No predictions generated!")
        return

    # 保存预测
    pred_path = os.path.join(OUTPUT_DIR, "predictions.csv")
    pred.to_csv(pred_path, index=False)
    log(f"Saved: {pred_path}")

    # 评估
    port_metrics = evaluate_portfolio(pred)
    pred_metrics = evaluate_predictions(pred)
    kaggle_metrics = evaluate_portfolio_kaggle(pred)

    log("\n" + "=" * 40)
    log("PREDICTION METRICS")
    log("=" * 40)
    log(f"RMSE: {pred_metrics['rmse']:.6f}")
    log(f"Spearman: {pred_metrics['spearman']:.4f}")
    log(f"Hit Ratio: {pred_metrics['hit']:.2%}")

    log("\n" + "=" * 40)
    log("PORTFOLIO METRICS")
    log("=" * 40)
    log(f"Rebalance Days: {port_metrics['num_days']}")
    log(f"Total Spread: {port_metrics['spread']:.4%}")
    log(f"Sharpe: {port_metrics['sharpe']:.4f}")
    log(f"Hit Ratio: {port_metrics['hit_ratio']:.2%}")

    log("\n" + "=" * 40)
    log("KAGGLE OFFICIAL EVALUATION")
    log("=" * 40)
    log(f"Kaggle Sharpe: {kaggle_metrics['kaggle_sharpe']:.4f}")

    # 保存指标
    metrics = {**pred_metrics, **port_metrics, **kaggle_metrics}
    if 'daily_df' in port_metrics:
        del port_metrics['daily_df']
    metrics_df = pd.DataFrame([metrics])
    metrics_path = os.path.join(OUTPUT_DIR, "metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)

    total_time = time.time() - start_time
    log(f"\nDone! Time: {total_time/60:.1f} min")


if __name__ == "__main__":
    main()
