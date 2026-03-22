# LSTM Model Configuration

import os
import torch

# Logging function
def log(msg):
    print(f"[INFO] {msg}")

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

OUTPUT_DIR = "output_lstm"
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# Feature columns for LSTM input - aligned with itransformer_model.py (41 features)
USE_FEATURES = [
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

# Legacy OHLCV features (no longer used)
RAW_FEATURES = ["Open", "High", "Low", "Close", "Volume"]

# Configuration
TEST_YEAR = 2021
ROLL_TRAIN_YEARS = 2
TARGET_HORIZON = 30
TOP_K = 200
BOTTOM_K = 200
SEQ_LENGTH = 90  # 60-day lookback window with raw OHLCV data

# Trading costs - same as train.py
TRADING_COST_RATE = 0.0004
SLIPPAGE_RATE = 0.0002

# LSTM hyperparameters
LSTM_HIDDEN_SIZE = 64
LSTM_NUM_LAYERS = 2
LSTM_DROPOUT = 0.2
LSTM_EPOCHS = 50
LSTM_BATCH_SIZE = 1024
LSTM_LEARNING_RATE = 0.005


# Advanced features for LSTM
LSTM_FEATURES = [
    # 价格动量特征 (7)
    "stk_ret_1", "stk_ret_2", "stk_ret_3", "stk_ret_5", "stk_ret_10", "stk_ret_20",
    "stk_logret_1",

    # 日内价格结构特征 (4)
    "stk_hl_spread", "stk_oc_spread",
    "stk_close_to_ma_5", "stk_close_to_ma_20",

    # 波动率特征 (3)
    "stk_vol_5", "stk_vol_10", "stk_vol_20",

    # 成交量特征 (4)
    "stk_volume_chg_1",
    "stk_volume_to_ma_5", "stk_volume_to_ma_10", "stk_volume_to_ma_20",

    # 横截面特征 (3)
    "stk_ret_1_rank", "stk_vol_20_rank", "stk_volume_chg_1_rank",

    # 期权特征 (1)
    "iv_avg",

    # 交易流向特征 (4)
    "trd_individual", "trd_foreigners", "trd_securitiescos", "trd_investmenttrusts",

    # 财务/市值特征 (3)
    "stk_mcap", "stk_sector", "stk_market_segment",
]
