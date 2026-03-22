# iTransformer Model Configuration

import os
import torch

# Logging function
def log(msg):
    print(f"[INFO] {msg}")

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

OUTPUT_DIR = "output_itransformer"
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# Configuration
TEST_YEAR = 2021
ROLL_TRAIN_YEARS = 2
TARGET_HORIZON = 30
TOP_K = 200
BOTTOM_K = 200
SEQ_LENGTH = 20  # 20-day lookback window

# Trading costs
TRADING_COST_RATE = 0.0004
SLIPPAGE_RATE = 0.0002

# iTransformer hyperparameters
ITRANSFORMER_D_MODEL = 128
ITRANSFORMER_NUM_HEADS = 4
ITRANSFORMER_NUM_LAYERS = 2
ITRANSFORMER_DROPOUT = 0.1
ITRANSFORMER_EPOCHS = 10
ITRANSFORMER_BATCH_SIZE = 4
ITRANSFORMER_LEARNING_RATE = 0.001
ITRANSFORMER_EMBED_DIM = 64

# Features
USE_FEATURES = [
    # Stock returns (7)
    "stk_ret_1", "stk_ret_2", "stk_ret_3", "stk_ret_5", "stk_ret_10", "stk_ret_20",
    "stk_logret_1",

    # Volatility (3)
    "stk_vol_5", "stk_vol_10", "stk_vol_20",

    # Spreads (2)
    "stk_hl_spread", "stk_oc_spread",

    # Volume (4)
    "stk_volume_chg_1",
    "stk_volume_to_ma_5", "stk_volume_to_ma_10", "stk_volume_to_ma_20",

    # Rolling mean returns (3)
    "stk_ret_mean_5", "stk_ret_mean_10", "stk_ret_mean_20",

    # MA ratios (3)
    "stk_close_to_ma_5", "stk_close_to_ma_10", "stk_close_to_ma_20",

    # Distribution (1)
    "stk_skew_20",

    # Time (2)
    "stk_dayofweek", "stk_month",

    # Company (2)
    "stk_expected_dividend", "stk_supervision_flag",

    # Fundamentals (3)
    "stk_mcap", "stk_sector", "stk_market_segment",

    # Options (1)
    "iv_avg",

    # Trades (4)
    "trd_individual", "trd_foreigners", "trd_securitiescos", "trd_investmenttrusts",

    # Financials (6)
    "fin_netsales", "fin_operatingprofit", "fin_ordinaryprofit",
    "fin_profit", "fin_totalassets", "fin_equity",
]
