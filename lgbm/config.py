# LightGBM Model Configuration

import os

# Logging function
def log(msg):
    print(f"[INFO] {msg}")

OUTPUT_DIR = "output_lgbm_20d_all"
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

TEST_YEAR = 2021
ROLL_TRAIN_YEARS = 2
ROLL_RETRAIN_FREQ = "M"
TARGET_HORIZON = 20
DATA_CONFIG = "stock+all"

# LightGBM hyperparameters
LGBM_NUM_LEAVES = 31
LGBM_MAX_DEPTH = 5
LGBM_LEARNING_RATE = 0.05
LGBM_N_ESTIMATORS = 100
LGBM_SUBSAMPLE = 0.8
LGBM_COLSAMPLE_BYTREE = 0.8
LGBM_MIN_CHILD_SAMPLES = 20
LGBM_REG_ALPHA = 0.1
LGBM_REG_LAMBDA = 0.1

TOP_K = 200
BOTTOM_K = 200
