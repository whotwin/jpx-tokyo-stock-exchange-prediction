# LGBM Model Configuration

import os

# Logging function
def log(msg):
    print(f"[INFO] {msg}")


OUTPUT_DIR = "output_lgbm"
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# Configuration
TEST_YEAR = 2021
ROLL_TRAIN_YEARS = 3  # Use 3 quarters to predict next quarter (consistent with lstm_gridsearch)
TARGET_HORIZON = 1  # Predict next quarter
TOP_K = 200
BOTTOM_K = 200

# Trading costs
TRADING_COST_RATE = 0.0004
SLIPPAGE_RATE = 0.0002

# Use GPU
USE_GPU = os.environ.get("CUDA_VISIBLE_DEVICES", "") != ""

# Validation years (consistent with lstm_gridsearch_cmd1.py)
VAL_YEARS = [2018, 2019, 2020]

# LGBM default hyperparameters (will be tuned)
LGBM_PARAMS = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "max_depth": 6,
    "num_leaves": 31,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

# XGBoost default parameters (with GPU support)
XGB_PARAMS = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "max_depth": 6,
    "min_child_weight": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "random_state": 42,
    "tree_method": "hist",  # Use hist for faster training
    "device": "cuda" if USE_GPU else "cpu",
    "verbosity": 0,
}

# Hyperparameter search space (for grid search, consistent with lstm_gridsearch)
PARAM_GRID = {
    "n_estimators": [100, 200],
    "max_depth": [4, 6],
    "num_leaves": [15, 31],
    "learning_rate": [0.001, 0.005, 0.01],
}

# Data paths
TRAIN_DATA_PATH = "cleaned_data/train_dataset_clean.csv"
TEST_DATA_PATH = "cleaned_data/test_dataset.csv"

# CV settings
N_CV_FOLDS = 3  # Number of CV folds for hyperparameter search
CV_WINDOW_SIZE = 3  # Number of quarters for CV validation (consistent with lstm)

# Model choice: "lgbm" or "xgb"
MODEL_TYPE = "xgb"  # Use XGBoost with GPU by default
