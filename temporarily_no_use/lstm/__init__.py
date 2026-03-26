# LSTM Model Package

from .config import (
    OUTPUT_DIR,
    PLOT_DIR,
    USE_FEATURES,
    RAW_FEATURES,
    TEST_YEAR,
    ROLL_TRAIN_YEARS,
    TARGET_HORIZON,
    TOP_K,
    BOTTOM_K,
    SEQ_LENGTH,
    TRADING_COST_RATE,
    SLIPPAGE_RATE,
    LSTM_HIDDEN_SIZE,
    LSTM_NUM_LAYERS,
    LSTM_DROPOUT,
    LSTM_EPOCHS,
    LSTM_BATCH_SIZE,
    LSTM_LEARNING_RATE,
    LSTM_FEATURES,
)

from .model import LSTMModel
from .data import (
    load_data_sources,
    stock_features,
    options_features,
    trades_features,
    financials_features,
    build_feature_table,
    build_30d_labels,
    load_all_data,
    add_cross_sectional_features,
    load_all_data_for_lstm,
    cross_sectional_normalize,
    create_sequences,
)

from .train import train_lstm_model, create_train_loader, create_test_loader
from .predict import predict_lstm, predict_with_lstm, calc_spread_return_sharpe, evaluate_predictions

__all__ = [
    # Config
    "OUTPUT_DIR",
    "PLOT_DIR",
    "USE_FEATURES",
    "RAW_FEATURES",
    "TEST_YEAR",
    "ROLL_TRAIN_YEARS",
    "TARGET_HORIZON",
    "TOP_K",
    "BOTTOM_K",
    "SEQ_LENGTH",
    "TRADING_COST_RATE",
    "SLIPPAGE_RATE",
    "LSTM_HIDDEN_SIZE",
    "LSTM_NUM_LAYERS",
    "LSTM_DROPOUT",
    "LSTM_EPOCHS",
    "LSTM_BATCH_SIZE",
    "LSTM_LEARNING_RATE",
    "LSTM_FEATURES",
    # Model
    "LSTMModel",
    # Data
    "load_data_sources",
    "stock_features",
    "options_features",
    "trades_features",
    "financials_features",
    "build_feature_table",
    "build_30d_labels",
    "load_all_data",
    "add_cross_sectional_features",
    "load_all_data_for_lstm",
    "cross_sectional_normalize",
    "create_sequences",
    # Train
    "train_lstm_model",
    "create_train_loader",
    "create_test_loader",
    # Predict
    "predict_lstm",
    "predict_with_lstm",
    "calc_spread_return_sharpe",
    "evaluate_predictions",
]
