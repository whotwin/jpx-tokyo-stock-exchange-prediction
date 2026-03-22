# Transformer Model Package

from .config import (
    OUTPUT_DIR,
    PLOT_DIR,
    TEST_YEAR,
    ROLL_TRAIN_YEARS,
    TARGET_HORIZON,
    TOP_K,
    BOTTOM_K,
    SEQ_LENGTH,
    TRADING_COST_RATE,
    SLIPPAGE_RATE,
    TRANSFORMER_D_MODEL,
    TRANSFORMER_NUM_HEADS,
    TRANSFORMER_NUM_LAYERS,
    TRANSFORMER_DROPOUT,
    TRANSFORMER_EPOCHS,
    TRANSFORMER_BATCH_SIZE,
    TRANSFORMER_LEARNING_RATE,
    USE_FEATURES,
)

from .model import TransformerModel, PositionalEncoding
from .data import (
    load_data_sources,
    stock_features,
    options_features,
    trades_features,
    financials_features,
    build_feature_table,
    build_30d_labels,
    load_all_data,
    load_dataset,
    create_sequences,
)

from .train import train_transformer_model, create_train_loader, create_test_loader
from .predict import (
    predict_transformer,
    predict_with_transformer,
    evaluate_portfolio,
    calc_spread_return_sharpe,
    evaluate_predictions,
)

__all__ = [
    # Config
    "OUTPUT_DIR",
    "PLOT_DIR",
    "TEST_YEAR",
    "ROLL_TRAIN_YEARS",
    "TARGET_HORIZON",
    "TOP_K",
    "BOTTOM_K",
    "SEQ_LENGTH",
    "TRADING_COST_RATE",
    "SLIPPAGE_RATE",
    "TRANSFORMER_D_MODEL",
    "TRANSFORMER_NUM_HEADS",
    "TRANSFORMER_NUM_LAYERS",
    "TRANSFORMER_DROPOUT",
    "TRANSFORMER_EPOCHS",
    "TRANSFORMER_BATCH_SIZE",
    "TRANSFORMER_LEARNING_RATE",
    "USE_FEATURES",
    # Model
    "TransformerModel",
    "PositionalEncoding",
    # Data
    "load_data_sources",
    "stock_features",
    "options_features",
    "trades_features",
    "financials_features",
    "build_feature_table",
    "build_30d_labels",
    "load_all_data",
    "load_dataset",
    "create_sequences",
    # Train
    "train_transformer_model",
    "create_train_loader",
    "create_test_loader",
    # Predict
    "predict_transformer",
    "predict_with_transformer",
    "evaluate_portfolio",
    "calc_spread_return_sharpe",
    "evaluate_predictions",
]
