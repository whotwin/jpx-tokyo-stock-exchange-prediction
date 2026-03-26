# iTransformer Model Package

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
    ITRANSFORMER_D_MODEL,
    ITRANSFORMER_NUM_HEADS,
    ITRANSFORMER_NUM_LAYERS,
    ITRANSFORMER_DROPOUT,
    ITRANSFORMER_EPOCHS,
    ITRANSFORMER_BATCH_SIZE,
    ITRANSFORMER_LEARNING_RATE,
    ITRANSFORMER_EMBED_DIM,
    USE_FEATURES,
    log,
)

from .data import (
    to_num,
    load_data_sources,
    stock_features,
    build_30d_labels,
    options_features,
    trades_features,
    financials_features,
    load_all_data,
    prepare_itransformer_data,
    create_itransformer_sequences,
)

from .train import (
    train_itransformer_model,
    cross_sectional_normalize,
    masked_mse_loss,
    FeatureProjector,
)

from .predict import (
    predict_itransformer,
    predict_with_itransformer,
    calc_spread_return_sharpe,
    prepare_for_kaggle_eval,
    evaluate_portfolio_kaggle,
    evaluate_portfolio,
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
    "ITRANSFORMER_D_MODEL",
    "ITRANSFORMER_NUM_HEADS",
    "ITRANSFORMER_NUM_LAYERS",
    "ITRANSFORMER_DROPOUT",
    "ITRANSFORMER_EPOCHS",
    "ITRANSFORMER_BATCH_SIZE",
    "ITRANSFORMER_LEARNING_RATE",
    "ITRANSFORMER_EMBED_DIM",
    "USE_FEATURES",
    "log",
    # Data
    "to_num",
    "load_data_sources",
    "stock_features",
    "build_30d_labels",
    "options_features",
    "trades_features",
    "financials_features",
    "load_all_data",
    "prepare_itransformer_data",
    "create_itransformer_sequences",
    # Train
    "train_itransformer_model",
    "cross_sectional_normalize",
    "masked_mse_loss",
    "FeatureProjector",
    # Predict
    "predict_itransformer",
    "predict_with_itransformer",
    "calc_spread_return_sharpe",
    "prepare_for_kaggle_eval",
    "evaluate_portfolio_kaggle",
    "evaluate_portfolio",
    "evaluate_predictions",
]
