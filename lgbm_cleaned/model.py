# LGBM Model Definition

import numpy as np
import pandas as pd
import lightgbm as lgb
from lightgbm import LGBMRegressor

# Optional: xgboost
try:
    import xgboost as xgb
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    xgb = None
    XGBRegressor = None

from sklearn.model_selection import TimeSeriesSplit
import os

# Optional: optuna
try:
    import optuna
    from optuna.samplers import TPESampler
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    optuna = None

# Optional: shap
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    shap = None

from .config import log, LGBM_PARAMS, N_CV_FOLDS


def fit_lgbm(train_df, feature_cols, target_col, params=None):
    """Train LGBM model."""
    if params is None:
        params = LGBM_PARAMS

    log(f"Training LGBM with {len(train_df)} samples...")

    model = LGBMRegressor(**params)
    model.fit(
        train_df[feature_cols],
        train_df[target_col].values,
    )

    return model


def fit_xgb(train_df, feature_cols, target_col, params=None):
    """Train XGBoost model with GPU support."""
    if not HAS_XGBOOST:
        raise ImportError("XGBoost is not installed. Please install xgboost or use LGBM.")
    if params is None:
        from .config import USE_GPU, XGB_PARAMS
        params = XGB_PARAMS.copy()
        params["device"] = "cuda" if USE_GPU else "cpu"

    log(f"Training XGBoost with {len(train_df)} samples...")

    model = XGBRegressor(**params)
    model.fit(
        train_df[feature_cols],
        train_df[target_col].values,
    )

    return model


def fit_model(train_df, feature_cols, target_col, params=None):
    """Train model - uses LGBM by default."""
    return fit_lgbm(train_df, feature_cols, target_col, params)


def predict_lgbm(model, test_df, feature_cols):
    """Predict with LGBM model."""
    predictions = model.predict(test_df[feature_cols])
    return predictions


def predict_xgb(model, test_df, feature_cols):
    """Predict with XGBoost model."""
    predictions = model.predict(test_df[feature_cols])
    return predictions


def predict_model(model, test_df, feature_cols):
    """Predict - uses LGBM by default."""
    return predict_lgbm(model, test_df, feature_cols)


def calculate_sharpe(pred_df, top_k=200, bottom_k=200):
    """
    Calculate Sharpe ratio based on portfolio strategy.

    Buy top_k predicted returns, sell bottom_k predicted returns.
    """
    if pred_df.empty or len(pred_df) < top_k + bottom_k:
        return np.nan

    # Sort by prediction
    sorted_df = pred_df.sort_values("pred", ascending=False).reset_index(drop=True)

    # Get top and bottom
    top_returns = sorted_df.head(top_k)["y_true"].values
    bottom_returns = sorted_df.tail(bottom_k)["y_true"].values

    # Calculate spread
    spread = np.mean(top_returns) - np.mean(bottom_returns)

    return spread


def cv_objective_lgbm(trial, train_df, feature_cols, target_col, n_splits=3):
    """Objective function for Optuna LGBM hyperparameter search."""
    # Sample hyperparameters
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "num_leaves": trial.suggest_int("num_leaves", 10, 64),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }

    return cv_evaluate(train_df, feature_cols, target_col, n_splits, params, "lgbm")


def cv_objective_xgb(trial, train_df, feature_cols, target_col, n_splits=3):
    """Objective function for Optuna XGBoost hyperparameter search."""
    # Sample hyperparameters
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 5, 50),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "random_state": 42,
        "tree_method": "hist",
        "device": "cuda",
        "verbosity": 0,
    }

    return cv_evaluate(train_df, feature_cols, target_col, n_splits, params, "xgb")


def cv_evaluate(train_df, feature_cols, target_col, n_splits, params, model_type):
    """Common CV evaluation function."""
    # Get unique quarters for time-series CV
    quarters = sorted(train_df["Target_Quarter"].unique())

    if len(quarters) < n_splits + 1:
        return -np.inf

    # Time-series split
    tscv = TimeSeriesSplit(n_splits=n_splits)

    sharpe_scores = []

    for train_idx, val_idx in tscv.split(quarters):
        train_quarters = [quarters[i] for i in train_idx]
        val_quarters = [quarters[i] for i in val_idx]

        # Get train and validation data
        cv_train = train_df[train_df["Target_Quarter"].isin(train_quarters)].copy()
        cv_val = train_df[train_df["Target_Quarter"].isin(val_quarters)].copy()

        if cv_train.empty or cv_val.empty:
            continue

        # Remove rows with missing target
        cv_train = cv_train[cv_train[target_col].notna()].copy()
        cv_val = cv_val[cv_val[target_col].notna()].copy()

        if len(cv_train) < 100 or len(cv_val) < 100:
            continue

        # Train model
        try:
            if model_type == "xgb":
                model = XGBRegressor(**params)
            else:
                model = LGBMRegressor(**params)

            model.fit(
                cv_train[feature_cols],
                cv_train[target_col].values,
            )

            # Predict
            cv_val = cv_val.copy()
            cv_val["pred"] = model.predict(cv_val[feature_cols])
            cv_val = cv_val.rename(columns={target_col: "y_true"})

            # Calculate Sharpe
            sharpe = calculate_sharpe(cv_val)
            if not np.isnan(sharpe):
                sharpe_scores.append(sharpe)

            del model
        except Exception as e:
            log(f"CV error: {e}")
            continue

    if not sharpe_scores:
        return -np.inf

    return np.mean(sharpe_scores)


def hyperparameter_search(train_df, feature_cols, target_col="Target", n_trials=20, n_splits=3):
    """
    Search for best hyperparameters using Optuna.
    """
    from .config import MODEL_TYPE

    log(f"Starting hyperparameter search with {n_trials} trials using {MODEL_TYPE}...")

    # Prepare data - remove NaN targets
    train_data = train_df[train_df[target_col].notna()].copy()

    if len(train_data) < 300:
        log(f"Not enough data for hyperparameter search ({len(train_data)} samples)")
        if MODEL_TYPE == "xgb":
            return XGB_PARAMS.copy()
        return LGBM_PARAMS.copy()

    # Create study
    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    # Select objective function based on model type
    if MODEL_TYPE == "xgb":
        objective = lambda trial: cv_objective_xgb(trial, train_data, feature_cols, target_col, n_splits)
    else:
        objective = lambda trial: cv_objective_lgbm(trial, train_data, feature_cols, target_col, n_splits)

    # Run optimization
    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=True,
    )

    log(f"Best trial Sharpe: {study.best_value:.4f}")
    log(f"Best params: {study.best_params}")

    # Return best parameters
    best_params = study.best_params.copy()
    best_params["random_state"] = 42

    if MODEL_TYPE == "xgb":
        best_params["tree_method"] = "hist"
        best_params["device"] = "cuda"
        best_params["verbosity"] = 0
    else:
        best_params["n_jobs"] = -1
        best_params["verbose"] = -1

    return best_params


def explain_model_shap(model, train_df, feature_cols, output_dir):
    """
    Explain model using SHAP values.
    """
    log("Computing SHAP values...")

    try:
        # Use a subset of data for SHAP calculation if data is too large
        if len(train_df) > 5000:
            sample_df = train_df.sample(n=5000, random_state=42)
        else:
            sample_df = train_df

        # Create SHAP explainer
        explainer = shap.TreeExplainer(model)

        # Calculate SHAP values
        shap_values = explainer.shap_values(sample_df[feature_cols])

        # Calculate mean absolute SHAP values for feature importance
        shap_importance = np.abs(shap_values).mean(axis=0)
        importance_df = pd.DataFrame({
            "feature": feature_cols,
            "importance": shap_importance
        }).sort_values("importance", ascending=False)

        # Save feature importance
        importance_path = os.path.join(output_dir, "shap_feature_importance.csv")
        importance_df.to_csv(importance_path, index=False)
        log(f"SHAP feature importance saved to: {importance_path}")

        # Plot top 20 features
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            plt.figure(figsize=(12, 8))
            top_features = importance_df.head(20)
            plt.barh(range(len(top_features)), top_features["importance"].values)
            plt.yticks(range(len(top_features)), top_features["feature"].values)
            plt.xlabel("Mean |SHAP value|")
            plt.title("Top 20 Feature Importance (SHAP)")
            plt.tight_layout()

            plot_path = os.path.join(output_dir, "shap_feature_importance.png")
            plt.savefig(plot_path, dpi=150)
            plt.close()
            log(f"SHAP plot saved to: {plot_path}")
        except Exception as e:
            log(f"Could not create SHAP plot: {e}")

        # Save summary
        log("\nTop 10 most important features:")
        for i, row in importance_df.head(10).iterrows():
            log(f"  {row['feature']}: {row['importance']:.4f}")

        return importance_df

    except Exception as e:
        log(f"SHAP analysis failed: {e}")
        return None
