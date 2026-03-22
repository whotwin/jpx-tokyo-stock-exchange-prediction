# LGBM Prediction and Evaluation Functions
# Consistent with lstm1.py and lstm_gridsearch_cmd1.py

import gc
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from itertools import product

from .config import log, ROLL_TRAIN_YEARS, VAL_YEARS, OUTPUT_DIR
from .model import fit_model, predict_model, explain_model_shap, hyperparameter_search
from .data import prepare_expanding_window_data, get_lagged_feature_cols


def calc_rankic(y_pred, y_true):
    """Calculate RankIC (Spearman correlation) - consistent with lstm_gridsearch_cmd1.py."""
    if len(y_pred) < 2:
        return np.nan
    if np.std(y_pred) == 0 or np.std(y_true) == 0:
        return np.nan
    return spearmanr(y_pred, y_true)[0]


def evaluate_signal(df, signal_col, save_prefix):
    """
    Evaluate signal - consistent with lstm1.py evaluate_signal function.

    Calculates per-quarter RankIC, Top200Avg, Bottom200Avg, LongShortSpread.
    """
    df = df.copy()

    # Rank predictions within each quarter
    df["PredRank"] = df.groupby("Quarter")[signal_col].rank(method="first", ascending=False)

    rankic_rows = []
    top_list = []
    bottom_list = []

    log(f"\n========== Evaluating {signal_col} ==========")

    for quarter, group in df.groupby("Quarter"):
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

        log(f"  {str(quarter.date())} | RankIC={round(rank_ic, 6) if not np.isnan(rank_ic) else 'nan'} | "
            f"Top200Avg={round(top_avg, 6)} | Bottom200Avg={round(bottom_avg, 6)} | LongShort={round(long_short, 6)}")

    rankic_df = pd.DataFrame(rankic_rows)

    # Save files
    df.to_csv(os.path.join(OUTPUT_DIR, f"{save_prefix}_ranking.csv"), index=False)
    rankic_df.to_csv(os.path.join(OUTPUT_DIR, f"{save_prefix}_rankic.csv"), index=False)

    top200_df = pd.concat(top_list, ignore_index=True)
    bottom200_df = pd.concat(bottom_list, ignore_index=True)
    top200_df.to_csv(os.path.join(OUTPUT_DIR, f"{save_prefix}_top200.csv"), index=False)
    bottom200_df.to_csv(os.path.join(OUTPUT_DIR, f"{save_prefix}_bottom200.csv"), index=False)

    # Calculate overall metrics
    if df[signal_col].nunique() > 1 and df["y_true"].nunique() > 1:
        overall_rankic = spearmanr(df[signal_col], df["y_true"])[0]
    else:
        overall_rankic = np.nan

    avg_quarterly_rankic = rankic_df["RankIC"].mean()
    avg_long_short = rankic_df["LongShortSpread"].mean()

    log(f"\nSummary for {signal_col}:")
    log(f"  Overall RankIC = {round(overall_rankic, 6) if not np.isnan(overall_rankic) else 'nan'}")
    log(f"  Average Quarterly RankIC = {round(avg_quarterly_rankic, 6) if not np.isnan(avg_quarterly_rankic) else 'nan'}")
    log(f"  Average LongShort Spread = {round(avg_long_short, 6) if not np.isnan(avg_long_short) else 'nan'}")

    return {
        "signal": signal_col,
        "overall_rankic": overall_rankic,
        "avg_quarterly_rankic": avg_quarterly_rankic,
        "avg_long_short": avg_long_short
    }


def predict_with_lgbm(train_df, test_df, feature_cols, target_col="Target", do_hyperopt=True, n_trials=20):
    """Run LGBM prediction with expanding window training and hyperparameter search."""
    log(f"Running LGBM prediction with expanding window training...")
    log(f"Feature columns: {len(feature_cols)}")

    # Get lagged feature columns
    lagged_feature_cols = get_lagged_feature_cols(feature_cols, ROLL_TRAIN_YEARS)
    log(f"Lagged feature columns: {len(lagged_feature_cols)}")

    # Prepare expanding window training data
    log("\n[1/4] Preparing expanding window training data...")
    train_expand = prepare_expanding_window_data(train_df, feature_cols, ROLL_TRAIN_YEARS)

    if train_expand.empty:
        log("No training data available!")
        return pd.DataFrame()

    # Remove rows with missing target
    train_expand = train_expand[train_expand[target_col].notna()].copy()
    log(f"Training samples after removing NaN targets: {len(train_expand):,}")

    # Get feature columns for model (excluding non-feature columns)
    model_feature_cols = [c for c in train_expand.columns
                          if c not in ["SecuritiesCode", "Quarter", target_col, "Target_Quarter"]]
    log(f"Model feature columns: {len(model_feature_cols)}")

    # Hyperparameter search
    if do_hyperopt:
        log(f"\n[2/4] Running hyperparameter search ({n_trials} trials)...")
        best_params = hyperparameter_search(train_expand, model_feature_cols, target_col, n_trials=n_trials)
        log(f"Best parameters found: {best_params}")
    else:
        log("\n[2/4] Using default parameters (skipping hyperparameter search)...")
        from .config import LGBM_PARAMS
        best_params = LGBM_PARAMS

    # Get test quarters
    test_quarters = sorted(test_df["Quarter"].unique())
    log(f"\n[3/4] Running expanding window prediction...")
    log(f"Test quarters: {test_quarters}")

    all_predictions = []

    # For each test quarter, train on all previous data using expanding window
    for i, test_quarter in enumerate(test_quarters):
        log("\n" + "="*50)
        log(f"Quarter {i+1}/{len(test_quarters)}: Test on {test_quarter}")
        log("="*50)

        # Get training data: all data before test_quarter (expanding window)
        train_win = train_expand[train_expand["Target_Quarter"] < test_quarter].copy()

        # Get test data for this quarter
        infer_df = test_df[test_df["Quarter"] == test_quarter].copy()

        if train_win.empty or infer_df.empty:
            log(f"  No data for this quarter, skipping...")
            continue

        log(f"  Train samples: {len(train_win):,}, Test samples: {len(infer_df):,}")

        # Prepare test data with lagged features
        # For prediction, we need to get lagged features from the quarter before test_quarter
        prev_quarter = test_quarter - pd.DateOffset(months=3)
        train_for_lags = train_df[train_df["Quarter"] < test_quarter].copy()

        if train_for_lags.empty:
            log(f"  No data for creating lagged features, skipping...")
            continue

        # Create lagged features for test
        from .data import create_quarterly_features
        train_for_lags = create_quarterly_features(train_for_lags, feature_cols, ROLL_TRAIN_YEARS)

        if train_for_lags.empty:
            log(f"  Could not create lagged features, skipping...")
            continue

        # Get lagged features for securities in test set
        securities_test = set(infer_df["SecuritiesCode"].unique())
        securities_train = set(train_for_lags["SecuritiesCode"].unique())
        common_securities = securities_test & securities_train

        if not common_securities:
            log(f"  No common securities for prediction, skipping...")
            continue

        # Filter and prepare test data
        test_lags = train_for_lags[train_for_lags["SecuritiesCode"].isin(common_securities)].copy()

        # Get the latest lagged features for each security (most recent available)
        test_samples = []
        for sec in common_securities:
            sec_lags = test_lags[test_lags["SecuritiesCode"] == sec]
            if len(sec_lags) > 0:
                latest = sec_lags.iloc[-1:].copy()
                test_samples.append(latest)

        if not test_samples:
            log(f"  No test samples after feature creation, skipping...")
            continue

        test_features = pd.concat(test_samples, ignore_index=True)

        # Merge with actual target from test set
        # Keep the target from test data (infer_df) and rename to avoid conflict
        test_target = infer_df[["SecuritiesCode", "Quarter", target_col]].copy()
        test_target = test_target.rename(columns={"Quarter": "Target_Quarter_Actual", target_col: "Target_Actual"})

        test_features = test_features.merge(
            test_target,
            on="SecuritiesCode",
            how="inner"
        )

        if test_features.empty:
            log(f"  No matching test samples, skipping...")
            continue

        # Now rename for output
        test_features["Quarter"] = test_features["Target_Quarter_Actual"]
        test_features[target_col] = test_features["Target_Actual"]
        test_features = test_features.drop(columns=["Target_Quarter_Actual", "Target_Actual"])

        log(f"  Test samples with features: {len(test_features):,}")

        # Train model
        model = fit_model(train_win, model_feature_cols, target_col, best_params)

        # Predict
        out = test_features[["Quarter", "SecuritiesCode", target_col]].copy()
        out = out.rename(columns={target_col: "y_true"})
        out["pred"] = predict_model(model, test_features, model_feature_cols)
        out = out.rename(columns={"Quarter": "Date"})
        out["test_quarter"] = test_quarter

        all_predictions.append(out)

        del model
        gc.collect()

    # Combine all predictions
    if all_predictions:
        out = pd.concat(all_predictions, ignore_index=True)
        out = out.sort_values(["Quarter", "SecuritiesCode"]).reset_index(drop=True)
    else:
        out = pd.DataFrame()

    # Save predictions
    if not out.empty:
        from .config import OUTPUT_DIR
        pred_file = os.path.join(OUTPUT_DIR, "predictions.csv")
        out.to_csv(pred_file, index=False)
        log(f"\nPredictions saved to: {pred_file}")

    log(f"\nTotal predictions: {len(out):,}")
    return out, best_params


def train_one_fold_lgbm(train_expand, val_year, feature_cols, target_col, params, model_type="xgb"):
    """
    Train one fold and calculate RankIC per quarter.

    Consistent with lstm_gridsearch_cmd1.py's train_one_fold function.

    Args:
        train_expand: expanding window training data
        val_year: validation year
        feature_cols: feature columns for model
        target_col: target column name
        params: model hyperparameters
        model_type: "lgbm" or "xgb"

    Returns:
        fold_mean_ic: mean RankIC across quarters
        quarter_result: list of dicts with per-quarter metrics
    """
    from .model import fit_model, predict_model

    # Get train and val data
    train_mask = train_expand["Target_Quarter"].dt.year.isin([2017])  # First year for first fold
    val_mask = train_expand["Target_Quarter"].dt.year == val_year

    # Adjust train mask based on val_year
    if val_year == 2018:
        train_years = [2017]
    elif val_year == 2019:
        train_years = [2017, 2018]
    elif val_year == 2020:
        train_years = [2017, 2018, 2019]
    elif val_year == 2021:
        train_years = [2017, 2018, 2019, 2020]
    else:
        train_years = [2017]

    train_mask = train_expand["Target_Quarter"].dt.year.isin(train_years)
    val_mask = train_expand["Target_Quarter"].dt.year == val_year

    train_data = train_expand[train_mask].copy()
    val_data = train_expand[val_mask].copy()

    if train_data.empty or val_data.empty:
        return np.nan, []

    # Train model
    model = fit_model(train_data, feature_cols, target_col, params)

    # Predict on validation set
    val_data = val_data.copy()
    val_data["pred"] = predict_model(model, val_data, feature_cols)
    val_data = val_data.rename(columns={target_col: "y_true"})

    # Calculate RankIC per quarter
    quarter_result = []
    quarter_ic_list = []

    quarter_names = sorted(val_data["Target_Quarter"].unique())

    for q in quarter_names:
        temp = val_data[val_data["Target_Quarter"] == q]
        ic = calc_rankic(temp["pred"].values, temp["y_true"].values)
        quarter_ic_list.append(ic)

        quarter_result.append({
            "LabelQuarter": q.strftime("%Y-%m-%d") if hasattr(q, 'strftime') else str(q),
            "n_stocks": len(temp),
            "rankIC": ic
        })

    fold_mean_ic = np.nanmean(quarter_ic_list)

    del model
    gc.collect()

    return fold_mean_ic, quarter_result


def predict_with_lgbm_gridsearch(train_df, test_df, feature_cols, target_col="Target", do_hyperopt=True, n_trials=20):
    """
    Run LGBM grid search with expanding window validation.

    Consistent with lstm_gridsearch_cmd1.py:
    - Uses VAL_YEARS for validation
    - Calculates RankIC per quarter
    - Outputs gridsearch_results.csv and quarter_rankic_results.csv
    """
    log("="*60)
    log("LGBM Grid Search - Consistent with lstm_gridsearch_cmd1.py")
    log("="*60)

    # Get lagged feature columns
    lagged_feature_cols = get_lagged_feature_cols(feature_cols, ROLL_TRAIN_YEARS)
    log(f"Using {ROLL_TRAIN_YEARS} quarters lag")
    log(f"Feature columns: {len(lagged_feature_cols)}")

    # Prepare expanding window training data
    log("\nPreparing expanding window training data...")
    train_expand = prepare_expanding_window_data(train_df, feature_cols, ROLL_TRAIN_YEARS)

    if train_expand.empty:
        log("No training data available!")
        return pd.DataFrame(), {}

    # Remove rows with missing target
    train_expand = train_expand[train_expand[target_col].notna()].copy()
    log(f"Training samples: {len(train_expand):,}")

    # Get model feature columns
    model_feature_cols = [c for c in train_expand.columns
                          if c not in ["SecuritiesCode", "Quarter", target_col, "Target_Quarter"]]

    # Add LabelYear and LabelQuarter for filtering
    train_expand["LabelYear"] = train_expand["Target_Quarter"].dt.year
    train_expand["LabelQuarter"] = train_expand["Target_Quarter"].dt.year.astype(str) + "Q" + \
                                   ((train_expand["Target_Quarter"].dt.month - 1) // 3 + 1).astype(str)

    log(f"LabelYear distribution:\n{train_expand['LabelYear'].value_counts().sort_index()}")
    log(f"LabelQuarter sample: {sorted(train_expand['LabelQuarter'].unique())[:8]}")

    # Define folds (consistent with lstm_gridsearch_cmd1.py)
    folds = [
        ([2017], 2018),
        ([2017, 2018], 2019),
        ([2017, 2018, 2019], 2020),
        ([2017, 2018, 2019, 2020], 2021),
    ]

    # Parameter grid
    from .config import LGBM_PARAMS, XGB_PARAMS, MODEL_TYPE
    from itertools import product

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [4, 6],
        "num_leaves": [15, 31],
        "learning_rate": [0.001, 0.005],
    }

    param_combinations = list(product(
        param_grid["n_estimators"],
        param_grid["max_depth"],
        param_grid["num_leaves"],
        param_grid["learning_rate"],
    ))

    results = []
    quarter_results = []

    log(f"\n{'='*60}")
    log(f"Starting Grid Search - {len(param_combinations)} combinations")
    log(f"{'='*60}")

    for combo_idx, combo in enumerate(param_combinations, start=1):
        params = {
            "n_estimators": combo[0],
            "max_depth": combo[1],
            "num_leaves": combo[2],
            "learning_rate": combo[3],
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }

        log(f"\nCombo {combo_idx}/{len(param_combinations)}: n_est={combo[0]}, depth={combo[1]}, leaves={combo[2]}, lr={combo[3]}")

        fold_mean_list = []

        for fold_idx, (train_years, val_year) in enumerate(folds, start=1):
            log(f"  Fold {fold_idx}: train={train_years}, val={val_year}")

            # Filter data
            train_mask = train_expand["LabelYear"].isin(train_years)
            val_mask = train_expand["LabelYear"] == val_year

            train_data = train_expand[train_mask].copy()
            val_data = train_expand[val_mask].copy()

            log(f"    Train samples: {len(train_data)}, Val samples: {len(val_data)}")
            log(f"    Val quarters: {sorted(val_data['LabelQuarter'].unique())}")

            if train_data.empty or val_data.empty:
                log(f"    Skip empty fold")
                fold_mean_list.append(np.nan)
                continue

            # Train model
            model = fit_model(train_data, model_feature_cols, target_col, params)

            # Predict
            val_data = val_data.copy()
            val_data["pred"] = predict_model(model, val_data, model_feature_cols)
            val_data = val_data.rename(columns={target_col: "y_true"})

            # Calculate RankIC per quarter
            quarter_ic_list = []
            fold_quarter_result = []

            quarter_names = sorted(val_data["LabelQuarter"].unique())

            for q in quarter_names:
                temp = val_data[val_data["LabelQuarter"] == q]
                ic = calc_rankic(temp["pred"].values, temp["y_true"].values)
                quarter_ic_list.append(ic)

                fold_quarter_result.append({
                    "combo_idx": combo_idx,
                    "fold_idx": fold_idx,
                    "train_years": str(train_years),
                    "val_year": val_year,
                    "n_estimators": combo[0],
                    "max_depth": combo[1],
                    "num_leaves": combo[2],
                    "learning_rate": combo[3],
                    "LabelQuarter": q,
                    "n_stocks": len(temp),
                    "rankIC": ic
                })

            fold_mean_ic = np.nanmean(quarter_ic_list)
            fold_mean_list.append(fold_mean_ic)

            log(f"    Fold mean RankIC: {fold_mean_ic:.4f}")

            # Add quarter results
            quarter_results.extend(fold_quarter_result)

            del model
            gc.collect()

        avg_rankic = np.nanmean(fold_mean_list)

        results.append({
            "combo_idx": combo_idx,
            "n_estimators": combo[0],
            "max_depth": combo[1],
            "num_leaves": combo[2],
            "learning_rate": combo[3],
            "fold1_rankic": fold_mean_list[0] if len(fold_mean_list) > 0 else np.nan,
            "fold2_rankic": fold_mean_list[1] if len(fold_mean_list) > 1 else np.nan,
            "fold3_rankic": fold_mean_list[2] if len(fold_mean_list) > 2 else np.nan,
            "fold4_rankic": fold_mean_list[3] if len(fold_mean_list) > 3 else np.nan,
            "avg_rankic": avg_rankic
        })

        log(f"  Combo avg RankIC: {avg_rankic:.4f}")

    # Save results
    log(f"\n{'='*60}")
    log("Saving Results")
    log(f"{'='*60}")

    results_df = pd.DataFrame(results).sort_values("avg_rankic", ascending=False).reset_index(drop=True)
    quarter_results_df = pd.DataFrame(quarter_results)

    results_df.to_csv(os.path.join(OUTPUT_DIR, "gridsearch_results.csv"), index=False)
    quarter_results_df.to_csv(os.path.join(OUTPUT_DIR, "quarter_rankic_results.csv"), index=False)

    log(f"Parameter combinations: {len(results_df)}")
    log(f"Quarter-level RankIC: {len(quarter_results_df)}")

    log(f"\nTop 10 results:")
    log(results_df.head(10).to_string(index=False))

    if len(results_df) > 0:
        best_params = {
            "n_estimators": int(results_df.iloc[0]["n_estimators"]),
            "max_depth": int(results_df.iloc[0]["max_depth"]),
            "num_leaves": int(results_df.iloc[0]["num_leaves"]),
            "learning_rate": float(results_df.iloc[0]["learning_rate"]),
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }
        log(f"\nBest params: {best_params}")
        log(f"Best avg RankIC: {results_df.iloc[0]['avg_rankic']:.4f}")

        # Train best model and evaluate with signal_pos/signal_neg (consistent with lstm1.py)
        log("\n" + "="*60)
        log("Training final model for signal evaluation...")
        log("="*60)

        # Use all training data for final model
        final_train = train_expand.copy()
        log(f"Training samples: {len(final_train):,}")

        # Prepare test data for 2021 evaluation
        test_quarters = sorted(test_df["Quarter"].unique())
        log(f"Test quarters: {test_quarters}")

        all_test_preds = []
        for test_quarter in test_quarters:
            # Get training data for this quarter
            train_for_quarter = train_expand[train_expand["Target_Quarter"] < test_quarter].copy()
            if train_for_quarter.empty:
                continue

            # Get test data
            infer_df = test_df[test_df["Quarter"] == test_quarter].copy()
            if infer_df.empty:
                continue

            # Create lagged features for test
            train_for_lags = train_df[train_df["Quarter"] < test_quarter].copy()
            if train_for_lags.empty:
                continue

            from .data import create_quarterly_features
            train_for_lags = create_quarterly_features(train_for_lags, feature_cols, ROLL_TRAIN_YEARS)

            if train_for_lags.empty:
                continue

            # Get common securities
            securities_test = set(infer_df["SecuritiesCode"].unique())
            securities_train = set(train_for_lags["SecuritiesCode"].unique())
            common_securities = securities_test & securities_train

            if not common_securities:
                continue

            # Prepare test features
            test_lags = train_for_lags[train_for_lags["SecuritiesCode"].isin(common_securities)].copy()
            test_samples = []
            for sec in common_securities:
                sec_lags = test_lags[test_lags["SecuritiesCode"] == sec]
                if len(sec_lags) > 0:
                    latest = sec_lags.iloc[-1:].copy()
                    test_samples.append(latest)

            if not test_samples:
                continue

            test_features = pd.concat(test_samples, ignore_index=True)

            # Merge with target
            test_target = infer_df[["SecuritiesCode", "Quarter", target_col]].copy()
            test_target = test_target.rename(columns={"Quarter": "Target_Quarter_Actual", target_col: "Target_Actual"})
            test_features = test_features.merge(test_target, on="SecuritiesCode", how="inner")

            if test_features.empty:
                continue

            test_features["Quarter"] = test_features["Target_Quarter_Actual"]
            test_features[target_col] = test_features["Target_Actual"]
            test_features = test_features.drop(columns=["Target_Quarter_Actual", "Target_Actual"])

            # Train model
            model = fit_model(train_for_quarter, model_feature_cols, target_col, best_params)

            # Predict
            out = test_features[["Quarter", "SecuritiesCode", target_col]].copy()
            out = out.rename(columns={target_col: "y_true"})
            out["y_pred"] = predict_model(model, test_features, model_feature_cols)

            all_test_preds.append(out)
            del model
            gc.collect()

        # Combine all predictions and evaluate
        if all_test_preds:
            ranking_df = pd.concat(all_test_preds, ignore_index=True)
            ranking_df = ranking_df.sort_values(["Quarter", "SecuritiesCode"]).reset_index(drop=True)

            # Add signal_pos and signal_neg (consistent with lstm1.py)
            ranking_df["signal_pos"] = ranking_df["y_pred"]
            ranking_df["signal_neg"] = -ranking_df["y_pred"]

            # Save ranking
            ranking_df.to_csv(os.path.join(OUTPUT_DIR, "test_2021_ranking.csv"), index=False)
            log(f"\nRanking saved to: {os.path.join(OUTPUT_DIR, 'test_2021_ranking.csv')}")

            # Evaluate both signals
            result_pos = evaluate_signal(ranking_df, "signal_pos", "signal_pos")
            result_neg = evaluate_signal(ranking_df, "signal_neg", "signal_neg")

            # Summary comparison
            summary_df = pd.DataFrame([result_pos, result_neg])
            summary_df.to_csv(os.path.join(OUTPUT_DIR, "signal_compare_summary.csv"), index=False)

            log("\n" + "="*60)
            log("FINAL COMPARISON")
            log("="*60)
            log(summary_df.to_string(index=False))
            log(f"\nSaved in: {OUTPUT_DIR}")

    return results_df, quarter_results_df


def run_with_shap(train_df, test_df, feature_cols, target_col="Target", best_params=None):
    """Run final model with SHAP analysis."""
    log("\n[4/4] Running final model with SHAP analysis...")

    from .config import OUTPUT_DIR

    # Prepare expanding window training data
    train_expand = prepare_expanding_window_data(train_df, feature_cols, ROLL_TRAIN_YEARS)
    train_expand = train_expand[train_expand[target_col].notna()].copy()

    if train_expand.empty:
        log("No training data for SHAP analysis!")
        return None

    # Get feature columns for model
    model_feature_cols = [c for c in train_expand.columns
                          if c not in ["SecuritiesCode", "Quarter", target_col, "Target_Quarter"]]

    # Train final model on all data before 2021
    if best_params is None:
        from .config import LGBM_PARAMS
        best_params = LGBM_PARAMS

    # Use all training data (excluding 2021) for final model
    final_train = train_expand.copy()
    log(f"Training final model with {len(final_train):,} samples...")

    model = fit_model(final_train, model_feature_cols, target_col, best_params)

    # SHAP analysis
    importance_df = explain_model_shap(model, final_train, model_feature_cols, OUTPUT_DIR)

    return model, importance_df
