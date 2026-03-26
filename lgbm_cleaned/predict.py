# LGBM Prediction and Evaluation Functions
# Consistent with lstm1.py and lstm_gridsearch_cmd1.py

import gc
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from itertools import product

from .config import log, OUTPUT_DIR
from .model import fit_model, predict_model, explain_model_shap
from .data import prepare_expanding_window_data


def calc_rankic(y_pred, y_true):
    """Calculate RankIC (Spearman correlation) - consistent with lstm_gridsearch_cmd1.py."""
    if len(y_pred) < 2:
        return np.nan
    if np.std(y_pred) == 0 or np.std(y_true) == 0:
        return np.nan
    return spearmanr(y_pred, y_true)[0]


def predict_with_lgbm_gridsearch(train_df, feature_cols, target_col="Target"):
    """
    Run LGBM grid search with expanding window validation.
    Consistent with lstm_gridsearch_cmd1.py:
    - 3 folds: (train[2017], val[2018]), (train[2017,2018], val[2019]), (train[2017,2018,2019], val[2020])
    - Calculates RankIC per quarter
    - Outputs gridsearch_results.csv and quarter_rankic_results.csv
    """
    log("=" * 60)
    log("LGBM Grid Search - Consistent with lstm_gridsearch_cmd1.py")
    log("=" * 60)

    log(f"Feature columns: {len(feature_cols)}")

    # Prepare expanding window training data (no lag features)
    train_expand = prepare_expanding_window_data(train_df, target_col)

    if train_expand.empty:
        log("No training data available!")
        return pd.DataFrame(), pd.DataFrame()

    # Get model feature columns
    model_feature_cols = [c for c in train_expand.columns
                          if c not in ["SecuritiesCode", "Quarter", target_col,
                                       "LabelYear", "LabelQuarter"]]

    log(f"Model feature columns: {len(model_feature_cols)}")
    log(f"LabelYear distribution:\n{train_expand['LabelYear'].value_counts().sort_index()}")
    log(f"LabelQuarter sample: {sorted(train_expand['LabelQuarter'].unique())}")

    # Define 3 folds (consistent with lstm_gridsearch_cmd1.py)
    folds = [
        ([2017], 2018),
        ([2017, 2018], 2019),
        ([2017, 2018, 2019], 2020),
    ]

    # Parameter grid
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

    log(f"\n{'=' * 60}")
    log(f"Starting Grid Search - {len(param_combinations)} combinations")
    log(f"{'=' * 60}")

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

        log(f"\nCombo {combo_idx}/{len(param_combinations)}: "
            f"n_est={combo[0]}, depth={combo[1]}, "
            f"leaves={combo[2]}, lr={combo[3]}")

        fold_mean_list = []

        for fold_idx, (train_years, val_year) in enumerate(folds, start=1):
            log(f"  Fold {fold_idx}: train={train_years}, val={val_year}")

            # Filter data
            train_mask = train_expand["LabelYear"].isin(train_years)
            val_mask = train_expand["LabelYear"] == val_year

            train_data = train_expand[train_mask].copy()
            val_data = train_expand[val_mask].copy()

            log(f"    Train samples: {len(train_data)}, Val samples: {len(val_data)}")

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

            # Calculate RankIC and LongShort per quarter
            quarter_ic_list = []
            fold_quarter_result = []

            quarter_names = sorted(val_data["LabelQuarter"].unique())
            log(f"    Val quarters: {quarter_names}")

            for q in quarter_names:
                temp = val_data[val_data["LabelQuarter"] == q].sort_values("pred", ascending=False).reset_index(drop=True)
                ic = calc_rankic(temp["pred"].values, temp["y_true"].values)
                top_avg = temp.head(20)["y_true"].mean()
                bottom_avg = temp.tail(20)["y_true"].mean()
                long_short = top_avg - bottom_avg
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
                    "rankIC": ic,
                    "Top20AvgTrue": top_avg,
                    "Bottom20AvgTrue": bottom_avg,
                    "LongShortSpread": long_short
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
            "avg_rankic": avg_rankic
        })

        log(f"  Combo avg RankIC: {avg_rankic:.4f}")

    # Save results
    log(f"\n{'=' * 60}")
    log("Saving Results")
    log(f"{'=' * 60}")

    results_df = pd.DataFrame(results).sort_values("avg_rankic", ascending=False).reset_index(drop=True)
    quarter_results_df = pd.DataFrame(quarter_results)

    # Compute avg_long_short for each combo from quarter results
    avg_ls = quarter_results_df.groupby("combo_idx")["LongShortSpread"].mean().reset_index()
    avg_ls.columns = ["combo_idx", "avg_long_short"]
    results_df = results_df.merge(avg_ls, on="combo_idx")

    results_df.to_csv(os.path.join(OUTPUT_DIR, "gridsearch_results.csv"), index=False)
    quarter_results_df.to_csv(os.path.join(OUTPUT_DIR, "quarter_rankic_results.csv"), index=False)

    log(f"Parameter combinations: {len(results_df)}")
    log(f"Quarter-level RankIC: {len(quarter_results_df)}")

    log(f"\nTop 10 results:")
    log(results_df.head(10).to_string(index=False))

    if len(results_df) > 0:
        log(f"\nBest avg RankIC: {results_df.iloc[0]['avg_rankic']:.4f}")

    return results_df, quarter_results_df


def predict_2021_final(train_df, test_df, target_col="Target", best_params=None):
    """
    Final evaluation: train on 2017-2020, predict 2021.
    Consistent with lstm1.py (step 9-11).
    """
    log("\n" + "=" * 60)
    log("Final 2021 Evaluation - Train on 2017-2020, Predict 2021")
    log("=" * 60)

    # Prepare training data (2017-2020 only)
    train_expand = prepare_expanding_window_data(train_df, target_col)

    # Filter to 2017-2020
    train_expand = train_expand[train_expand["LabelYear"] < 2021].copy()

    # Prepare test data (2021)
    test_2021 = test_df.copy()
    test_2021["LabelYear"] = test_2021["Quarter"].dt.year
    test_2021["LabelQuarter"] = test_2021["Quarter"].dt.year.astype(str) + "Q" + \
                                 ((test_2021["Quarter"].dt.month - 1) // 3 + 1).astype(str)

    # Get feature columns (exclude non-feature columns)
    model_feature_cols = [c for c in train_expand.columns
                         if c not in ["SecuritiesCode", "Quarter", target_col,
                                      "LabelYear", "LabelQuarter"]]

    log(f"Training samples (2017-2020): {len(train_expand):,}")
    log(f"Test samples (2021): {len(test_2021):,}")
    log(f"Model feature columns: {len(model_feature_cols)}")

    # Use default params if not provided
    if best_params is None:
        best_params = {
            "n_estimators": 200,
            "max_depth": 6,
            "num_leaves": 31,
            "learning_rate": 0.005,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }

    # Train final model
    model = fit_model(train_expand, model_feature_cols, target_col, best_params)

    # Predict on 2021 test data
    test_2021 = test_2021.copy()
    test_2021["pred"] = predict_model(model, test_2021, model_feature_cols)
    test_2021 = test_2021.rename(columns={target_col: "y_true"})

    # Save detailed predictions
    ranking_detail = test_2021[["Quarter", "SecuritiesCode", "y_true", "pred"]].copy()
    ranking_detail = ranking_detail.sort_values(["Quarter", "SecuritiesCode"]).reset_index(drop=True)
    ranking_detail.to_csv(os.path.join(OUTPUT_DIR, "test_2021_ranking_detail.csv"), index=False)

    # Calculate RankIC and LongShort per quarter
    rankic_rows = []
    for quarter, group in test_2021.groupby("Quarter"):
        group = group.sort_values("pred", ascending=False).reset_index(drop=True)

        n_stocks = len(group)
        ic = calc_rankic(group["pred"].values, group["y_true"].values)

        top_avg = group.head(20)["y_true"].mean()
        bottom_avg = group.tail(20)["y_true"].mean()
        long_short = top_avg - bottom_avg

        rankic_rows.append({
            "Quarter": quarter,
            "NumStocks": n_stocks,
            "RankIC": ic,
            "Top20AvgTrue": top_avg,
            "Bottom20AvgTrue": bottom_avg,
            "LongShortSpread": long_short
        })

        log(f"  {quarter.strftime('%YQ%m')}: RankIC={ic:.4f}, Top20={top_avg:.4f}, Bottom20={bottom_avg:.4f}, LS={long_short:.4f}")

    rankic_df = pd.DataFrame(rankic_rows)
    rankic_df.to_csv(os.path.join(OUTPUT_DIR, "test_2021_rankic.csv"), index=False)

    overall_ic = calc_rankic(test_2021["pred"].values, test_2021["y_true"].values)
    avg_quarterly_ic = rankic_df["RankIC"].mean()
    avg_long_short = rankic_df["LongShortSpread"].mean()

    log(f"\nOverall RankIC: {overall_ic:.4f}")
    log(f"Average Quarterly RankIC: {avg_quarterly_ic:.4f}")
    log(f"Average LongShort Spread: {avg_long_short:.4f}")
    log(f"Results saved to: {OUTPUT_DIR}/test_2021_*.csv")

    del model
    gc.collect()

    return ranking_detail, rankic_df


def run_with_shap(train_df, target_col="Target", best_params=None):
    """Run final model with SHAP analysis using 2017-2020 data."""
    log("\n" + "=" * 60)
    log("SHAP Analysis - Train on 2017-2020")
    log("=" * 60)

    # Prepare expanding window training data
    train_expand = prepare_expanding_window_data(train_df, target_col)

    # Filter to 2017-2020
    train_expand = train_expand[train_expand["LabelYear"] < 2021].copy()

    if train_expand.empty:
        log("No training data for SHAP analysis!")
        return None, None

    # Get feature columns for model
    model_feature_cols = [c for c in train_expand.columns
                         if c not in ["SecuritiesCode", "Quarter", target_col,
                                      "LabelYear", "LabelQuarter"]]

    # Use default params if not provided
    if best_params is None:
        best_params = {
            "n_estimators": 200,
            "max_depth": 6,
            "num_leaves": 31,
            "learning_rate": 0.005,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }

    log(f"Training final model with {len(train_expand):,} samples...")

    model = fit_model(train_expand, model_feature_cols, target_col, best_params)

    # SHAP analysis
    importance_df = explain_model_shap(model, train_expand, model_feature_cols, OUTPUT_DIR)

    return model, importance_df
