# LGBM Main Entry Point

import os
import sys

# Handle both module import and script execution
if __name__ == "__main__" and __package__ is None:
    # Running as script - add parent directory to path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    # Now we can import from the package
    from lgbm_cleaned.config import OUTPUT_DIR, log
    from lgbm_cleaned.data import load_cleaned_data, prepare_data
    from lgbm_cleaned.predict import predict_with_lgbm_gridsearch, run_with_shap
else:
    # Running as module - use relative imports
    from .config import OUTPUT_DIR, log
    from .data import load_cleaned_data, prepare_data
    from .predict import predict_with_lgbm_gridsearch, run_with_shap


def main(do_gridsearch=True, do_shap=True):
    """Main function to run LGBM model grid search (consistent with lstm_gridsearch_cmd1.py).

    Args:
        do_gridsearch: If True, run grid search. If False, skip and use default params.
        do_shap: If True, run SHAP analysis after grid search.
    """
    print("=" * 60)
    print("JPX Stock Prediction - LGBM Grid Search")
    print("Consistent with lstm_gridsearch_cmd1.py")
    print("=" * 60)

    # Load data
    print("\n[1/4] Loading data...")
    train_df, test_df = load_cleaned_data()

    # Prepare data
    print("\n[2/4] Preparing data...")
    train_df_processed, test_df_2021, feature_cols = prepare_data(train_df, test_df)

    print(f"Feature columns ({len(feature_cols)})")
    print(f"ROLL_TRAIN_YEARS: 3 quarters")

    # Run grid search (consistent with lstm_gridsearch_cmd1.py)
    print("\n[3/4] Running LGBM grid search...")
    results_df, quarter_results_df = predict_with_lgbm_gridsearch(
        train_df_processed,
        test_df_2021,
        feature_cols,
        target_col="Target",
        do_hyperopt=do_gridsearch
    )

    if results_df is None or results_df.empty:
        print("Grid search failed!")
        return

    print("\n" + "=" * 60)
    print("Grid Search Results Summary:")
    print("=" * 60)
    print(results_df.head(20).to_string(index=False))

    # SHAP analysis using best params
    if do_shap and len(results_df) > 0:
        print("\n[4/4] Running SHAP analysis with best params...")
        best_params = {
            "n_estimators": int(results_df.iloc[0]["n_estimators"]),
            "max_depth": int(results_df.iloc[0]["max_depth"]),
            "num_leaves": int(results_df.iloc[0]["num_leaves"]),
            "learning_rate": float(results_df.iloc[0]["learning_rate"]),
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }

        model, importance_df = run_with_shap(
            train_df_processed,
            test_df_2021,
            feature_cols,
            target_col="Target",
            best_params=best_params
        )

        if importance_df is not None:
            print("\nTop 10 Most Important Features:")
            for _, row in importance_df.head(10).iterrows():
                print(f"  {row['feature']}: {row['importance']:.4f}")

    print("\n" + "=" * 60)
    print("LGBM Grid Search Complete!")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="JPX Stock Prediction - LGBM Grid Search")
    parser.add_argument("--no-gridsearch", action="store_true", help="Skip grid search")
    parser.add_argument("--no-shap", action="store_true", help="Skip SHAP analysis")

    args = parser.parse_args()

    main(
        do_gridsearch=not args.no_gridsearch,
        do_shap=not args.no_shap
    )
