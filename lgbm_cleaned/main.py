# LGBM Main Entry Point

import os
import sys

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from lgbm_cleaned.data import load_cleaned_data, prepare_data
    from lgbm_cleaned.predict import predict_with_lgbm_gridsearch, predict_2021_final, run_with_shap, OUTPUT_DIR
else:
    from .data import load_cleaned_data, prepare_data
    from .predict import predict_with_lgbm_gridsearch, predict_2021_final, run_with_shap, OUTPUT_DIR


def main(do_gridsearch=True, do_shap=True, do_final_test=True):
    """Main function to run LGBM model.

    Consistent with lstm1.py and lstm_gridsearch_cmd1.py:
    - Grid search with 3-fold expanding window CV
    - Final evaluation on 2021 data
    """
    print("=" * 60)
    print("JPX Stock Prediction - LGBM Grid Search + 2021 Final Test")
    print("=" * 60)

    # Load data
    print("\n[1/5] Loading data...")
    train_df, test_df = load_cleaned_data()

    # Prepare data
    print("\n[2/5] Preparing data...")
    train_df_processed, test_df_2021, feature_cols = prepare_data(train_df, test_df)

    print(f"Feature columns ({len(feature_cols)})")

    # Run grid search
    if do_gridsearch:
        print("\n[3/5] Running LGBM grid search...")
        results_df, quarter_results_df = predict_with_lgbm_gridsearch(
            train_df_processed,
            feature_cols,
            target_col="Target",
        )

        if results_df is None or results_df.empty:
            print("Grid search failed!")
            return

        print("\n" + "=" * 60)
        print("Grid Search Results Summary:")
        print("=" * 60)
        print(results_df.head(20).to_string(index=False))

        # Get best params
        best_params = {
            "n_estimators": int(results_df.iloc[0]["n_estimators"]),
            "max_depth": int(results_df.iloc[0]["max_depth"]),
            "num_leaves": int(results_df.iloc[0]["num_leaves"]),
            "learning_rate": float(results_df.iloc[0]["learning_rate"]),
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }
    else:
        results_df = None
        best_params = None

    # Final 2021 test
    if do_final_test:
        print("\n[4/5] Running final 2021 evaluation...")
        ranking_detail, rankic_df = predict_2021_final(
            train_df_processed,
            test_df_2021,
            target_col="Target",
            best_params=best_params,
        )

        if rankic_df is not None and not rankic_df.empty:
            print("\n2021 Quarterly RankIC:")
            print(rankic_df.to_string(index=False))

        # Final summary output matching lstm1.py format
        import pandas as pd

        # Show sample ranking detail with signal columns
        detail_path = os.path.join(OUTPUT_DIR, "test_2021_ranking_detail.csv")
        if os.path.exists(detail_path):
            detail = pd.read_csv(detail_path)
            detail["LabelYear"] = pd.to_datetime(detail["Quarter"]).dt.year
            detail["signal"] = detail["pred"]

            # Calculate final comparison metrics
            from scipy.stats import spearmanr
            overall_ic = spearmanr(detail["pred"], detail["y_true"])[0]
            avg_q_ic = rankic_df["RankIC"].mean()
            avg_ls = rankic_df["LongShortSpread"].mean()

            print("\n================ FINAL COMPARISON ================")
            print(f"signal      overall_rankic  avg_quarterly_rankic  avg_long_short")
            print(f"signal_pos  {overall_ic:>14.6f}  {avg_q_ic:>20.6f}  {avg_ls:>16.6f}")
            print(f"\nPrediction finished.")
            print("Training finished.")

            # Print top20 and bottom20 for each quarter
            for quarter in sorted(detail["Quarter"].unique()):
                q_data = detail[detail["Quarter"] == quarter].sort_values("pred", ascending=False)
                print(f"\n--- {quarter} ---")
                print("Top20:")
                top20 = q_data.head(20)[["SecuritiesCode", "pred", "y_true"]]
                print(top20.to_string(index=False))
                print("Bottom20:")
                bot20 = q_data.tail(20)[["SecuritiesCode", "pred", "y_true"]]
                print(bot20.to_string(index=False))

    # SHAP analysis
    if do_shap and best_params is not None:
        print("\n[5/5] Running SHAP analysis with best params...")
        _, importance_df = run_with_shap(
            train_df_processed,
            target_col="Target",
            best_params=best_params,
        )

        if importance_df is not None:
            print("\nTop 10 Most Important Features:")
            for _, row in importance_df.head(10).iterrows():
                print(f"  {row['feature']}: {row['importance']:.4f}")

    print("\n" + "=" * 60)
    print("LGBM Complete!")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="JPX Stock Prediction - LGBM")
    parser.add_argument("--no-gridsearch", action="store_true", help="Skip grid search")
    parser.add_argument("--no-shap", action="store_true", help="Skip SHAP analysis")
    parser.add_argument("--no-final-test", action="store_true", help="Skip 2021 final test")

    args = parser.parse_args()

    main(
        do_gridsearch=not args.no_gridsearch,
        do_shap=not args.no_shap,
        do_final_test=not args.no_final_test,
    )
