# iTransformer Main Entry Point

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from itransformer.config import OUTPUT_DIR
from itransformer.data import load_all_data
from itransformer.predict import predict_with_itransformer


def main():
    """Main function to run iTransformer model."""
    print("=" * 60)
    print("JPX Stock Prediction - iTransformer Model")
    print("=" * 60)

    # Load data
    print("\n[1/3] Loading data...")
    full_df, feature_cols = load_all_data()

    # Get stock codes (top 2000 by market cap for efficiency)
    stock_codes = sorted(full_df["SecuritiesCode"].unique())[:2000]
    print(f"Using {len(stock_codes)} stocks")

    # Run prediction with expanding window
    print("\n[2/3] Running iTransformer prediction...")
    result = predict_with_itransformer(full_df, feature_cols, stock_codes)

    if result is None or result.empty:
        print("No predictions generated!")
        return

    # Save predictions
    print("\n[3/3] Saving results...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pred_file = os.path.join(OUTPUT_DIR, "predictions.csv")
    result.to_csv(pred_file, index=False)
    print(f"Predictions saved to: {pred_file}")

    print("\n" + "=" * 60)
    print("iTransformer Model Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
