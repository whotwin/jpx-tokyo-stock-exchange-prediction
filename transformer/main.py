# Transformer Main Entry Point

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformer.config import OUTPUT_DIR
from transformer.data import load_all_data
from transformer.predict import predict_with_transformer


def main():
    """Main function to run Transformer model."""
    print("=" * 60)
    print("JPX Stock Prediction - Transformer Model")
    print("=" * 60)

    # Load data
    print("\n[1/3] Loading data...")
    full_df, feature_cols = load_all_data()

    # Target column
    target_col = "target_30d"

    # Run prediction with expanding window
    print("\n[2/3] Running Transformer prediction...")
    result = predict_with_transformer(full_df, feature_cols, target_col)

    if result.empty:
        print("No predictions generated!")
        return

    # Save predictions
    print("\n[3/3] Saving results...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pred_file = os.path.join(OUTPUT_DIR, "transformer_predictions.csv")
    result.to_csv(pred_file, index=False)
    print(f"Predictions saved to: {pred_file}")

    print("\n" + "=" * 60)
    print("Transformer Model Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
