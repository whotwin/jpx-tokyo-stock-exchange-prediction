# LSTM Main Entry Point

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lstm.config import OUTPUT_DIR, LSTM_FEATURES
from lstm.data import load_all_data_for_lstm
from lstm.predict import predict_with_lstm


def main():
    """Main function to run LSTM model."""
    print("=" * 60)
    print("JPX Stock Prediction - LSTM Model")
    print("=" * 60)

    # Load data
    print("\n[1/3] Loading data...")
    full_df, feature_cols = load_all_data_for_lstm()

    # Get available features
    available_features = [f for f in LSTM_FEATURES if f in full_df.columns]
    missing_features = [f for f in LSTM_FEATURES if f not in full_df.columns]

    if missing_features:
        print(f"Warning: {len(missing_features)} features not found: {missing_features[:5]}...")

    print(f"Using {len(available_features)} features out of {len(LSTM_FEATURES)} requested")

    # Target column
    target_col = "target_30d"

    # Run prediction with expanding window
    print("\n[2/3] Running LSTM prediction...")
    result = predict_with_lstm(full_df, available_features, target_col)

    if result.empty:
        print("No predictions generated!")
        return

    # Save predictions
    print("\n[3/3] Saving results...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pred_file = os.path.join(OUTPUT_DIR, "lstm_predictions.csv")
    result.to_csv(pred_file, index=False)
    print(f"Predictions saved to: {pred_file}")

    print("\n" + "=" * 60)
    print("LSTM Model Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
