# LGBM Data Loading and Preparation Functions

import pandas as pd

from .config import log, TRAIN_DATA_PATH, TEST_DATA_PATH


def load_cleaned_data():
    """Load cleaned train and test data from CSV files."""
    log("Loading cleaned data...")

    train_df = pd.read_csv(TRAIN_DATA_PATH)
    log(f"Train data shape: {train_df.shape}")

    test_df = pd.read_csv(TEST_DATA_PATH)
    log(f"Test data shape: {test_df.shape}")

    return train_df, test_df


def prepare_data(train_df, test_df):
    """Prepare data for LGBM model - simplified, no lag features.

    Consistent with lstm1.py:
    - Convert Quarter to datetime
    - Sort by SecuritiesCode and Quarter
    - Handle missing values: ffill().bfill() then fillna(0)
    - Keep all data including 2021
    """
    log("Preparing data...")

    # Convert Quarter to datetime
    train_df["Quarter"] = pd.to_datetime(train_df["Quarter"])
    test_df["Quarter"] = pd.to_datetime(test_df["Quarter"])

    # Sort by SecuritiesCode and Quarter
    train_df = train_df.sort_values(["SecuritiesCode", "Quarter"]).reset_index(drop=True)
    test_df = test_df.sort_values(["SecuritiesCode", "Quarter"]).reset_index(drop=True)

    # Get feature columns (exclude non-feature columns and target)
    non_feature_cols = ["SecuritiesCode", "Quarter", "Target"]
    feature_cols = [c for c in train_df.columns if c not in non_feature_cols]

    log(f"Number of features: {len(feature_cols)}")
    log(f"Features: {feature_cols}")

    # Handle missing values - consistent with lstm1.py
    log("Filling missing values (ffill -> bfill -> fillna(0))...")
    train_df[feature_cols] = train_df.groupby("SecuritiesCode")[feature_cols].transform(lambda x: x.ffill().bfill())
    train_df[feature_cols] = train_df[feature_cols].fillna(0)

    test_df[feature_cols] = test_df.groupby("SecuritiesCode")[feature_cols].transform(lambda x: x.ffill().bfill())
    test_df[feature_cols] = test_df[feature_cols].fillna(0)

    # Keep all data including 2021 (no filtering)
    log(f"Train data shape: {train_df.shape}")
    log(f"Test data shape (2021): {test_df.shape}")

    return train_df, test_df, feature_cols


def prepare_expanding_window_data(train_df, target_col="Target"):
    """Prepare data for expanding window training - simplified, no lag features.

    Each sample = (stock, quarter, raw_features) -> Target
    Each quarter each stock is one independent sample.
    This is equivalent to cross-sectional regression at each quarter.
    """
    log("Preparing expanding window data (no lag features)...")

    train_df = train_df.copy()

    # Skip rows with NaN target (consistent with lstm1.py line 96-97)
    train_df = train_df[train_df[target_col].notna()].copy()

    # Add LabelYear and LabelQuarter for filtering
    train_df["LabelYear"] = train_df["Quarter"].dt.year
    train_df["LabelQuarter"] = train_df["Quarter"].dt.year.astype(str) + "Q" + \
                               ((train_df["Quarter"].dt.month - 1) // 3 + 1).astype(str)

    log(f"Expanding window data shape: {train_df.shape}")
    log(f"LabelYear distribution:\n{train_df['LabelYear'].value_counts().sort_index()}")

    return train_df


def get_feature_cols(df):
    """Get feature columns from dataframe."""
    non_feature_cols = ["SecuritiesCode", "Quarter", "Target", "Target_Quarter",
                       "LabelYear", "LabelQuarter"]
    feature_cols = [c for c in df.columns if c not in non_feature_cols]
    return feature_cols


def get_quarters_in_year(year):
    """Get all quarters for a given year."""
    return [f"{year}-03-31", f"{year}-06-30", f"{year}-09-30", f"{year}-12-31"]
