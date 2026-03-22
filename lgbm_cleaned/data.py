# LGBM Data Loading and Preparation Functions

import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import log, TARGET_HORIZON, TRAIN_DATA_PATH, TEST_DATA_PATH, ROLL_TRAIN_YEARS


def load_cleaned_data():
    """Load cleaned train and test data from CSV files."""
    log("Loading cleaned data...")

    # Load training data
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    log(f"Train data shape: {train_df.shape}")

    # Load test data
    test_df = pd.read_csv(TEST_DATA_PATH)
    log(f"Test data shape: {test_df.shape}")

    return train_df, test_df


def prepare_data(train_df, test_df):
    """Prepare data for LGBM model with quarterly prediction.

    Consistent with lstm_gridsearch_cmd1.py:
    - Convert Quarter to datetime
    - Sort by SecuritiesCode and Quarter
    - Handle missing values: ffill().bfill() then fillna(0)
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

    # Handle missing values - consistent with lstm_gridsearch_cmd1.py
    log("Filling missing values (ffill -> bfill -> fillna(0))...")
    train_df[feature_cols] = train_df.groupby("SecuritiesCode")[feature_cols].transform(lambda x: x.ffill().bfill())
    train_df[feature_cols] = train_df[feature_cols].fillna(0)

    test_df[feature_cols] = test_df.groupby("SecuritiesCode")[feature_cols].transform(lambda x: x.ffill().bfill())
    test_df[feature_cols] = test_df[feature_cols].fillna(0)

    # Filter to 2021 test data only
    test_df_2021 = test_df[test_df["Quarter"].dt.year == 2021].copy()
    log(f"Test data 2021 shape: {test_df_2021.shape}")

    # Filter train data to years before 2021
    train_df_before_2021 = train_df[train_df["Quarter"].dt.year < 2021].copy()
    log(f"Train data before 2021 shape: {train_df_before_2021.shape}")

    return train_df_before_2021, test_df_2021, feature_cols


def create_quarterly_features(df, feature_cols, n_quarters=None):
    """
    Create features using previous n_quarters data to predict next quarter.

    For each stock at quarter t, we use features from quarters t-n_quarters to t-1
    to predict the Target at quarter t.

    Returns a DataFrame with lagged features.
    """
    if n_quarters is None:
        n_quarters = ROLL_TRAIN_YEARS

    log(f"Creating quarterly features with {n_quarters} quarters lag...")

    df = df.sort_values(["SecuritiesCode", "Quarter"]).reset_index(drop=True)

    # Get unique securities
    securities = df["SecuritiesCode"].unique()

    result_frames = []

    for sec in securities:
        sec_data = df[df["SecuritiesCode"] == sec].copy()
        sec_data = sec_data.sort_values("Quarter").reset_index(drop=True)

        # Create lagged features
        for lag in range(1, n_quarters + 1):
            for col in feature_cols:
                sec_data[f"{col}_lag{lag}"] = sec_data[col].shift(lag)

        # Keep only rows where we have enough lag data
        sec_data = sec_data.iloc[n_quarters:]

        if len(sec_data) > 0:
            result_frames.append(sec_data)

    if result_frames:
        result = pd.concat(result_frames, ignore_index=True)
    else:
        result = pd.DataFrame()

    log(f"After creating lagged features: {result.shape}")

    return result


def get_lagged_feature_cols(feature_cols, n_quarters=None):
    """Get the list of lagged feature column names."""
    if n_quarters is None:
        n_quarters = ROLL_TRAIN_YEARS
    lagged_cols = []
    for lag in range(1, n_quarters + 1):
        for col in feature_cols:
            lagged_cols.append(f"{col}_lag{lag}")
    return lagged_cols


def prepare_train_data_for_quarter(train_df, target_quarter, feature_cols, n_quarters=2):
    """
    Prepare training data for predicting a specific target quarter.

    Uses data from quarters before target_quarter to predict target_quarter.
    """
    # Filter to data before the target quarter
    train_data = train_df[train_df["Quarter"] < target_quarter].copy()

    if train_data.empty:
        return pd.DataFrame(), []

    # Create lagged features
    train_data = create_quarterly_features(train_data, feature_cols, n_quarters)

    # Get the target quarter data
    target_data = train_df[train_df["Quarter"] == target_quarter].copy()

    if train_data.empty or target_data.empty:
        return pd.DataFrame(), []

    # Merge to get training samples (stock must have data in all lag quarters)
    # We need securities that appear in both train_data (with lagged features) and target_data
    securities_with_lags = set(train_data["SecuritiesCode"].unique())
    securities_with_target = set(target_data["SecuritiesCode"].unique())
    common_securities = securities_with_lags & securities_with_target

    if not common_securities:
        return pd.DataFrame(), []

    # Filter target data to common securities
    target_data = target_data[target_data["SecuritiesCode"].isin(common_securities)].copy()

    # Create training dataset by matching securities
    # For each security in target quarter, get its lagged features from previous quarters
    train_samples = []

    for sec in common_securities:
        sec_target = target_data[target_data["SecuritiesCode"] == sec]
        sec_lags = train_data[train_data["SecuritiesCode"] == sec]

        if len(sec_lags) > 0 and len(sec_target) > 0:
            # Get the latest lagged features (from n_quarters ago)
            latest_lags = sec_lags.iloc[-1:]

            # Add target
            for _, target_row in sec_target.iterrows():
                sample = latest_lags.copy()
                sample["Target"] = target_row["Target"]
                sample["Target_Quarter"] = target_row["Quarter"]
                train_samples.append(sample)

    if train_samples:
        result = pd.concat(train_samples, ignore_index=True)
    else:
        result = pd.DataFrame()

    return result


def prepare_expanding_window_data(train_df, feature_cols, n_quarters=None):
    """
    Prepare data for expanding window training.

    For each quarter, create training samples using previous quarters' lagged features
    to predict the current quarter's target.
    """
    if n_quarters is None:
        n_quarters = ROLL_TRAIN_YEARS

    log("Preparing expanding window data...")

    # Get all unique quarters sorted
    quarters = sorted(train_df["Quarter"].unique())

    if len(quarters) < n_quarters + 1:
        log(f"Not enough quarters for training (need at least {n_quarters + 1})")
        return pd.DataFrame()

    all_samples = []

    # For each target quarter starting from the (n_quarters+1)th quarter
    for i in range(n_quarters, len(quarters)):
        target_quarter = quarters[i]

        # Get training data up to quarter before target
        train_data = train_df[train_df["Quarter"] < target_quarter].copy()

        if train_data.empty:
            continue

        # Get target quarter data
        target_data = train_df[train_df["Quarter"] == target_quarter].copy()

        if target_data.empty:
            continue

        # Create lagged features for training data
        train_data = create_quarterly_features(train_data, feature_cols, n_quarters)

        if train_data.empty:
            continue

        # Find common securities
        securities_with_lags = set(train_data["SecuritiesCode"].unique())
        securities_with_target = set(target_data["SecuritiesCode"].unique())
        common_securities = securities_with_lags & securities_with_target

        if not common_securities:
            continue

        # Filter target data to common securities
        target_data = target_data[target_data["SecuritiesCode"].isin(common_securities)]

        # Create training samples (skip NaN targets - consistent with lstm1.py)
        for sec in common_securities:
            sec_target = target_data[target_data["SecuritiesCode"] == sec]
            sec_lags = train_data[train_data["SecuritiesCode"] == sec]

            if len(sec_lags) > 0 and len(sec_target) > 0:
                target_val = sec_target["Target"].values[0]
                # Skip if target is NaN (consistent with lstm1.py line 96-97)
                if pd.isna(target_val):
                    continue
                latest_lags = sec_lags.iloc[-1:].copy()
                latest_lags["Target"] = target_val
                latest_lags["Target_Quarter"] = sec_target["Quarter"].values[0]
                all_samples.append(latest_lags)

    if all_samples:
        result = pd.concat(all_samples, ignore_index=True)
    else:
        result = pd.DataFrame()

    log(f"Expanding window data shape: {result.shape}")
    return result


def get_feature_cols(df):
    """Get feature columns from dataframe."""
    non_feature_cols = ["SecuritiesCode", "Quarter", "Target", "Target_Quarter"]
    feature_cols = [c for c in df.columns if c not in non_feature_cols]
    return feature_cols


def get_quarters_in_year(year):
    """Get all quarters for a given year."""
    return [f"{year}-03-31", f"{year}-06-30", f"{year}-09-30", f"{year}-12-31"]
