# AGENTS.md - JPX Tokyo Stock Exchange Prediction Project

## Project Overview

This is a machine learning project for predicting Japanese stock returns using JPX Tokyo Stock Exchange data. The project uses LightGBM as the primary model with walk-forward training and various feature engineering from stock prices, options, financials, and trader data.

## Directory Structure

```
jpx-tokyo-stock-exchange-prediction/
├── train_files/                    # Training data (2017-2021)
│   ├── stock_prices.csv           # Main stock price data
│   ├── secondary_stock_prices.csv #创业板 stock prices
│   ├── options.csv                # Options data
│   ├── trades.csv                 # Investor trading data
│   └── financials.csv             # Company financial data
├── data_specifications/           # Data field specifications
├── example_test_files/            # Test data examples
├── supplemental_files/             # Supplementary data
├── output_v2/                      # Main experiment results
├── output_horizon_compare/        # Horizon comparison results
├── demo_v2.py                     # Core feature engineering & training
├── horizon_model_comparison.py   # Multi-model/horizon experiments
└── lgbm_20d_all.py               # Best performing model script
```

## Build/Lint/Test Commands

Since this project does not have a formal test framework or linting setup, use the following commands for development:

### Running Main Experiments

```bash
# Run horizon comparison experiments (all models x all horizons)
python horizon_model_comparison.py

# Run best performing model (LightGBM 20d with all data sources)
python lgbm_20d_all.py

# Run main demo (feature engineering + portfolio optimization)
python demo_v2.py
```

### Single Test / Development Run

To test a single component, you can import and run specific functions:

```python
# Example: Test data loading
import demo_v2 as base
sources = base.load_data_sources("train_files")
print(f"Loaded {len(sources['stock_prices'])} stock price rows")

# Example: Test feature engineering
df, cols = base.stock_features(sources['stock_prices'])
print(f"Created {len(cols)} stock features")
```

### Virtual Environment

```bash
# Activate virtual environment
source .venv/bin/activate

# Install dependencies (if needed)
pip install -r requirements.txt  # Check if exists first
```

## Code Style Guidelines

### General Principles

- Write clean, readable code with meaningful variable names
- Keep functions focused on a single responsibility
- Add docstrings to public functions explaining purpose, args, and returns
- Use comments sparingly - code should be self-explanatory

### Naming Conventions

```python
# Functions and variables: snake_case
def load_data_sources(data_dir):
    stock_features = []
    target_horizon = 20

# Constants: UPPER_CASE with underscore
OUTPUT_DIR = "output_v2"
TEST_YEAR = 2021
ROLL_TRAIN_YEARS = 2

# Classes: PascalCase (if used)
class PortfolioOptimizer:
    pass
```

### Import Organization

Order imports by category with blank lines between:

```python
# 1. Standard library
import os
import gc
import warnings
from datetime import datetime

# 2. Third-party libraries
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
import cvxpy as cp

# 3. Local imports
import demo_v2 as base
```

### Type Hints

Add type hints for function signatures when beneficial:

```python
def load_data_sources(data_dir: str) -> dict:
    """Load all data sources from directory."""
    ...

def build_feature_table(
    sources: dict,
    start_date: str = None,
    end_date: str = None
) -> tuple[pd.DataFrame, dict]:
    ...
```

### Error Handling

- Use `warnings.filterwarnings("ignore")` sparingly - prefer explicit handling
- Use `pd.to_numeric(..., errors="coerce")` for flexible numeric parsing
- Validate critical inputs at function entry points
- Handle missing data explicitly with `.fillna()` or `.dropna()`

### Data Processing Patterns

```python
# Convert columns to numeric, handling errors gracefully
def to_num(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

# Use groupby for per-stock calculations
g = df.groupby("SecuritiesCode", sort=False)
df["return_1d"] = g["Close"].pct_change(1)

# Sort data before groupby operations
df = df.sort_values(["SecuritiesCode", "Date"])
```

### Logging

Use the project's logging pattern:

```python
def log(msg):
    print(f"[INFO] {msg}")

log("Loading data...")
log(f"Created {len(features)} features")
```

### Matplotlib/Visualization

Set backend before importing plt:

```python
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
```

### Configuration

Use module-level constants for configuration at the top of files:

```python
# Constants
OUTPUT_DIR = "output_v2"
TEST_YEAR = 2021
VALID_YEAR = 2020
RANDOM_STATE = 21

# Model hyperparameters
ROLL_TRAIN_YEARS = 2
ROLL_RETRAIN_FREQ = "M"  # Monthly
TARGET_HORIZON = 20
```

### File Paths with os.path.join()

Use for path construction:

```python
import os
OUTPUT_DIR = "output_v2"
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
```

### Memory Management

- Delete large objects and call `gc.collect()` after processing large dataframes
- Use appropriate dtypes (int32 instead of int64 where possible)
- Process data in chunks if memory-constrained

### Testing Custom Code

Since there's no formal test framework:

1. Create a small test script to verify your function works correctly
2. Test with a subset of data before running full experiments
3. Verify output shapes and value ranges match expectations
4. Compare against known good results from demo_v2.py

### Common Gotchas

- Data leakage: Ensure training data does not include future information
- Date handling: Use `parse_dates=["Date"]` when reading CSVs
- GroupBy performance: Use `sort=False` when data is already sorted
- NaN handling: Check for NaN values after pct_change and division operations
