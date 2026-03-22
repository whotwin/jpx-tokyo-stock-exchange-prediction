# iTransformer Model Definition
# Most functions are imported from the original itransformer_model.py

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import training functions from original file
from itransformer.itransformer_model import (
    train_itransformer_model,
    predict_itransformer,
)

__all__ = [
    "train_itransformer_model",
    "predict_itransformer",
]
