# LightGBM Model Definition (Placeholder)
# LightGBM uses the LGBMRegressor from lightgbm package
# No custom model class needed

from .data import fit_lgbm, predict_timeseries_lgbm

__all__ = ["fit_lgbm", "predict_timeseries_lgbm"]
