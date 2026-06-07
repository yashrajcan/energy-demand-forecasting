"""
Energy Demand Forecasting — source package.
"""
from .data_loader import load_processed, train_val_test_split, build_dataset
from .features import build_feature_matrix, get_feature_columns
from .models import (
    SARIMAForecaster,
    ProphetForecaster,
    XGBoostForecaster,
    LSTMForecaster,
    EnsembleForecaster,
)
from .evaluate import (
    rmse, mae, mape, smape,
    evaluate_all, comparison_table, full_evaluation,
    plot_forecast, plot_residuals, plot_metric_comparison,
    plot_shap_summary, plot_decomposition,
)

__all__ = [
    "load_processed", "train_val_test_split", "build_dataset",
    "build_feature_matrix", "get_feature_columns",
    "SARIMAForecaster", "ProphetForecaster", "XGBoostForecaster",
    "LSTMForecaster", "EnsembleForecaster",
    "rmse", "mae", "mape", "smape",
    "evaluate_all", "comparison_table", "full_evaluation",
    "plot_forecast", "plot_residuals", "plot_metric_comparison",
    "plot_shap_summary", "plot_decomposition",
]
