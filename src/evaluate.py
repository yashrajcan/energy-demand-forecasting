"""
evaluate.py
-----------
Evaluation metrics, residual diagnostics, and visualisation utilities
for comparing time series forecasting models.
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

ROOT    = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

# Consistent style
plt.rcParams.update({
    "figure.dpi":       150,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "font.family":      "sans-serif",
})
PALETTE = ["#2563EB", "#16A34A", "#D97706", "#DC2626", "#7C3AED"]


# ─── Metrics ──────────────────────────────────────────────────────────────────
def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - predicted)))


def mape(actual: np.ndarray, predicted: np.ndarray, eps: float = 1e-8) -> float:
    return float(np.mean(np.abs((actual - predicted) / (np.abs(actual) + eps))) * 100)


def smape(actual: np.ndarray, predicted: np.ndarray, eps: float = 1e-8) -> float:
    num = np.abs(actual - predicted)
    den = (np.abs(actual) + np.abs(predicted)) / 2 + eps
    return float(np.mean(num / den) * 100)


def evaluate_all(actual: np.ndarray, predicted: np.ndarray, model_name: str = "") -> dict:
    """Return a dict of all metrics for a single model."""
    return {
        "model":  model_name,
        "RMSE":   round(rmse(actual, predicted), 2),
        "MAE":    round(mae(actual, predicted), 2),
        "MAPE":   round(mape(actual, predicted), 4),
        "sMAPE":  round(smape(actual, predicted), 4),
    }


def comparison_table(results: list[dict]) -> pd.DataFrame:
    """Build a sorted comparison DataFrame from a list of evaluate_all dicts."""
    df = pd.DataFrame(results).set_index("model").sort_values("RMSE")
    return df


# ─── Residual diagnostics ─────────────────────────────────────────────────────
def plot_residuals(
    actual: np.ndarray,
    predicted: np.ndarray,
    model_name: str = "Model",
    save: bool = True,
) -> plt.Figure:
    """
    4-panel residual diagnostic plot:
    1. Residuals over time
    2. Histogram of residuals
    3. Q-Q plot (normality check)
    4. ACF of residuals (autocorrelation check)
    """
    from statsmodels.graphics.tsaplots import plot_acf
    from scipy import stats

    residuals = actual - predicted
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"{model_name} — Residual Diagnostics", fontsize=14, y=1.01)

    # 1. Residuals over time
    ax = axes[0, 0]
    ax.plot(residuals, color=PALETTE[0], linewidth=0.4, alpha=0.7)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_title("Residuals over time")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Residual (MW)")

    # 2. Histogram
    ax = axes[0, 1]
    ax.hist(residuals, bins=50, color=PALETTE[0], alpha=0.8, edgecolor="white")
    ax.set_title("Residual distribution")
    ax.set_xlabel("Residual (MW)")

    # 3. Q-Q plot
    ax = axes[1, 0]
    (osm, osr), (slope, intercept, r) = stats.probplot(residuals, dist="norm")
    ax.scatter(osm, osr, s=4, color=PALETTE[0], alpha=0.5)
    line_x = np.array([osm[0], osm[-1]])
    ax.plot(line_x, slope * line_x + intercept, color=PALETTE[3], linewidth=1.5)
    ax.set_title(f"Q-Q plot  (R²={r**2:.3f})")
    ax.set_xlabel("Theoretical quantiles")
    ax.set_ylabel("Sample quantiles")

    # 4. ACF of residuals
    ax = axes[1, 1]
    plot_acf(residuals, lags=48, ax=ax, color=PALETTE[0], zero=False)
    ax.set_title("ACF of residuals (lags up to 48h)")

    plt.tight_layout()
    if save:
        path = FIG_DIR / f"residuals_{model_name.lower().replace(' ', '_')}.png"
        fig.savefig(path, bbox_inches="tight")
        logger.info(f"Residual plot saved → {path}")
    return fig


# ─── Forecast vs actual plot ──────────────────────────────────────────────────
def plot_forecast(
    actual: pd.Series,
    forecasts: dict[str, np.ndarray],
    title: str = "Forecast comparison",
    save: bool = True,
    zoom_days: int = 14,
) -> plt.Figure:
    """
    Plot actual vs. forecasts for all models over the first `zoom_days` of test period.
    """
    fig, ax = plt.subplots(figsize=(14, 5))
    n = min(zoom_days * 24, len(actual))
    idx = actual.index[:n]

    ax.plot(idx, actual.values[:n], color="black", linewidth=1.2,
            label="Actual", zorder=5)
    for i, (name, pred) in enumerate(forecasts.items()):
        ax.plot(idx, pred[:n], linewidth=1, alpha=0.85,
                color=PALETTE[i % len(PALETTE)], label=name)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax.set_title(title)
    ax.set_ylabel("Load (MW)")
    ax.legend(loc="upper right", framealpha=0.9)

    plt.tight_layout()
    if save:
        path = FIG_DIR / "forecast_comparison.png"
        fig.savefig(path, bbox_inches="tight")
        logger.info(f"Forecast plot saved → {path}")
    return fig


# ─── Model comparison bar chart ───────────────────────────────────────────────
def plot_metric_comparison(
    results: list[dict],
    metric: str = "RMSE",
    save: bool = True,
) -> plt.Figure:
    df = pd.DataFrame(results).sort_values(metric)
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(df["model"], df[metric], color=PALETTE[: len(df)], alpha=0.85)
    ax.bar_label(bars, fmt="%.0f", padding=4, fontsize=10)
    ax.set_xlabel(metric)
    ax.set_title(f"Model comparison — {metric}")
    ax.invert_yaxis()

    plt.tight_layout()
    if save:
        path = FIG_DIR / f"comparison_{metric.lower()}.png"
        fig.savefig(path, bbox_inches="tight")
        logger.info(f"Metric comparison plot saved → {path}")
    return fig


# ─── SHAP summary plot ────────────────────────────────────────────────────────
def plot_shap_summary(shap_values, feature_names: list[str], save: bool = True):
    """
    Beeswarm SHAP summary plot for the XGBoost model.
    """
    import shap
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, feature_names=feature_names,
                      show=False, plot_size=None)
    plt.title("SHAP feature importance — XGBoost")
    plt.tight_layout()
    if save:
        path = FIG_DIR / "shap_summary.png"
        fig.savefig(path, bbox_inches="tight")
        logger.info(f"SHAP plot saved → {path}")
    return fig


# ─── Decomposition plot ───────────────────────────────────────────────────────
def plot_decomposition(series: pd.Series, model: str = "additive", save: bool = True):
    """Seasonal decomposition plot using statsmodels."""
    from statsmodels.tsa.seasonal import seasonal_decompose
    result = seasonal_decompose(series.dropna(), model=model, period=8760)

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    titles  = ["Observed", "Trend", "Seasonal", "Residual"]
    comps   = [result.observed, result.trend, result.seasonal, result.resid]
    colors  = [PALETTE[0], PALETTE[1], PALETTE[2], "gray"]

    for ax, title, comp, col in zip(axes, titles, comps, colors):
        ax.plot(comp, color=col, linewidth=0.5)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel("MW")

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    plt.suptitle("Energy demand — seasonal decomposition", fontsize=14)
    plt.tight_layout()

    if save:
        path = FIG_DIR / "decomposition.png"
        fig.savefig(path, bbox_inches="tight")
        logger.info(f"Decomposition plot saved → {path}")
    return fig


# ─── Convenience: run full evaluation ─────────────────────────────────────────
def full_evaluation(
    actual: pd.Series,
    forecasts: dict[str, np.ndarray],
) -> pd.DataFrame:
    """
    Compute metrics for all models and print a formatted table.
    """
    results = [
        evaluate_all(actual.values[:len(pred)], pred, model_name=name)
        for name, pred in forecasts.items()
    ]
    table = comparison_table(results)
    print("\n" + "=" * 55)
    print("MODEL COMPARISON (sorted by RMSE)")
    print("=" * 55)
    print(table.to_string())
    print("=" * 55 + "\n")
    return table
