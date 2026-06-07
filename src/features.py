"""
features.py
-----------
Feature engineering pipeline for energy demand forecasting.
Generates lag features, rolling statistics, calendar features,
and Fourier terms for seasonality encoding.
"""

import logging
import numpy as np
import pandas as pd
import holidays as hd
from pathlib import Path

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DATA_PROCESSED = ROOT / "data" / "processed"


# ─── Calendar features ────────────────────────────────────────────────────────
def add_calendar_features(df: pd.DataFrame, country: str = "DE") -> pd.DataFrame:
    """
    Add hour-of-day, day-of-week, month, quarter, weekend flag,
    and public holiday flag.
    """
    df = df.copy()
    idx = df.index

    df["hour"]          = idx.hour
    df["day_of_week"]   = idx.dayofweek          # 0=Mon, 6=Sun
    df["day_of_year"]   = idx.dayofyear
    df["month"]         = idx.month
    df["quarter"]       = idx.quarter
    df["week_of_year"]  = idx.isocalendar().week.astype(int)
    df["is_weekend"]    = (idx.dayofweek >= 5).astype(int)

    # Public holidays
    country_holidays = hd.country_holidays(country)
    df["is_holiday"] = idx.normalize().map(
        lambda d: int(d.date() in country_holidays)
    )
    # Day before / after holiday (demand often drops)
    df["is_holiday_eve"] = df["is_holiday"].shift(-24, fill_value=0)

    logger.info("Calendar features added.")
    return df


# ─── Fourier features (encode seasonality without OHE) ───────────────────────
def add_fourier_features(
    df: pd.DataFrame,
    period_hours: int = 8760,
    n_terms: int = 3,
    prefix: str = "annual",
) -> pd.DataFrame:
    """
    Add sin/cos Fourier pairs to capture smooth cyclic patterns.
    Default: annual seasonality (8760 hours).
    Also adds daily (24h) and weekly (168h) components.
    """
    df = df.copy()
    t = np.arange(len(df))

    for period, name in [(24, "daily"), (168, "weekly"), (8760, "annual")]:
        for k in range(1, n_terms + 1):
            df[f"sin_{name}_{k}"] = np.sin(2 * np.pi * k * t / period)
            df[f"cos_{name}_{k}"] = np.cos(2 * np.pi * k * t / period)

    logger.info(f"Fourier features added (daily + weekly + annual, {n_terms} terms each).")
    return df


# ─── Lag features ─────────────────────────────────────────────────────────────
def add_lag_features(
    df: pd.DataFrame,
    target_col: str = "load_mw",
    lags: list[int] | None = None,
) -> pd.DataFrame:
    """
    Lag the target at key offsets:
    - t-1h, t-2h, t-3h (short-term autocorrelation)
    - t-24h (same hour yesterday)
    - t-48h (day before yesterday)
    - t-168h (same hour last week)
    - t-336h (two weeks ago)
    """
    if lags is None:
        lags = [1, 2, 3, 24, 48, 168, 336]

    df = df.copy()
    for lag in lags:
        df[f"lag_{lag}h"] = df[target_col].shift(lag)

    logger.info(f"Lag features added: {lags}")
    return df


# ─── Rolling statistics ───────────────────────────────────────────────────────
def add_rolling_features(
    df: pd.DataFrame,
    target_col: str = "load_mw",
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """
    Rolling mean and std over 24h, 168h (week) windows.
    Uses shift(1) to avoid leaking future data.
    """
    if windows is None:
        windows = [24, 168]

    df = df.copy()
    for w in windows:
        shifted = df[target_col].shift(1)
        df[f"roll_mean_{w}h"] = shifted.rolling(window=w, min_periods=1).mean()
        df[f"roll_std_{w}h"]  = shifted.rolling(window=w, min_periods=1).std()
        df[f"roll_max_{w}h"]  = shifted.rolling(window=w, min_periods=1).max()
        df[f"roll_min_{w}h"]  = shifted.rolling(window=w, min_periods=1).min()

    logger.info(f"Rolling features added: windows={windows}")
    return df


# ─── Decomposition features ───────────────────────────────────────────────────
def add_decomposition_features(
    df: pd.DataFrame,
    target_col: str = "load_mw",
    period: int = 8760,
) -> pd.DataFrame:
    """
    Approximate trend using a centred 4-week rolling mean.
    Residual = actual - trend (captures irregular component).
    """
    df = df.copy()
    df["trend"] = (
        df[target_col]
        .rolling(window=672, center=True, min_periods=1)
        .mean()
    )
    df["detrended"] = df[target_col] - df["trend"]
    logger.info("Decomposition features (trend, detrended) added.")
    return df


# ─── Temperature interaction ──────────────────────────────────────────────────
def add_temperature_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Heating/cooling degree hours and temperature-hour interaction.
    Only applied if 'temperature_c' column exists.
    """
    if "temperature_c" not in df.columns:
        logger.warning("'temperature_c' not found — skipping temperature features.")
        return df

    df = df.copy()
    base = 18.0  # degree-day base temperature
    df["hdh"] = np.maximum(base - df["temperature_c"], 0)   # heating degree hours
    df["cdh"] = np.maximum(df["temperature_c"] - base, 0)   # cooling degree hours
    df["temp_hour_interaction"] = df["temperature_c"] * df["hour"]
    logger.info("Temperature interaction features added.")
    return df


# ─── Full pipeline ────────────────────────────────────────────────────────────
def build_feature_matrix(
    df: pd.DataFrame,
    target_col: str = "load_mw",
    drop_na: bool = True,
) -> pd.DataFrame:
    """
    Run the complete feature engineering pipeline.
    Returns a DataFrame with all features + target.
    """
    df = add_calendar_features(df)
    df = add_fourier_features(df)
    df = add_lag_features(df, target_col=target_col)
    df = add_rolling_features(df, target_col=target_col)
    df = add_decomposition_features(df, target_col=target_col)
    df = add_temperature_features(df)

    if drop_na:
        before = len(df)
        df = df.dropna()
        logger.info(f"Dropped {before - len(df)} rows with NaN after feature engineering.")

    out = DATA_PROCESSED / "features.csv"
    df.to_csv(out)
    logger.info(f"Feature matrix saved → {out}  shape={df.shape}")
    return df


def get_feature_columns(df: pd.DataFrame, target_col: str = "load_mw") -> list[str]:
    """Return list of feature column names (excludes target)."""
    exclude = {target_col, "trend", "detrended"}
    return [c for c in df.columns if c not in exclude]


if __name__ == "__main__":
    from src.data_loader import load_processed
    df = load_processed()
    features = build_feature_matrix(df)
    print(f"\nFeature matrix shape: {features.shape}")
    print(f"\nColumns:\n{features.columns.tolist()}")
