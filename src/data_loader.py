"""
data_loader.py
--------------
Downloads the Open Power System Data (OPSD) hourly time series for Germany,
cleans it, and optionally merges temperature data from Open-Meteo API.

Usage:
    python src/data_loader.py
"""

import os
import time
import logging
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from functools import wraps

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"

DATA_RAW.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

# ─── Source URLs ──────────────────────────────────────────────────────────────
OPSD_URL = (
    "https://data.open-power-system-data.org/time_series/2020-10-06/"
    "time_series_60min_singleindex.csv"
)


# ─── Decorators ───────────────────────────────────────────────────────────────
def timer(func):
    """Log execution time of any function."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logger.info(f"{func.__name__} completed in {elapsed:.2f}s")
        return result
    return wrapper


def cache_to_disk(filepath: Path):
    """Skip download if file already exists on disk."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if filepath.exists():
                logger.info(f"Cache hit → loading from {filepath.name}")
                return pd.read_csv(filepath, index_col=0, parse_dates=True)
            result = func(*args, **kwargs)
            result.to_csv(filepath)
            logger.info(f"Saved to {filepath}")
            return result
        return wrapper
    return decorator


# ─── Download ─────────────────────────────────────────────────────────────────
@timer
@cache_to_disk(DATA_RAW / "opsd_raw.csv")
def download_opsd() -> pd.DataFrame:
    """Download OPSD hourly electricity load data for Germany."""
    logger.info("Downloading OPSD time series (~150 MB, please wait)…")
    response = requests.get(OPSD_URL, stream=True, timeout=120)
    response.raise_for_status()

    raw_path = DATA_RAW / "opsd_raw.csv"
    with open(raw_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    df = pd.read_csv(raw_path, index_col=0, parse_dates=True, low_memory=False)
    logger.info(f"Downloaded {len(df):,} rows × {df.shape[1]} columns")
    return df


# ─── Extract Germany load ─────────────────────────────────────────────────────
@timer
def extract_germany_load(df: pd.DataFrame) -> pd.Series:
    """Pull the DE_load_actual_entsoe_transparency column and clean it."""
    col = "DE_load_actual_entsoe_transparency"
    if col not in df.columns:
        # Fallback column name used in older OPSD versions
        col = "DE_load_actual_entsoe_power_statistics"

    series = df[col].copy()
    series.index = pd.to_datetime(series.index, utc=True).tz_localize(None)
    series = series.sort_index()

    # Keep 2006–2017 for a clean decade
    series = series["2006":"2017"]
    logger.info(f"Germany load extracted: {series.index[0]} → {series.index[-1]}, "
                f"{series.isna().sum()} NaNs")
    return series


# ─── Clean ────────────────────────────────────────────────────────────────────
@timer
def clean_load(series: pd.Series) -> pd.Series:
    """
    1. Fill short gaps (≤ 3h) by linear interpolation.
    2. Cap obvious outliers outside [μ ± 4σ] via rolling statistics.
    3. Resample to ensure a complete hourly DatetimeIndex.
    """
    # Complete hourly index
    full_idx = pd.date_range(series.index[0], series.index[-1], freq="h")
    series = series.reindex(full_idx)

    # Short-gap interpolation
    series = series.interpolate(method="time", limit=3)

    # Outlier capping using 7-day rolling window
    rolling_mean = series.rolling(window=168, center=True, min_periods=24).mean()
    rolling_std  = series.rolling(window=168, center=True, min_periods=24).std()
    lower = rolling_mean - 4 * rolling_std
    upper = rolling_mean + 4 * rolling_std
    outlier_mask = (series < lower) | (series > upper)
    series[outlier_mask] = np.nan
    series = series.interpolate(method="time", limit=6)

    # Any remaining NaN → forward-fill (last resort)
    series = series.ffill().bfill()

    logger.info(f"Cleaned series: {series.isna().sum()} NaNs remaining, "
                f"shape={series.shape}")
    return series


# ─── Fetch weather (Open-Meteo, free, no key) ─────────────────────────────────
@timer
@cache_to_disk(DATA_RAW / "weather_berlin.csv")
def fetch_weather(start: str = "2006-01-01", end: str = "2017-12-31") -> pd.DataFrame:
    """
    Pull hourly temperature and cloud-cover for Berlin from Open-Meteo
    historical archive (free, no API key required).
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": 52.52,
        "longitude": 13.41,
        "start_date": start,
        "end_date": end,
        "hourly": "temperature_2m,cloudcover,windspeed_10m",
        "timezone": "Europe/Berlin",
    }
    logger.info("Fetching weather data from Open-Meteo…")
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()["hourly"]
    df = pd.DataFrame(data)
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time")
    df.columns = ["temperature_c", "cloudcover_pct", "windspeed_kmh"]
    return df


# ─── Merge & save ─────────────────────────────────────────────────────────────
@timer
def build_dataset() -> pd.DataFrame:
    """Full pipeline: download → clean → merge weather → save."""
    raw = download_opsd()
    load = extract_germany_load(raw)
    load = clean_load(load)

    try:
        weather = fetch_weather()
        weather.index = weather.index.tz_localize(None)
        df = pd.DataFrame({"load_mw": load}).join(weather, how="left")
        df[["temperature_c", "cloudcover_pct", "windspeed_kmh"]] = (
            df[["temperature_c", "cloudcover_pct", "windspeed_kmh"]]
            .interpolate(method="time", limit=2)
            .ffill()
            .bfill()
        )
    except Exception as e:
        logger.warning(f"Weather fetch failed ({e}). Proceeding without weather features.")
        df = pd.DataFrame({"load_mw": load})

    out = DATA_PROCESSED / "energy_dataset.csv"
    df.to_csv(out)
    logger.info(f"Final dataset saved → {out}  shape={df.shape}")
    return df


# ─── Loaders for notebooks ────────────────────────────────────────────────────
def load_processed() -> pd.DataFrame:
    """Load the processed dataset (run build_dataset() first)."""
    path = DATA_PROCESSED / "energy_dataset.csv"
    if not path.exists():
        raise FileNotFoundError(
            "Processed dataset not found. Run `python src/data_loader.py` first."
        )
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index.freq = pd.infer_freq(df.index)
    return df


def train_val_test_split(
    df: pd.DataFrame,
    val_end: str = "2016-12-31",
    test_start: str = "2017-01-01",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Time-aware split — no data leakage.
    Train: 2006–2015  |  Val: 2016  |  Test: 2017
    """
    train = df[: "2015-12-31"]
    val   = df["2016-01-01" : val_end]
    test  = df[test_start:]
    logger.info(
        f"Split → Train:{len(train):,}  Val:{len(val):,}  Test:{len(test):,}"
    )
    return train, val, test


if __name__ == "__main__":
    build_dataset()
