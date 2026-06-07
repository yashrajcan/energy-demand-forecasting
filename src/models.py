"""
models.py
---------
OOP wrappers for all forecasting models used in the project.
Each model inherits from BaseForecaster and exposes a unified
fit() / predict() / save() / load() interface.

Models implemented:
    - SARIMAForecaster
    - ProphetForecaster
    - XGBoostForecaster
    - LSTMForecaster
    - EnsembleForecaster
"""

import logging
import joblib
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from pathlib import Path
from functools import wraps
from typing import Optional

logger = logging.getLogger(__name__)

ROOT   = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)


# ─── Utilities ────────────────────────────────────────────────────────────────
def require_fit(method):
    """Decorator: raise if model hasn't been trained yet."""
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        if not self.is_fitted:
            raise RuntimeError(
                f"Call .fit() before .{method.__name__}()"
            )
        return method(self, *args, **kwargs)
    return wrapper


def log_step(method):
    """Decorator: log method name and timing."""
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        import time
        logger.info(f"[{self.__class__.__name__}] {method.__name__} started…")
        t0 = time.time()
        result = method(self, *args, **kwargs)
        logger.info(f"[{self.__class__.__name__}] {method.__name__} done in {time.time()-t0:.1f}s")
        return result
    return wrapper


# ─── Base class ───────────────────────────────────────────────────────────────
class BaseForecaster(ABC):
    """Abstract base class for all forecasters."""

    def __init__(self, name: str):
        self.name       = name
        self.is_fitted  = False
        self._model     = None

    @abstractmethod
    def fit(self, train: pd.Series | pd.DataFrame, **kwargs) -> "BaseForecaster":
        ...

    @abstractmethod
    def predict(self, horizon: int, **kwargs) -> np.ndarray:
        ...

    def save(self, path: Optional[Path] = None) -> Path:
        path = path or MODEL_DIR / f"{self.name}.joblib"
        joblib.dump(self, path)
        logger.info(f"Model saved → {path}")
        return path

    @classmethod
    def load(cls, path: Path) -> "BaseForecaster":
        obj = joblib.load(path)
        logger.info(f"Model loaded ← {path}")
        return obj

    def __repr__(self):
        return f"{self.__class__.__name__}(fitted={self.is_fitted})"


# ─── 1. SARIMA ────────────────────────────────────────────────────────────────
class SARIMAForecaster(BaseForecaster):
    """
    Seasonal ARIMA forecaster.
    Default order tuned for daily electricity data.
    """

    def __init__(self, order=(1, 1, 1), seasonal_order=(1, 1, 1, 24)):
        super().__init__("sarima")
        self.order          = order
        self.seasonal_order = seasonal_order

    @log_step
    def fit(self, train: pd.Series, **kwargs) -> "SARIMAForecaster":
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        self._model_fit = SARIMAX(
            train,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False, **kwargs)

        self.is_fitted     = True
        self._last_obs     = train
        return self

    @require_fit
    def predict(self, horizon: int = 24, **kwargs) -> np.ndarray:
        forecast = self._model_fit.forecast(steps=horizon)
        return np.array(forecast)

    @require_fit
    def predict_with_ci(self, horizon: int = 24, alpha: float = 0.05):
        """Return (forecast, lower, upper) confidence interval arrays."""
        res = self._model_fit.get_forecast(steps=horizon)
        ci  = res.conf_int(alpha=alpha)
        return (
            np.array(res.predicted_mean),
            np.array(ci.iloc[:, 0]),
            np.array(ci.iloc[:, 1]),
        )


# ─── 2. Prophet ───────────────────────────────────────────────────────────────
class ProphetForecaster(BaseForecaster):
    """
    Facebook Prophet forecaster with German holiday support.
    """

    def __init__(self, changepoint_prior_scale: float = 0.05):
        super().__init__("prophet")
        self.changepoint_prior_scale = changepoint_prior_scale

    @log_step
    def fit(self, train: pd.Series, **kwargs) -> "ProphetForecaster":
        from prophet import Prophet
        import holidays as hd

        # Build holidays DataFrame for Prophet
        de_holidays = hd.country_holidays("DE", years=range(2005, 2018))
        hol_df = pd.DataFrame(
            [{"ds": str(d), "holiday": name} for d, name in de_holidays.items()]
        )
        hol_df["ds"] = pd.to_datetime(hol_df["ds"])

        self._model = Prophet(
            changepoint_prior_scale=self.changepoint_prior_scale,
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=True,
            holidays=hol_df,
        )

        prophet_df = pd.DataFrame({"ds": train.index, "y": train.values})
        self._model.fit(prophet_df, **kwargs)

        self.is_fitted    = True
        self._last_index  = train.index
        return self

    @require_fit
    def predict(self, horizon: int = 24, **kwargs) -> np.ndarray:
        last_dt  = self._last_index[-1]
        future   = pd.date_range(last_dt, periods=horizon + 1, freq="h")[1:]
        future_df = pd.DataFrame({"ds": future})
        forecast  = self._model.predict(future_df)
        return np.array(forecast["yhat"])

    @require_fit
    def predict_with_ci(self, horizon: int = 24):
        last_dt   = self._last_index[-1]
        future    = pd.date_range(last_dt, periods=horizon + 1, freq="h")[1:]
        future_df = pd.DataFrame({"ds": future})
        fc        = self._model.predict(future_df)
        return (
            np.array(fc["yhat"]),
            np.array(fc["yhat_lower"]),
            np.array(fc["yhat_upper"]),
        )


# ─── 3. XGBoost ───────────────────────────────────────────────────────────────
class XGBoostForecaster(BaseForecaster):
    """
    Gradient Boosted Trees forecaster.
    Requires a pre-built feature matrix from features.py.
    """

    def __init__(self, n_estimators: int = 500, learning_rate: float = 0.05,
                 max_depth: int = 6, subsample: float = 0.8,
                 colsample_bytree: float = 0.8):
        super().__init__("xgboost")
        self.params = dict(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=42,
            n_jobs=-1,
        )

    @log_step
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        **kwargs,
    ) -> "XGBoostForecaster":
        import xgboost as xgb

        self._model      = xgb.XGBRegressor(**self.params)
        eval_set         = [(X_val, y_val)] if X_val is not None else None
        self._model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False,
            **kwargs,
        )
        self._feature_names = list(X_train.columns)
        self.is_fitted      = True
        return self

    @require_fit
    def predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        return self._model.predict(X)

    @require_fit
    def shap_values(self, X: pd.DataFrame):
        """Return SHAP explanation object. Requires shap package."""
        import shap
        explainer = shap.TreeExplainer(self._model)
        return explainer(X)

    @property
    def feature_importance(self) -> pd.Series:
        if not self.is_fitted:
            raise RuntimeError("Not fitted yet.")
        return pd.Series(
            self._model.feature_importances_,
            index=self._feature_names,
        ).sort_values(ascending=False)


# ─── 4. LSTM ──────────────────────────────────────────────────────────────────
class LSTMForecaster(BaseForecaster):
    """
    LSTM-based sequence forecaster.
    Inputs a sliding window of `lookback` hours and outputs `horizon` steps.
    """

    def __init__(self, lookback: int = 168, horizon: int = 24,
                 units: int = 64, dropout: float = 0.2):
        super().__init__("lstm")
        self.lookback = lookback
        self.horizon  = horizon
        self.units    = units
        self.dropout  = dropout
        self._scaler  = None

    def _build_sequences(self, series: np.ndarray):
        X, y = [], []
        for i in range(self.lookback, len(series) - self.horizon + 1):
            X.append(series[i - self.lookback : i])
            y.append(series[i : i + self.horizon])
        return np.array(X)[..., np.newaxis], np.array(y)

    @log_step
    def fit(
        self,
        train: pd.Series,
        val: pd.Series | None = None,
        epochs: int = 30,
        batch_size: int = 64,
        **kwargs,
    ) -> "LSTMForecaster":
        import tensorflow as tf
        from sklearn.preprocessing import MinMaxScaler

        self._scaler = MinMaxScaler()
        scaled_train = self._scaler.fit_transform(train.values.reshape(-1, 1)).flatten()

        X_tr, y_tr = self._build_sequences(scaled_train)

        val_data = None
        if val is not None:
            full = np.concatenate([train.values[-self.lookback:], val.values])
            scaled_full = self._scaler.transform(full.reshape(-1, 1)).flatten()
            X_v, y_v   = self._build_sequences(scaled_full)
            val_data    = (X_v, y_v)

        # Build model
        inp = tf.keras.Input(shape=(self.lookback, 1))
        x   = tf.keras.layers.LSTM(self.units, return_sequences=True)(inp)
        x   = tf.keras.layers.Dropout(self.dropout)(x)
        x   = tf.keras.layers.LSTM(self.units // 2)(x)
        x   = tf.keras.layers.Dropout(self.dropout)(x)
        out = tf.keras.layers.Dense(self.horizon)(x)

        self._model = tf.keras.Model(inp, out)
        self._model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-3),
            loss="mse",
        )

        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5),
        ]

        self._model.fit(
            X_tr, y_tr,
            validation_data=val_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
            **kwargs,
        )
        self._last_series = train
        self.is_fitted    = True
        return self

    @require_fit
    def predict(self, horizon: int | None = None, seed_series: pd.Series | None = None, **kwargs) -> np.ndarray:
        horizon    = horizon or self.horizon
        seed       = seed_series if seed_series is not None else self._last_series
        seed_scaled = self._scaler.transform(
            seed.values[-self.lookback:].reshape(-1, 1)
        ).flatten()
        X = seed_scaled[np.newaxis, :, np.newaxis]
        y_scaled = self._model.predict(X, verbose=0)[0][:horizon]
        return self._scaler.inverse_transform(y_scaled.reshape(-1, 1)).flatten()

    def save(self, path: Optional[Path] = None) -> Path:
        path = path or MODEL_DIR / "lstm"
        self._model.save(str(path) + ".keras")
        # Save scaler and metadata
        meta = {k: v for k, v in self.__dict__.items() if k != "_model"}
        joblib.dump(meta, str(path) + "_meta.joblib")
        logger.info(f"LSTM saved → {path}")
        return path


# ─── 5. Ensemble ──────────────────────────────────────────────────────────────
class EnsembleForecaster(BaseForecaster):
    """
    Weighted average ensemble of fitted forecasters.
    Learns weights by minimising MAE on a validation set.
    """

    def __init__(self, forecasters: dict[str, BaseForecaster]):
        super().__init__("ensemble")
        self.forecasters = forecasters   # {name: fitted_forecaster}
        self.weights_    = None

    @log_step
    def fit(
        self,
        val_series: pd.Series,
        horizon: int = 24,
        X_val: pd.DataFrame | None = None,
        **kwargs,
    ) -> "EnsembleForecaster":
        """Optimise weights on validation residuals using scipy."""
        from scipy.optimize import minimize

        preds = self._collect_predictions(
            horizon=horizon, n_windows=len(val_series) // horizon,
            val_series=val_series, X_val=X_val,
        )
        actuals = val_series.values[:len(list(preds.values())[0])]

        def objective(w):
            w = np.array(w)
            w = np.abs(w) / np.abs(w).sum()
            ensemble = sum(w[i] * p for i, p in enumerate(preds.values()))
            return np.mean(np.abs(ensemble - actuals))

        n = len(self.forecasters)
        res = minimize(
            objective, x0=np.ones(n) / n,
            bounds=[(0, 1)] * n,
            method="SLSQP",
            constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1},
        )
        raw = np.abs(res.x)
        self.weights_ = raw / raw.sum()
        self.is_fitted = True
        logger.info(
            "Ensemble weights: " +
            ", ".join(f"{n}={w:.3f}" for n, w in zip(self.forecasters, self.weights_))
        )
        return self

    def _collect_predictions(self, horizon, n_windows, val_series, X_val):
        preds = {}
        for name, model in self.forecasters.items():
            if isinstance(model, XGBoostForecaster) and X_val is not None:
                preds[name] = model.predict(X_val.iloc[:n_windows * horizon])
            else:
                preds[name] = model.predict(horizon=horizon)
        return preds

    @require_fit
    def predict(self, horizon: int = 24, X: pd.DataFrame | None = None, **kwargs) -> np.ndarray:
        forecasts = []
        for name, model in self.forecasters.items():
            if isinstance(model, XGBoostForecaster) and X is not None:
                forecasts.append(model.predict(X))
            else:
                forecasts.append(model.predict(horizon=horizon))

        stacked = np.vstack(forecasts)
        return np.average(stacked, axis=0, weights=self.weights_)
