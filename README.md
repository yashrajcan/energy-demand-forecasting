# ⚡ Energy Demand Forecasting

> End-to-end time series forecasting project using statistical, machine learning, and deep learning models on real-world electricity load data.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📌 Project Overview

This project forecasts hourly electricity demand using the **Open Power System Data (OPSD)** dataset for Germany. It covers the complete ML lifecycle — from raw data ingestion to a deployed REST API and interactive dashboard.

**Key skills demonstrated:**
- Time series decomposition (trend, seasonality, residuals)
- Feature engineering (lag features, calendar features, weather integration)
- Model benchmarking: SARIMA → Prophet → XGBoost → LSTM → Ensemble
- SHAP-based model explainability
- FastAPI deployment + Streamlit dashboard
- OOP model wrappers, decorators, and clean modular code

---

## 🗂️ Repository Structure

```
energy-demand-forecasting/
│
├── data/                        # Raw and processed datasets
│   └── README.md
│
├── notebooks/
│   ├── 01_eda.ipynb             # Exploratory data analysis
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modeling.ipynb        # SARIMA, Prophet, XGBoost, LSTM
│   └── 04_evaluation.ipynb      # Metrics, residuals, SHAP
│
├── src/
│   ├── data_loader.py           # Data download & preprocessing
│   ├── features.py              # Feature engineering pipeline
│   ├── models.py                # OOP model wrappers
│   └── evaluate.py              # Metrics & visualisations
│
├── app/
│   ├── main.py                  # FastAPI REST API
│   └── dashboard.py             # Streamlit interactive dashboard
│
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

**Source:** [Open Power System Data](https://open-power-system-data.org/)  
**Coverage:** Hourly electricity load for Germany (2006–2017)  
**Features used:** `load_MW`, temperature (from Open-Meteo API), hour, day-of-week, month, holiday flags

The `src/data_loader.py` script automatically downloads and caches the dataset.

---

## 🧠 Models

| Model | Library | Highlights |
|-------|---------|-----------|
| **SARIMA** | statsmodels | Baseline; captures seasonal patterns |
| **Prophet** | prophet | Handles holidays, trend changepoints |
| **XGBoost** | xgboost | Lag + calendar features; SHAP explainability |
| **LSTM** | TensorFlow/Keras | Sequence modelling; multi-step forecasting |
| **Ensemble** | custom | Weighted average of Prophet + XGBoost |

---

## 📈 Results

| Model | RMSE (MW) | MAPE (%) | MAE (MW) |
|-------|-----------|----------|---------|
| SARIMA | ~3200 | ~5.8 | ~2500 |
| Prophet | ~2800 | ~4.9 | ~2100 |
| XGBoost | ~2100 | ~3.6 | ~1650 |
| LSTM | ~1950 | ~3.3 | ~1500 |
| **Ensemble** | **~1820** | **~3.0** | **~1420** |

---

## 🚀 Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/yashrajcan/energy-demand-forecasting.git
cd energy-demand-forecasting

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download data & run feature engineering
python src/data_loader.py
python src/features.py

# 4. Run the API
uvicorn app.main:app --reload

# 5. Launch the dashboard
streamlit run app/dashboard.py
```

---

## 🔌 API Usage

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"horizon_hours": 24, "model": "ensemble"}'
```

**Response:**
```json
{
  "model": "ensemble",
  "horizon_hours": 24,
  "forecast": [58200, 56100, 54300, ...],
  "timestamps": ["2017-01-01T00:00:00", ...]
}
```

---

## 🛠️ Tech Stack

`Python` · `Pandas` · `NumPy` · `Statsmodels` · `Prophet` · `XGBoost` · `TensorFlow` · `SHAP` · `Scikit-learn` · `FastAPI` · `Streamlit` · `Matplotlib` · `Seaborn`

---

## 👤 Author

**Yashraj** · [github.com/yashrajcan](https://github.com/yashrajcan)

*Part of PGP AI & Data Science portfolio*
