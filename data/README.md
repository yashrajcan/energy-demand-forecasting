# Data Directory

This directory is populated automatically when you run:

```bash
python src/data_loader.py
```

## What gets downloaded

| File | Description |
|------|-------------|
| `raw/opsd_raw.csv` | Full OPSD hourly time series (Germany, 2006–2017) |
| `raw/weather_berlin.csv` | Hourly temperature + cloud cover from Open-Meteo API |
| `processed/energy_dataset.csv` | Merged, cleaned dataset |
| `processed/features.csv` | Full feature matrix for ML models |

## Sources

- **Load data**: [Open Power System Data](https://open-power-system-data.org/) — free, no API key required
- **Weather data**: [Open-Meteo](https://open-meteo.com/) — free, no API key required

Raw files are excluded from git (see `.gitignore`). The download script handles caching.
