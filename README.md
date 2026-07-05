# Flood Analysis and Prediction Pipeline

This project combines the rainfall CSVs, river-level/discharge CSVs, and flood-event windows into one modeling dataset.

Raw data is stored under `data/`. Notebook workflows are stored under `notebooks/`.

## Setup

```powershell
pip install -r requirements.txt
```

## Notebook Workflow

Use these notebooks for the project work:

1. `notebooks/01_eda.ipynb` - combines the data and performs EDA with tables and plots.
2. `notebooks/02_model_training.ipynb` - trains Random Forest and XGBoost and reports metrics/feature importance.

## Script Runner

The notebooks import reusable functions from `flood_pipeline.py`. The script can still be used for fast reruns.

Build the combined dataset and EDA only:

```powershell
python flood_pipeline.py --skip-models
```

Run EDA plus Random Forest and XGBoost:

```powershell
python flood_pipeline.py
```

## Outputs

- `outputs/combined_flood_dataset.csv`
- `outputs/eda/state_year_summary.csv`
- `outputs/eda/event_vs_nonevent_summary.csv`
- `outputs/eda/missing_values.csv`
- `outputs/eda/*_rainfall_timeline.png`
- `outputs/eda/correlation_heatmap.png`
- `outputs/models/model_metrics.json`
- `outputs/models/random_forest_feature_importance.csv`
- `outputs/models/xgboost_feature_importance.csv`

## Notes

- Flood-event windows are encoded inside `flood_pipeline.py`.
- Odisha river data contains manual daily discharge rather than hourly river water level, so the pipeline keeps it as a river feature and flags it with `river_is_discharge`.
