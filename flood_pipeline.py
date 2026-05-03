from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
SOIL_CACHE = DATA_DIR / "soil_moisture_cache.csv"


@dataclass(frozen=True)
class FloodEvent:
    state: str
    year: int
    start: str
    end: str


EVENTS: list[FloodEvent] = [
    FloodEvent("Kerala", 2018, "2018-06-01", "2018-08-19"),
    FloodEvent("Kerala", 2019, "2019-08-01", "2019-08-31"),
    FloodEvent("Kerala", 2020, "2020-06-01", "2020-08-18"),
    FloodEvent("Kerala", 2021, "2021-10-11", "2021-10-26"),
    FloodEvent("Kerala", 2022, "2022-08-01", "2022-09-10"),
    FloodEvent("Assam", 2019, "2019-07-10", "2019-08-02"),
    FloodEvent("Assam", 2020, "2020-06-02", "2020-09-30"),
    FloodEvent("Assam", 2021, "2021-06-07", "2021-09-02"),
    FloodEvent("Assam", 2022, "2022-05-18", "2022-07-17"),
    FloodEvent("Assam", 2023, "2023-06-16", "2023-08-31"),
    FloodEvent("Odisha", 2019, "2019-08-01", "2019-09-20"),
    FloodEvent("Odisha", 2020, "2020-08-05", "2020-09-15"),
    FloodEvent("Odisha", 2021, "2021-08-01", "2021-09-20"),
    FloodEvent("Odisha", 2022, "2022-08-14", "2022-09-10"),
    FloodEvent("Odisha", 2023, "2023-07-25", "2023-09-10"),
    FloodEvent("Uttar Pradesh", 2019, "2019-07-25", "2019-08-15"),
    FloodEvent("Uttar Pradesh", 2020, "2020-06-25", "2020-08-15"),
    FloodEvent("Uttar Pradesh", 2021, "2021-08-15", "2021-09-10"),
    FloodEvent("Uttar Pradesh", 2022, "2022-08-01", "2022-09-10"),
    FloodEvent("Uttar Pradesh", 2023, "2023-07-05", "2023-08-20"),
]


STATE_FILE_PREFIXES = {
    "Kerala": ["Kerala", "kerala"],
    "Assam": ["Assam"],
    "Odisha": ["Odisha"],
    "Uttar Pradesh": ["Uttarpradesh"],
}

STATE_CENTROIDS = {
    "Kerala": (10.8505, 76.2711),
    "Assam": (26.2006, 92.9376),
    "Odisha": (20.9517, 85.0985),
    "Uttar Pradesh": (26.8467, 80.9462),
}


def normalize_state(value: str) -> str:
    text = str(value).strip().lower().replace("_", " ")
    if text in {"uttarpradesh", "uttar pradesh"}:
        return "Uttar Pradesh"
    if text == "kerala":
        return "Kerala"
    if text == "assam":
        return "Assam"
    if text == "odisha":
        return "Odisha"
    return str(value).strip()


def clean_district(value: str) -> str:
    return str(value).strip().upper()


def event_frame() -> pd.DataFrame:
    df = pd.DataFrame([event.__dict__ for event in EVENTS])
    df["start"] = pd.to_datetime(df["start"])
    df["end"] = pd.to_datetime(df["end"])
    return df


def rainfall_files() -> list[Path]:
    paths: list[Path] = []
    for prefixes in STATE_FILE_PREFIXES.values():
        for prefix in prefixes:
            paths.extend(DATA_DIR.glob(f"{prefix}_*.csv"))
    return sorted(p for p in set(paths) if "Riverlevel" not in p.name)


def read_rainfall() -> pd.DataFrame:
    frames = []
    for path in rainfall_files():
        df = pd.read_csv(path)
        required = {"State", "District", "Date", "Avg_rainfall"}
        if not required.issubset(df.columns):
            continue
        df = df[list(required)].copy()
        df["state"] = df["State"].map(normalize_state)
        df["district"] = df["District"].map(clean_district)
        df["date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["rainfall_mm"] = pd.to_numeric(df["Avg_rainfall"], errors="coerce")
        df = df.dropna(subset=["date", "rainfall_mm"])
        frames.append(df[["state", "district", "date", "rainfall_mm"]])
    if not frames:
        raise FileNotFoundError("No rainfall CSV files with State, District, Date, Avg_rainfall were found.")
    rainfall = pd.concat(frames, ignore_index=True)
    rainfall = rainfall[rainfall["state"].isin(STATE_FILE_PREFIXES)]
    return (
        rainfall.groupby(["state", "district", "date"], as_index=False)
        .agg(rainfall_mm=("rainfall_mm", "mean"))
        .sort_values(["state", "district", "date"])
    )


def find_measure_column(columns: Iterable[str]) -> tuple[str | None, str | None]:
    lowered = {column.lower(): column for column in columns}
    for key, column in lowered.items():
        if "river water level" in key:
            return column, "river_level_m"
    for key, column in lowered.items():
        if "river water discharge" in key or "discharge" in key:
            return column, "river_discharge_m3s"
    return None, None


def river_files() -> list[Path]:
    return sorted(DATA_DIR.glob("*_Riverlevel.csv"))


def read_river_levels() -> pd.DataFrame:
    frames = []
    for path in river_files():
        header = pd.read_csv(path, nrows=0).columns
        measure_col, measure_type = find_measure_column(header)
        if measure_col is None:
            continue
        usecols = ["State", "District", "Data Acquisition Time", measure_col]
        extra_cols = [c for c in ["Station", "Latitude", "Longitude"] if c in header]
        df = pd.read_csv(path, usecols=usecols + extra_cols)
        df["state"] = df["State"].map(normalize_state)
        df["district"] = df["District"].map(clean_district)
        df["date"] = pd.to_datetime(
            df["Data Acquisition Time"], format="%d-%m-%Y %H:%M", errors="coerce"
        ).dt.normalize()
        df["river_value"] = pd.to_numeric(df[measure_col], errors="coerce")
        df["river_measure_type"] = measure_type
        frames.append(df[["state", "district", "date", "river_value", "river_measure_type"]])
    if not frames:
        return pd.DataFrame(
            columns=["state", "district", "date", "river_value", "river_measure_type"]
        )
    river = pd.concat(frames, ignore_index=True).dropna(subset=["date", "river_value"])
    river = river[river["state"].isin(STATE_FILE_PREFIXES)]
    return (
        river.groupby(["state", "district", "date", "river_measure_type"], as_index=False)
        .agg(
            river_daily_mean=("river_value", "mean"),
            river_daily_max=("river_value", "max"),
            river_daily_min=("river_value", "min"),
            river_obs_count=("river_value", "size"),
        )
        .sort_values(["state", "district", "date"])
    )


def add_event_labels(df: pd.DataFrame) -> pd.DataFrame:
    events = event_frame()
    out = df.copy()
    out["is_flood_event"] = 0
    out["event_year"] = np.nan
    out["event_start"] = pd.NaT
    out["event_end"] = pd.NaT
    for event in events.itertuples(index=False):
        mask = (out["state"] == event.state) & (out["date"].between(event.start, event.end))
        out.loc[mask, "is_flood_event"] = 1
        out.loc[mask, "event_year"] = event.year
        out.loc[mask, "event_start"] = event.start
        out.loc[mask, "event_end"] = event.end
    out["event_year"] = out["event_year"].astype("Int64")
    return out


def fetch_open_meteo_soil_moisture(
    state: str, start_date: pd.Timestamp, end_date: pd.Timestamp, api_key: str | None
) -> pd.DataFrame:
    lat, lon = STATE_CENTROIDS[state]
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "daily": "soil_moisture_0_to_7cm_mean,soil_moisture_7_to_28cm_mean",
        "timezone": "auto",
    }
    if api_key:
        params["apikey"] = api_key
    response = requests.get("https://archive-api.open-meteo.com/v1/archive", params=params, timeout=60)
    response.raise_for_status()
    payload = response.json()
    daily = payload.get("daily", {})
    if "time" not in daily:
        raise ValueError(f"Soil moisture response for {state} did not include daily time values.")
    return pd.DataFrame(
        {
            "state": state,
            "date": pd.to_datetime(daily["time"]),
            "soil_moisture_0_7cm": daily.get("soil_moisture_0_to_7cm_mean"),
            "soil_moisture_7_28cm": daily.get("soil_moisture_7_to_28cm_mean"),
            "soil_source": "open_meteo_archive",
        }
    )


def load_or_fetch_soil_moisture(df: pd.DataFrame, fetch: bool) -> pd.DataFrame:
    if SOIL_CACHE.exists():
        soil = pd.read_csv(SOIL_CACHE, parse_dates=["date"])
    else:
        soil = pd.DataFrame()
    if fetch:
        api_key = os.getenv("SOIL_MOISTURE_API_KEY") or os.getenv("OPEN_METEO_API_KEY")
        fetched = []
        for state, state_df in df.groupby("state"):
            fetched.append(
                fetch_open_meteo_soil_moisture(
                    state, state_df["date"].min(), state_df["date"].max(), api_key
                )
            )
        soil = pd.concat([soil, *fetched], ignore_index=True) if not soil.empty else pd.concat(fetched)
        soil = soil.drop_duplicates(["state", "date"], keep="last").sort_values(["state", "date"])
        soil.to_csv(SOIL_CACHE, index=False)
    if soil.empty:
        return pd.DataFrame(columns=["state", "date", "soil_moisture_0_7cm", "soil_moisture_7_28cm"])
    soil["state"] = soil["state"].map(normalize_state)
    soil["date"] = pd.to_datetime(soil["date"]).dt.normalize()
    return soil


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["state", "district", "date"]).copy()
    out["year"] = out["date"].dt.year
    out["month"] = out["date"].dt.month
    out["day_of_year"] = out["date"].dt.dayofyear
    group = out.groupby(["state", "district"], group_keys=False)
    for window in [3, 7, 14]:
        out[f"rainfall_roll{window}_mean"] = group["rainfall_mm"].transform(
            lambda s: s.rolling(window, min_periods=1).mean()
        )
        out[f"rainfall_roll{window}_sum"] = group["rainfall_mm"].transform(
            lambda s: s.rolling(window, min_periods=1).sum()
        )
    for lag in [1, 3, 7]:
        out[f"rainfall_lag{lag}"] = group["rainfall_mm"].shift(lag)
    for column in ["river_daily_mean", "river_daily_max"]:
        if column in out:
            out[f"{column}_lag1"] = group[column].shift(1)
            out[f"{column}_roll3_mean"] = group[column].transform(
                lambda s: s.rolling(3, min_periods=1).mean()
            )
    return out


def build_dataset(fetch_soil: bool = False) -> pd.DataFrame:
    rainfall = read_rainfall()
    river = read_river_levels()
    merged = rainfall.merge(river, on=["state", "district", "date"], how="left")
    if "river_measure_type" in merged:
        merged["river_is_discharge"] = (merged["river_measure_type"] == "river_discharge_m3s").astype(int)
    if fetch_soil:
        soil = load_or_fetch_soil_moisture(merged, fetch=True)
        merged = merged.merge(soil, on=["state", "date"], how="left")
    merged = add_event_labels(merged)
    merged = add_time_features(merged)
    return merged


def write_eda(df: pd.DataFrame) -> None:
    eda_dir = OUTPUT_DIR / "eda"
    eda_dir.mkdir(parents=True, exist_ok=True)
    summary = (
        df.groupby(["state", "year"], as_index=False)
        .agg(
            rows=("is_flood_event", "size"),
            flood_event_rows=("is_flood_event", "sum"),
            districts=("district", "nunique"),
            avg_rainfall_mm=("rainfall_mm", "mean"),
            max_rainfall_mm=("rainfall_mm", "max"),
            avg_river_daily_mean=("river_daily_mean", "mean"),
        )
        .sort_values(["state", "year"])
    )
    summary.to_csv(eda_dir / "state_year_summary.csv", index=False)

    missing = df.isna().mean().sort_values(ascending=False).rename("missing_fraction")
    missing.to_csv(eda_dir / "missing_values.csv")

    event_summary = (
        df.groupby(["state", "is_flood_event"], as_index=False)
        .agg(
            rows=("is_flood_event", "size"),
            rainfall_mean=("rainfall_mm", "mean"),
            rainfall_p95=("rainfall_mm", lambda s: s.quantile(0.95)),
            river_mean=("river_daily_mean", "mean"),
        )
    )
    event_summary.to_csv(eda_dir / "event_vs_nonevent_summary.csv", index=False)

    for state, state_df in df.groupby("state"):
        daily = (
            state_df.groupby("date", as_index=False)
            .agg(rainfall_mm=("rainfall_mm", "mean"), is_flood_event=("is_flood_event", "max"))
            .sort_values("date")
        )
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(daily["date"], daily["rainfall_mm"], color="#1f77b4", linewidth=1.2)
        flood_days = daily[daily["is_flood_event"] == 1]
        ax.scatter(flood_days["date"], flood_days["rainfall_mm"], color="#d62728", s=8, label="Flood event")
        ax.set_title(f"{state}: daily mean rainfall with flood-event windows")
        ax.set_ylabel("Rainfall (mm)")
        ax.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(eda_dir / f"{state.replace(' ', '_').lower()}_rainfall_timeline.png", dpi=160)
        plt.close(fig)

    numeric = df.select_dtypes(include=[np.number])
    corr_cols = [
        c
        for c in numeric.columns
        if c
        in {
            "is_flood_event",
            "rainfall_mm",
            "rainfall_roll3_sum",
            "rainfall_roll7_sum",
            "river_daily_mean",
            "river_daily_max",
        }
    ]
    corr = numeric[corr_cols].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr.index)), corr.index)
    fig.colorbar(image, ax=ax)
    ax.set_title("Feature correlation")
    fig.tight_layout()
    fig.savefig(eda_dir / "correlation_heatmap.png", dpi=160)
    plt.close(fig)


def stratified_limit(df: pd.DataFrame, target: str, max_rows: int, random_state: int = 42) -> pd.DataFrame:
    if len(df) <= max_rows:
        return df
    class_counts = df[target].value_counts()
    allocations = (class_counts / class_counts.sum() * max_rows).round().astype(int)
    allocations = allocations.clip(lower=1)
    while allocations.sum() > max_rows:
        largest = allocations.idxmax()
        allocations.loc[largest] -= 1
    while allocations.sum() < max_rows:
        smallest = allocations.idxmin()
        allocations.loc[smallest] += 1
    parts = []
    for label, n_rows in allocations.items():
        class_df = df[df[target] == label]
        parts.append(class_df.sample(n=min(n_rows, len(class_df)), random_state=random_state))
    return pd.concat(parts).sample(frac=1, random_state=random_state).reset_index(drop=True)


def train_models(df: pd.DataFrame, max_train_rows: int = 60000, max_test_rows: int = 40000) -> None:
    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Model training needs scikit-learn and xgboost. Run: pip install -r requirements.txt"
        ) from exc
    try:
        from xgboost import XGBClassifier
    except ModuleNotFoundError as exc:
        raise RuntimeError("XGBoost is missing. Run: pip install -r requirements.txt") from exc

    model_dir = OUTPUT_DIR / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    model_df = df.copy()
    model_df = model_df[model_df["year"].between(2019, 2023)]
    target = "is_flood_event"
    numeric_features = [
        "rainfall_mm",
        "rainfall_roll3_mean",
        "rainfall_roll3_sum",
        "rainfall_roll7_mean",
        "rainfall_roll7_sum",
        "rainfall_roll14_mean",
        "rainfall_roll14_sum",
        "rainfall_lag1",
        "rainfall_lag3",
        "rainfall_lag7",
        "river_daily_mean",
        "river_daily_max",
        "river_daily_min",
        "river_obs_count",
        "river_daily_mean_lag1",
        "river_daily_mean_roll3_mean",
        "river_daily_max_lag1",
        "river_daily_max_roll3_mean",
        "river_is_discharge",
        "month",
        "day_of_year",
    ]
    numeric_features = [c for c in numeric_features if c in model_df.columns and not model_df[c].isna().all()]
    categorical_features = ["state", "district"]

    train_mask = model_df["year"] < 2023
    test_mask = model_df["year"] == 2023
    if test_mask.sum() == 0 or model_df.loc[test_mask, target].nunique() < 2:
        train_mask = model_df["year"] <= 2021
        test_mask = model_df["year"] >= 2022

    train_df = stratified_limit(model_df.loc[train_mask], target, max_train_rows)
    test_df = stratified_limit(model_df.loc[test_mask], target, max_test_rows)

    X_train = train_df[numeric_features + categorical_features]
    y_train = train_df[target]
    X_test = test_df[numeric_features + categorical_features]
    y_test = test_df[target]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_features),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )
    scale_pos_weight = max((y_train == 0).sum() / max((y_train == 1).sum(), 1), 1)
    models = {
        "random_forest": RandomForestClassifier(
            n_estimators=60,
            max_depth=None,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=80,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=1,
        ),
    }

    metrics: dict[str, dict[str, object]] = {
        "split": {
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "train_years": sorted(model_df.loc[train_mask, "year"].unique().tolist()),
            "test_years": sorted(model_df.loc[test_mask, "year"].unique().tolist()),
            "features": numeric_features + categorical_features,
        }
    }

    for name, model in models.items():
        pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        probabilities = pipeline.predict_proba(X_test)[:, 1]
        metrics[name] = {
            "roc_auc": float(roc_auc_score(y_test, probabilities)) if y_test.nunique() == 2 else None,
            "classification_report": classification_report(y_test, predictions, output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        }

        feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()
        importances = getattr(pipeline.named_steps["model"], "feature_importances_", None)
        if importances is not None:
            pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values(
                "importance", ascending=False
            ).head(40).to_csv(model_dir / f"{name}_feature_importance.csv", index=False)

    (model_dir / "model_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build flood analysis dataset, EDA, and ML models.")
    parser.add_argument(
        "--fetch-soil-moisture",
        action="store_true",
        help="Fetch soil moisture from Open-Meteo Archive and cache it in soil_moisture_cache.csv.",
    )
    parser.add_argument(
        "--skip-models",
        action="store_true",
        help="Only build the combined dataset and EDA outputs.",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)
    dataset = build_dataset(fetch_soil=args.fetch_soil_moisture)
    dataset.to_csv(OUTPUT_DIR / "combined_flood_dataset.csv", index=False)
    write_eda(dataset)
    if not args.skip_models:
        train_models(dataset)
    print(f"Wrote outputs to {OUTPUT_DIR}")
    print(f"Rows: {len(dataset):,}; flood-event rows: {int(dataset['is_flood_event'].sum()):,}")
    missing_2018_kerala = not ((dataset["state"] == "Kerala") & (dataset["year"] == 2018)).any()
    if missing_2018_kerala:
        print("Note: Kerala 2018 event is configured, but no Kerala_2018 rainfall CSV was found.")


if __name__ == "__main__":
    main()
