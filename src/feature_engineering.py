from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd

from src.utils import safe_ratio


PRIORITY_MAPPING = {"Low": 1, "Medium": 2, "High": 3}

MODEL_FEATURES = [
    "distance_km",
    "expected_time_hours",
    "weather_index",
    "traffic_index",
    "demand",
    "priority_level_encoded",
    "base_production_per_week",
    "production_variability",
    "max_storage",
    "traffic_weather_risk",
    "distance_time_ratio",
    "demand_to_storage_ratio",
    "production_risk",
    "is_high_priority",
    "is_medium_priority",
    "day_of_week",
    "month",
]


def create_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Create the business and operational features used by the ML pipeline."""
    feature_df = dataframe.copy()
    feature_df["date"] = pd.to_datetime(feature_df["date"], errors="coerce")
    feature_df["priority_level"] = (
        feature_df["priority_level"].astype(str).str.strip().str.title().fillna("Low")
    )
    feature_df["priority_level_encoded"] = (
        feature_df["priority_level"].map(PRIORITY_MAPPING).fillna(1).astype(int)
    )

    feature_df["traffic_weather_risk"] = feature_df["traffic_index"] * feature_df["weather_index"]
    feature_df["distance_time_ratio"] = safe_ratio(
        feature_df["distance_km"], feature_df["expected_time_hours"]
    )
    feature_df["demand_to_storage_ratio"] = safe_ratio(
        feature_df["demand"], feature_df["max_storage"]
    )
    feature_df["production_risk"] = feature_df["production_variability"] * feature_df["demand"]
    feature_df["is_high_priority"] = (feature_df["priority_level"] == "High").astype(int)
    feature_df["is_medium_priority"] = (feature_df["priority_level"] == "Medium").astype(int)
    feature_df["day_of_week"] = feature_df["date"].dt.dayofweek.fillna(0).astype(int)
    feature_df["month"] = feature_df["date"].dt.month.fillna(1).astype(int)

    for column in MODEL_FEATURES:
        feature_df[column] = pd.to_numeric(feature_df[column], errors="coerce")
        feature_df[column] = feature_df[column].fillna(feature_df[column].median())

    return feature_df


def get_model_inputs(feature_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = feature_df[MODEL_FEATURES].copy()
    y = feature_df["delay_flag"].astype(int).copy()
    return X, y


def build_inference_frame(payload: dict[str, Any]) -> pd.DataFrame:
    """
    Convert dashboard input values into the same feature layout used in training.
    """
    selected_date = payload.get("date")
    if selected_date is None:
        selected_date = date.today()
    selected_timestamp = pd.to_datetime(selected_date)

    base_frame = pd.DataFrame(
        [
            {
                "distance_km": payload["distance_km"],
                "expected_time_hours": payload["expected_time_hours"],
                "weather_index": payload["weather_index"],
                "traffic_index": payload["traffic_index"],
                "demand": payload["demand"],
                "priority_level": payload["priority_level"],
                "base_production_per_week": payload["base_production_per_week"],
                "production_variability": payload["production_variability"],
                "max_storage": payload["max_storage"],
                "date": selected_timestamp,
                "delay_flag": 0,
            }
        ]
    )

    feature_df = create_features(base_frame)
    return feature_df[MODEL_FEATURES]
