from __future__ import annotations

import pandas as pd

from src.feature_engineering import PRIORITY_MAPPING
from src.utils import min_max_normalize


def build_priority_recommendations(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a prioritization score and recommended action for every delivery.
    """
    priority_df = predictions_df.copy()
    priority_df["priority_level_score"] = (
        priority_df["priority_level"].map(PRIORITY_MAPPING).fillna(1).astype(int)
    )

    priority_df["predicted_delay_probability_normalized"] = min_max_normalize(
        priority_df["predicted_delay_probability"]
    )
    priority_df["demand_normalized"] = min_max_normalize(priority_df["demand"])
    priority_df["traffic_weather_risk_normalized"] = min_max_normalize(
        priority_df["traffic_weather_risk"]
    )
    priority_df["priority_level_score_normalized"] = min_max_normalize(
        priority_df["priority_level_score"]
    )
    priority_df["distance_km_normalized"] = min_max_normalize(priority_df["distance_km"])

    priority_df["priority_score"] = (
        0.35 * priority_df["predicted_delay_probability_normalized"]
        + 0.25 * priority_df["priority_level_score_normalized"]
        + 0.20 * priority_df["demand_normalized"]
        + 0.15 * priority_df["traffic_weather_risk_normalized"]
        + 0.05 * priority_df["distance_km_normalized"]
    )

    external_risk_threshold = priority_df["traffic_weather_risk"].quantile(0.75)
    priority_df["recommended_action"] = "Normal planning"

    immediate_mask = (
        (priority_df["predicted_delay_probability"] >= 0.75)
        & (priority_df["priority_level"] == "High")
    )
    high_risk_mask = priority_df["predicted_delay_probability"] >= 0.60
    high_priority_mask = priority_df["priority_level"] == "High"
    external_risk_mask = priority_df["traffic_weather_risk"] >= external_risk_threshold

    priority_df.loc[external_risk_mask, "recommended_action"] = (
        "External risk: consider alternate route or time"
    )
    priority_df.loc[high_priority_mask, "recommended_action"] = "Priority project: schedule early"
    priority_df.loc[high_risk_mask, "recommended_action"] = (
        "High risk: prioritize dispatch and monitor closely"
    )
    priority_df.loc[immediate_mask, "recommended_action"] = (
        "Immediate action: allocate fastest route and backup resources"
    )

    selected_columns = [
        "delivery_id",
        "factory_id",
        "project_id",
        "date",
        "priority_level",
        "demand",
        "distance_km",
        "weather_index",
        "traffic_index",
        "predicted_delay_probability",
        "priority_score",
        "recommended_action",
    ]
    return priority_df.sort_values(by="priority_score", ascending=False)[selected_columns]
