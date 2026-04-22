from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.utils import DATA_DIR, PLOTS_DIR, resolve_data_file


SOURCE_FILES = {
    "factories": "Factories.csv",
    "projects": "Projects.csv",
    "deliveries": "Deliveries.csv",
    "external_factors": "External_Factors.csv",
}


def load_source_data(data_dir: Path = DATA_DIR) -> dict[str, pd.DataFrame]:
    """Load all source CSV files and convert date columns where present."""
    frames: dict[str, pd.DataFrame] = {}
    for key, filename in SOURCE_FILES.items():
        file_path = resolve_data_file(data_dir, filename)
        frame = pd.read_csv(file_path)
        if "date" in frame.columns:
            frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frames[key] = frame
    return frames


def inspect_source_columns(frames: dict[str, pd.DataFrame]) -> dict[str, list[str]]:
    return {name: list(frame.columns) for name, frame in frames.items()}


def merge_datasets(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Merge deliveries with factory, project, weather, and traffic context."""
    deliveries = frames["deliveries"].copy()
    factories = frames["factories"].copy().rename(
        columns={
            "latitude": "factory_latitude",
            "longitude": "factory_longitude",
        }
    )
    projects = frames["projects"].copy().rename(
        columns={
            "latitude": "project_latitude",
            "longitude": "project_longitude",
        }
    )
    external_factors = frames["external_factors"].copy()

    required_keys = {
        "deliveries": {"delivery_id", "factory_id", "project_id", "date", "delay_flag"},
        "factories": {"factory_id"},
        "projects": {"project_id"},
        "external_factors": {"date"},
    }

    for frame_name, columns in required_keys.items():
        missing_columns = columns.difference(frames[frame_name].columns)
        if missing_columns:
            raise KeyError(f"Missing required columns in {frame_name}: {sorted(missing_columns)}")

    merged = deliveries.merge(factories, on="factory_id", how="left", validate="m:1")
    merged = merged.merge(projects, on="project_id", how="left", validate="m:1")
    merged = merged.merge(external_factors, on="date", how="left", validate="m:1")
    return merged


def clean_merged_data(merged_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Fill missing values, remove duplicates, and make sure the target dtype is correct.
    """
    cleaned = merged_df.copy()
    duplicate_count = int(cleaned.duplicated().sum())
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    numeric_columns = cleaned.select_dtypes(include=["number"]).columns.tolist()
    datetime_columns = cleaned.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns.tolist()
    categorical_columns = cleaned.select_dtypes(include=["object", "category"]).columns.tolist()

    missing_before = cleaned.isna().sum().to_dict()

    for column in numeric_columns:
        median_value = cleaned[column].median()
        cleaned[column] = cleaned[column].fillna(median_value)

    for column in categorical_columns:
        mode_values = cleaned[column].mode(dropna=True)
        fill_value = mode_values.iloc[0] if not mode_values.empty else "Unknown"
        cleaned[column] = cleaned[column].fillna(fill_value)

    for column in datetime_columns:
        mode_values = cleaned[column].mode(dropna=True)
        fill_value = mode_values.iloc[0] if not mode_values.empty else pd.Timestamp("1970-01-01")
        cleaned[column] = cleaned[column].fillna(fill_value)

    cleaned["delay_flag"] = pd.to_numeric(cleaned["delay_flag"], errors="coerce").fillna(0).astype(int)

    summary = {
        "row_count": int(len(cleaned)),
        "column_count": int(cleaned.shape[1]),
        "duplicate_rows_removed": duplicate_count,
        "missing_values_before_cleaning": missing_before,
        "missing_values_after_cleaning": cleaned.isna().sum().to_dict(),
        "dtypes": {column: str(dtype) for column, dtype in cleaned.dtypes.items()},
    }
    return cleaned, summary


def _save_plot(plot_path: Path, title: str) -> None:
    plt.title(title)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()


def generate_eda_outputs(cleaned_df: pd.DataFrame, plot_dir: Path = PLOTS_DIR) -> dict[str, Any]:
    """
    Create the required EDA charts and return business-friendly narrative insights.
    """
    plot_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="crest")

    eda_df = cleaned_df.copy()
    eda_df["delay_status"] = eda_df["delay_flag"].map({0: "On-time", 1: "Delayed"})
    eda_df["priority_level"] = (
        eda_df["priority_level"].astype(str).str.strip().str.title().fillna("Unknown")
    )
    priority_order = ["High", "Medium", "Low"]

    plt.figure(figsize=(7, 5))
    sns.countplot(data=eda_df, x="delay_status", order=["On-time", "Delayed"])
    _save_plot(plot_dir / "01_delay_distribution.png", "Delay Distribution")

    delay_percentages = (
        eda_df["delay_status"].value_counts(normalize=True).mul(100).reindex(["On-time", "Delayed"])
    )
    plt.figure(figsize=(7, 5))
    sns.barplot(x=delay_percentages.index, y=delay_percentages.values)
    plt.ylabel("Percentage of Deliveries")
    _save_plot(plot_dir / "02_delay_percentage.png", "Delay Percentage")

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=eda_df, x="delay_status", y="distance_km", order=["On-time", "Delayed"])
    _save_plot(plot_dir / "03_distance_vs_delay.png", "Distance vs Delay")

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=eda_df, x="delay_status", y="expected_time_hours", order=["On-time", "Delayed"])
    _save_plot(plot_dir / "04_expected_time_vs_delay.png", "Expected Time vs Delay")

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=eda_df, x="delay_status", y="weather_index", order=["On-time", "Delayed"])
    _save_plot(plot_dir / "05_weather_vs_delay.png", "Weather Index vs Delay")

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=eda_df, x="delay_status", y="traffic_index", order=["On-time", "Delayed"])
    _save_plot(plot_dir / "06_traffic_vs_delay.png", "Traffic Index vs Delay")

    plt.figure(figsize=(8, 5))
    sns.countplot(data=eda_df, x="priority_level", hue="delay_status", order=priority_order)
    _save_plot(plot_dir / "07_priority_level_vs_delay.png", "Priority Level vs Delay")

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=eda_df, x="delay_status", y="demand", order=["On-time", "Delayed"])
    _save_plot(plot_dir / "08_demand_vs_delay.png", "Demand vs Delay")

    plt.figure(figsize=(12, 9))
    numeric_df = eda_df.select_dtypes(include=["number"])
    correlation_matrix = numeric_df.corr(numeric_only=True)
    sns.heatmap(correlation_matrix, cmap="coolwarm", center=0, annot=False)
    _save_plot(plot_dir / "09_correlation_heatmap.png", "Correlation Heatmap")

    priority_delay_rate = (
        eda_df.groupby("priority_level", observed=False)["delay_flag"]
        .mean()
        .reindex(priority_order)
        .reset_index()
    )
    plt.figure(figsize=(8, 5))
    sns.barplot(data=priority_delay_rate, x="priority_level", y="delay_flag", order=priority_order)
    plt.ylabel("Average Delay Rate")
    _save_plot(plot_dir / "10_average_delay_rate_by_priority.png", "Average Delay Rate by Priority")

    traffic_threshold = eda_df["traffic_index"].median()
    weather_threshold = eda_df["weather_index"].median()
    distance_threshold = eda_df["distance_km"].median()
    demand_threshold = eda_df["demand"].median()

    high_traffic_rate = eda_df.loc[eda_df["traffic_index"] >= traffic_threshold, "delay_flag"].mean()
    low_traffic_rate = eda_df.loc[eda_df["traffic_index"] < traffic_threshold, "delay_flag"].mean()
    severe_weather_rate = eda_df.loc[eda_df["weather_index"] >= weather_threshold, "delay_flag"].mean()
    mild_weather_rate = eda_df.loc[eda_df["weather_index"] < weather_threshold, "delay_flag"].mean()
    long_distance_rate = eda_df.loc[eda_df["distance_km"] >= distance_threshold, "delay_flag"].mean()
    short_distance_rate = eda_df.loc[eda_df["distance_km"] < distance_threshold, "delay_flag"].mean()
    high_demand_group = eda_df.loc[eda_df["demand"] >= demand_threshold, "delay_flag"]
    low_demand_group = eda_df.loc[eda_df["demand"] < demand_threshold, "delay_flag"]
    high_demand_rate = high_demand_group.mean()
    low_demand_rate = low_demand_group.mean() if not low_demand_group.empty else None
    factory_delay_rate = (
        eda_df.groupby("factory_id", observed=False)["delay_flag"].mean().sort_values(ascending=False)
    )

    strongest_correlations = (
        correlation_matrix["delay_flag"]
        .drop(labels=["delay_flag"], errors="ignore")
        .abs()
        .sort_values(ascending=False)
        .head(5)
        .to_dict()
    )

    insights = {
        "overall_delay_rate": float(eda_df["delay_flag"].mean()),
        "delay_distribution": eda_df["delay_flag"].value_counts().sort_index().to_dict(),
        "high_traffic_delay_rate": float(high_traffic_rate),
        "low_traffic_delay_rate": float(low_traffic_rate),
        "severe_weather_delay_rate": float(severe_weather_rate),
        "mild_weather_delay_rate": float(mild_weather_rate),
        "long_distance_delay_rate": float(long_distance_rate),
        "short_distance_delay_rate": float(short_distance_rate),
        "high_demand_delay_rate": float(high_demand_rate),
        "low_demand_delay_rate": float(low_demand_rate) if low_demand_rate is not None else None,
        "priority_delay_rate": {
            row["priority_level"]: float(row["delay_flag"])
            for _, row in priority_delay_rate.dropna(subset=["priority_level"]).iterrows()
        },
        "factory_delay_rate": {index: float(value) for index, value in factory_delay_rate.items()},
        "strongest_numeric_correlations": {
            key: float(value) for key, value in strongest_correlations.items()
        },
        "narrative_insights": [
            (
                f"The dataset is extremely delay-heavy, with an overall delay rate of "
                f"{eda_df['delay_flag'].mean():.2%}. This makes business prioritization crucial "
                "and also means pure accuracy must be interpreted carefully."
            ),
            (
                f"High-traffic days show a delay rate of {high_traffic_rate:.2%} versus "
                f"{low_traffic_rate:.2%} on lower-traffic days."
            ),
            (
                f"More severe weather days show a delay rate of {severe_weather_rate:.2%} versus "
                f"{mild_weather_rate:.2%} when weather is milder."
            ),
            (
                f"Longer-distance deliveries have a delay rate of {long_distance_rate:.2%}, "
                f"compared with {short_distance_rate:.2%} for shorter routes."
            ),
            (
                "Demand is effectively constant in this dataset, so it does not create meaningful "
                "separation in delay risk and should be treated as a weak predictor here."
                if low_demand_group.empty
                else (
                    f"High-demand deliveries show a delay rate of {high_demand_rate:.2%}, versus "
                    f"{low_demand_rate:.2%} for lower-demand requests."
                )
            ),
            (
                f"The most delay-prone factory in this sample is {factory_delay_rate.index[0]} "
                f"with a delay rate of {factory_delay_rate.iloc[0]:.2%}."
            ),
        ],
    }
    return insights
