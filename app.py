from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.feature_engineering import build_inference_frame


APP_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = APP_ROOT / "outputs"
PLOTS_DIR = OUTPUT_DIR / "plots"


st.set_page_config(
    page_title="SmartProcure AI",
    page_icon="🚚",
    layout="wide",
)


@st.cache_resource
def load_model_bundle():
    model_path = OUTPUT_DIR / "trained_model.pkl"
    if not model_path.exists():
        return None
    return joblib.load(model_path)


@st.cache_data
def load_json_file(file_path: Path):
    if not file_path.exists():
        return None
    with file_path.open("r", encoding="utf-8") as file_pointer:
        return json.load(file_pointer)


@st.cache_data
def load_csv_file(file_path: Path, parse_dates: tuple[str, ...] = ()):
    if not file_path.exists():
        return None
    return pd.read_csv(file_path, parse_dates=list(parse_dates))


def risk_level(probability: float) -> str:
    if probability < 0.40:
        return "Low Risk"
    if probability <= 0.70:
        return "Medium Risk"
    return "High Risk"


def show_missing_artifact_message() -> None:
    st.warning(
        "Some generated artifacts are missing. Run `python src/train_model.py` first to create "
        "the trained model, plots, predictions, and recommendation files."
    )


def simulate_top_k_plan(reward_df: pd.DataFrame, top_k: int) -> pd.DataFrame:
    ordered = reward_df.sort_values(by=["date", "optimized_planning_score"], ascending=[True, False])
    return ordered.groupby("date", group_keys=False).head(top_k)


def render_overview(cleaned_df: pd.DataFrame | None, data_summary: dict | None) -> None:
    st.subheader("Overview")
    st.markdown(
        """
        SmartProcure AI predicts delivery delay risk before dispatch, highlights the main drivers
        of disruption, and turns those predictions into a practical prioritization and reward-based
        planning workflow.
        """
    )

    if cleaned_df is None:
        show_missing_artifact_message()
        return

    total_deliveries = int(cleaned_df["delivery_id"].nunique())
    delay_rate = float(cleaned_df["delay_flag"].mean())
    factory_count = int(cleaned_df["factory_id"].nunique())
    project_count = int(cleaned_df["project_id"].nunique())

    metric_columns = st.columns(4)
    metric_columns[0].metric("Total Deliveries", f"{total_deliveries:,}")
    metric_columns[1].metric("Delay Rate", f"{delay_rate:.2%}")
    metric_columns[2].metric("Factories", factory_count)
    metric_columns[3].metric("Projects in Deliveries", project_count)

    summary_columns = st.columns(2)
    with summary_columns[0]:
        st.markdown("**Dataset summary**")
        st.dataframe(cleaned_df.head(10), use_container_width=True)
    with summary_columns[1]:
        st.markdown("**EDA highlights**")
        narrative_insights = []
        if data_summary:
            narrative_insights = data_summary.get("eda_insights", {}).get("narrative_insights", [])
        for insight in narrative_insights[:5]:
            st.write(f"- {insight}")

    plot_columns = st.columns(2)
    delay_plot = PLOTS_DIR / "01_delay_distribution.png"
    heatmap_plot = PLOTS_DIR / "09_correlation_heatmap.png"
    if delay_plot.exists():
        plot_columns[0].image(str(delay_plot), caption="Delay distribution", use_container_width=True)
    if heatmap_plot.exists():
        plot_columns[1].image(str(heatmap_plot), caption="Correlation heatmap", use_container_width=True)


def render_delay_prediction(cleaned_df: pd.DataFrame | None, model_bundle) -> None:
    st.subheader("Delay Prediction")
    if cleaned_df is None or model_bundle is None:
        show_missing_artifact_message()
        return

    defaults = {
        "distance_km": float(cleaned_df["distance_km"].median()),
        "expected_time_hours": float(cleaned_df["expected_time_hours"].median()),
        "weather_index": float(cleaned_df["weather_index"].median()),
        "traffic_index": float(cleaned_df["traffic_index"].median()),
        "demand": float(cleaned_df["demand"].median()),
        "base_production_per_week": float(cleaned_df["base_production_per_week"].median()),
        "production_variability": float(cleaned_df["production_variability"].median()),
        "max_storage": float(cleaned_df["max_storage"].median()),
    }

    input_columns = st.columns(3)
    with input_columns[0]:
        distance_km = st.number_input("Distance (km)", min_value=0.0, value=defaults["distance_km"])
        weather_index = st.slider("Weather Index", 0.0, 1.0, defaults["weather_index"], 0.01)
        priority_level = st.selectbox("Priority Level", ["High", "Medium", "Low"], index=0)
    with input_columns[1]:
        expected_time_hours = st.number_input(
            "Expected Time (hours)",
            min_value=0.1,
            value=max(defaults["expected_time_hours"], 0.1),
        )
        traffic_index = st.slider("Traffic Index", 0.0, 1.0, defaults["traffic_index"], 0.01)
        demand = st.number_input("Demand", min_value=0.0, value=defaults["demand"])
    with input_columns[2]:
        base_production_per_week = st.number_input(
            "Base Production per Week",
            min_value=0.0,
            value=defaults["base_production_per_week"],
        )
        production_variability = st.slider(
            "Production Variability", 0.0, 1.0, defaults["production_variability"], 0.01
        )
        max_storage = st.number_input("Max Storage", min_value=0.0, value=defaults["max_storage"])

    selected_date = st.date_input("Planning Date", value=date.today())

    if st.button("Predict Delay Risk", type="primary"):
        payload = {
            "distance_km": distance_km,
            "expected_time_hours": expected_time_hours,
            "weather_index": weather_index,
            "traffic_index": traffic_index,
            "demand": demand,
            "priority_level": priority_level,
            "base_production_per_week": base_production_per_week,
            "production_variability": production_variability,
            "max_storage": max_storage,
            "date": selected_date,
        }

        inference_frame = build_inference_frame(payload)
        probability = float(model_bundle["pipeline"].predict_proba(inference_frame)[0, 1])
        predicted_class = int(model_bundle["pipeline"].predict(inference_frame)[0])
        predicted_label = "Delayed" if predicted_class == 1 else "On-time"

        result_columns = st.columns(3)
        result_columns[0].metric("Predicted Delay Probability", f"{probability:.2%}")
        result_columns[1].metric("Predicted Class", predicted_label)
        result_columns[2].metric("Risk Level", risk_level(probability))

        st.info(
            "The model evaluates operational, external, and capacity signals available before "
            "dispatch. `actual_time_hours` is intentionally excluded from prediction to avoid leakage."
        )


def render_feature_importance(feature_importance_df: pd.DataFrame | None) -> None:
    st.subheader("Feature Importance")
    if feature_importance_df is None:
        show_missing_artifact_message()
        return

    shap_plot = PLOTS_DIR / "shap_summary.png"
    importance_plot = PLOTS_DIR / "feature_importance.png"

    chart_column, table_column = st.columns([2, 1])
    with chart_column:
        if shap_plot.exists():
            st.image(str(shap_plot), caption="SHAP summary plot", use_container_width=True)
        elif importance_plot.exists():
            st.image(str(importance_plot), caption="Feature importance", use_container_width=True)
    with table_column:
        st.dataframe(feature_importance_df.head(10), use_container_width=True)

    top_features = feature_importance_df["feature"].head(5).tolist()
    st.markdown(
        "Top contributors in this run were "
        + ", ".join(f"`{feature}`" for feature in top_features)
        + ". These reflect how external conditions and route intensity shape delivery risk."
    )


def render_prioritization(priority_df: pd.DataFrame | None) -> None:
    st.subheader("Delivery Prioritization")
    if priority_df is None:
        show_missing_artifact_message()
        return

    priority_df = priority_df.copy()
    priority_df["date"] = pd.to_datetime(priority_df["date"])

    selected_priorities = st.multiselect(
        "Filter by Priority Level",
        options=["High", "Medium", "Low"],
        default=["High", "Medium", "Low"],
    )
    available_dates = priority_df["date"].dt.date
    date_range = st.date_input(
        "Filter by Date",
        value=(available_dates.min(), available_dates.max()),
    )

    filtered_df = priority_df[priority_df["priority_level"].isin(selected_priorities)]
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
        filtered_df = filtered_df[
            (filtered_df["date"].dt.date >= start_date) & (filtered_df["date"].dt.date <= end_date)
        ]

    summary_columns = st.columns(3)
    summary_columns[0].metric("Recommended Deliveries", len(filtered_df))
    summary_columns[1].metric(
        "Average Delay Probability",
        f"{filtered_df['predicted_delay_probability'].mean():.2%}" if not filtered_df.empty else "N/A",
    )
    summary_columns[2].metric(
        "Average Priority Score",
        f"{filtered_df['priority_score'].mean():.3f}" if not filtered_df.empty else "N/A",
    )

    st.dataframe(filtered_df.head(100), use_container_width=True)


def render_reward_optimization(
    reward_df: pd.DataFrame | None, reward_simulation_df: pd.DataFrame | None
) -> None:
    st.subheader("Reward Optimization")
    if reward_df is None:
        show_missing_artifact_message()
        return

    reward_df = reward_df.copy()
    reward_df["date"] = pd.to_datetime(reward_df["date"])

    top_k = st.slider("Deliveries prioritized per day", min_value=1, max_value=20, value=10)
    selected_df = simulate_top_k_plan(reward_df, top_k)

    metric_columns = st.columns(4)
    metric_columns[0].metric("Selected Deliveries", len(selected_df))
    metric_columns[1].metric("Total Expected Reward", f"{selected_df['reward_score'].sum():.2f}")
    metric_columns[2].metric(
        "High-Priority Deliveries Served", int((selected_df["priority_level"] == "High").sum())
    )
    metric_columns[3].metric(
        "Expected Delayed Deliveries", f"{selected_df['predicted_delay_probability'].sum():.2f}"
    )

    st.dataframe(selected_df.head(100), use_container_width=True)

    if reward_simulation_df is not None:
        st.markdown("**Saved simulation scenarios**")
        st.dataframe(reward_simulation_df, use_container_width=True)


def render_business_recommendations(feature_importance_df: pd.DataFrame | None) -> None:
    st.subheader("Business Recommendations")
    st.write("a. Dispatch high-risk, high-priority deliveries earlier in the day.")
    st.write("b. Avoid long-distance dispatches when weather and traffic scores spike together.")
    st.write("c. Keep backup suppliers or contingency stock for the most critical project windows.")
    st.write("d. Closely monitor factories with higher production variability or lower slack capacity.")
    st.write("e. Use the reward score with the priority score to rank limited daily dispatch capacity.")

    if feature_importance_df is not None:
        st.markdown("**Model-driven focus areas**")
        for _, row in feature_importance_df.head(5).iterrows():
            st.write(f"- `{row['feature']}`: importance {row['importance']:.4f}")


def main() -> None:
    st.title("SmartProcure AI: Delivery Delay Prediction & Planning Optimization")

    cleaned_df = load_csv_file(OUTPUT_DIR / "cleaned_merged_data.csv", parse_dates=("date",))
    feature_importance_df = load_csv_file(OUTPUT_DIR / "feature_importance.csv")
    priority_df = load_csv_file(
        OUTPUT_DIR / "delivery_priority_recommendations.csv",
        parse_dates=("date",),
    )
    reward_df = load_csv_file(
        OUTPUT_DIR / "reward_optimized_plan.csv",
        parse_dates=("date",),
    )
    reward_simulation_df = load_csv_file(OUTPUT_DIR / "reward_simulation_results.csv")
    data_summary = load_json_file(OUTPUT_DIR / "data_summary.json")
    model_bundle = load_model_bundle()

    section = st.sidebar.radio(
        "Navigate",
        [
            "Overview",
            "Delay Prediction",
            "Feature Importance",
            "Delivery Prioritization",
            "Reward Optimization",
            "Business Recommendations",
        ],
    )

    if section == "Overview":
        render_overview(cleaned_df, data_summary)
    elif section == "Delay Prediction":
        render_delay_prediction(cleaned_df, model_bundle)
    elif section == "Feature Importance":
        render_feature_importance(feature_importance_df)
    elif section == "Delivery Prioritization":
        render_prioritization(priority_df)
    elif section == "Reward Optimization":
        render_reward_optimization(reward_df, reward_simulation_df)
    else:
        render_business_recommendations(feature_importance_df)


if __name__ == "__main__":
    main()
