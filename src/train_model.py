from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_preprocessing import (
    clean_merged_data,
    generate_eda_outputs,
    inspect_source_columns,
    load_source_data,
    merge_datasets,
)
from src.evaluate_model import (
    get_feature_importance,
    save_best_model_diagnostics,
    save_feature_importance_outputs,
    train_and_compare_models,
)
from src.feature_engineering import MODEL_FEATURES, create_features, get_model_inputs
from src.prioritization import build_priority_recommendations
from src.reward_optimization import build_reward_optimized_plan, run_reward_simulation
from src.utils import OUTPUT_DIR, PLOTS_DIR, ensure_project_dirs, format_metric, save_json


def build_prediction_output(feature_df: pd.DataFrame, trained_pipeline: Any) -> pd.DataFrame:
    full_X = feature_df[MODEL_FEATURES].copy()
    prediction_output = feature_df[
        [
            "delivery_id",
            "factory_id",
            "project_id",
            "date",
            "distance_km",
            "expected_time_hours",
            "actual_time_hours",
            "priority_level",
            "demand",
            "weather_index",
            "traffic_index",
            "traffic_weather_risk",
        ]
    ].copy()
    prediction_output["predicted_delay_probability"] = trained_pipeline.predict_proba(full_X)[:, 1]
    prediction_output["predicted_delay_class"] = trained_pipeline.predict(full_X).astype(int)
    return prediction_output


def print_final_summary(
    best_result: dict[str, Any],
    feature_importance_df: pd.DataFrame,
    priority_df: pd.DataFrame,
    simulation_df: pd.DataFrame,
) -> None:
    print("\n=== SMART PROCUREMENT AI SUMMARY ===")
    print(f"Best model name: {best_result['model_name']}")
    print(f"Accuracy: {format_metric(best_result['accuracy'])}")
    print(f"Precision: {format_metric(best_result['precision'])}")
    print(f"Recall: {format_metric(best_result['recall'])}")
    print(f"F1-score: {format_metric(best_result['f1'])}")
    print(f"ROC-AUC: {format_metric(best_result['roc_auc'])}")

    print("\nTop 5 important features:")
    print(feature_importance_df.head(5).to_string(index=False))

    print("\nTop 10 deliveries to prioritize:")
    print(priority_df.head(10).to_string(index=False))

    print("\nReward simulation summary:")
    print(simulation_df.to_string(index=False))


def main() -> None:
    ensure_project_dirs()

    cleaned_path = OUTPUT_DIR / "cleaned_merged_data.csv"
    model_path = OUTPUT_DIR / "trained_model.pkl"
    metrics_path = OUTPUT_DIR / "model_metrics.json"
    model_comparison_path = OUTPUT_DIR / "model_comparison.csv"
    feature_importance_path = OUTPUT_DIR / "feature_importance.csv"
    predictions_path = OUTPUT_DIR / "delivery_delay_predictions.csv"
    prioritization_path = OUTPUT_DIR / "delivery_priority_recommendations.csv"
    reward_plan_path = OUTPUT_DIR / "reward_optimized_plan.csv"
    reward_simulation_path = OUTPUT_DIR / "reward_simulation_results.csv"
    data_summary_path = OUTPUT_DIR / "data_summary.json"

    source_frames = load_source_data()
    merged_df = merge_datasets(source_frames)
    cleaned_df, cleaning_summary = clean_merged_data(merged_df)
    cleaned_df.to_csv(cleaned_path, index=False)

    eda_insights = generate_eda_outputs(cleaned_df, plot_dir=PLOTS_DIR)
    feature_df = create_features(cleaned_df)
    X, y = get_model_inputs(feature_df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    best_result, comparison_df, all_results = train_and_compare_models(X_train, X_test, y_train, y_test)
    comparison_df.to_csv(model_comparison_path, index=False)

    save_best_model_diagnostics(best_result, output_dir=PLOTS_DIR)

    feature_importance_df = get_feature_importance(best_result["pipeline"])
    feature_importance_df.to_csv(feature_importance_path, index=False)
    interpretation_artifact = save_feature_importance_outputs(
        feature_importance_df,
        best_result["pipeline"],
        X_train,
        output_dir=PLOTS_DIR,
    )

    model_bundle = {
        "model_name": best_result["model_name"],
        "pipeline": best_result["pipeline"],
        "feature_columns": MODEL_FEATURES,
        "metrics": {
            "accuracy": best_result["accuracy"],
            "precision": best_result["precision"],
            "recall": best_result["recall"],
            "f1": best_result["f1"],
            "roc_auc": best_result["roc_auc"],
        },
    }
    joblib.dump(model_bundle, model_path)

    prediction_output = build_prediction_output(feature_df, best_result["pipeline"])
    prediction_output.to_csv(predictions_path, index=False)

    priority_input = prediction_output.copy()
    priority_df = build_priority_recommendations(priority_input)
    priority_df.to_csv(prioritization_path, index=False)

    reward_input = prediction_output.merge(
        priority_df[
            [
                "delivery_id",
                "priority_score",
                "recommended_action",
            ]
        ],
        on="delivery_id",
        how="left",
    )
    reward_df = build_reward_optimized_plan(reward_input)
    reward_df.to_csv(reward_plan_path, index=False)

    simulation_df = run_reward_simulation(reward_df)
    simulation_df.to_csv(reward_simulation_path, index=False)

    metrics_payload = {
        "best_model_name": best_result["model_name"],
        "best_model_metrics": {
            "accuracy": best_result["accuracy"],
            "precision": best_result["precision"],
            "recall": best_result["recall"],
            "f1": best_result["f1"],
            "roc_auc": best_result["roc_auc"],
        },
        "best_model_confusion_matrix": best_result["confusion_matrix"],
        "best_model_classification_report": best_result["classification_report"],
        "model_comparison": comparison_df.to_dict(orient="records"),
        "top_features": feature_importance_df.head(10).to_dict(orient="records"),
        "interpretability_artifact": interpretation_artifact,
    }
    save_json(metrics_payload, metrics_path)

    data_summary_payload = {
        "source_columns": inspect_source_columns(source_frames),
        "cleaning_summary": cleaning_summary,
        "eda_insights": eda_insights,
        "row_count": int(len(cleaned_df)),
        "factory_count": int(cleaned_df["factory_id"].nunique()),
        "project_count": int(cleaned_df["project_id"].nunique()),
        "delivery_count": int(cleaned_df["delivery_id"].nunique()),
        "date_range": {
            "min": cleaned_df["date"].min().strftime("%Y-%m-%d"),
            "max": cleaned_df["date"].max().strftime("%Y-%m-%d"),
        },
    }
    save_json(data_summary_payload, data_summary_path)

    print_final_summary(best_result, feature_importance_df, priority_df, simulation_df)


if __name__ == "__main__":
    main()
