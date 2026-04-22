from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight

from src.utils import PLOTS_DIR, strip_feature_prefix


RANDOM_STATE = 42


def _build_preprocessor(numeric_features: list[str], *, scale_numeric: bool) -> ColumnTransformer:
    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(steps=numeric_steps)
    return ColumnTransformer(
        transformers=[("num", numeric_pipeline, numeric_features)],
        remainder="drop",
    )


def get_model_candidates(numeric_features: list[str]) -> dict[str, Pipeline]:
    candidates: dict[str, Pipeline] = {
        "Logistic Regression": Pipeline(
            steps=[
                ("preprocessor", _build_preprocessor(numeric_features, scale_numeric=True)),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        solver="liblinear",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Decision Tree": Pipeline(
            steps=[
                ("preprocessor", _build_preprocessor(numeric_features, scale_numeric=False)),
                (
                    "model",
                    DecisionTreeClassifier(
                        max_depth=6,
                        min_samples_leaf=4,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("preprocessor", _build_preprocessor(numeric_features, scale_numeric=False)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=400,
                        min_samples_leaf=2,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "Gradient Boosting": Pipeline(
            steps=[
                ("preprocessor", _build_preprocessor(numeric_features, scale_numeric=False)),
                ("model", GradientBoostingClassifier(random_state=RANDOM_STATE)),
            ]
        ),
    }

    try:
        from xgboost import XGBClassifier

        candidates["XGBoost"] = Pipeline(
            steps=[
                ("preprocessor", _build_preprocessor(numeric_features, scale_numeric=False)),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=300,
                        max_depth=4,
                        learning_rate=0.05,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        objective="binary:logistic",
                        eval_metric="logloss",
                        random_state=RANDOM_STATE,
                        tree_method="hist",
                    ),
                ),
            ]
        )
    except ImportError:
        pass

    return candidates


def _safe_roc_auc(y_true: pd.Series, probabilities: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, probabilities))
    except ValueError:
        return float("nan")


def _fit_pipeline(pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    estimator = pipeline.named_steps["model"]
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
    fit_parameters: dict[str, Any] = {}
    if "sample_weight" in inspect.signature(estimator.fit).parameters:
        fit_parameters["model__sample_weight"] = sample_weight
    pipeline.fit(X_train, y_train, **fit_parameters)
    return pipeline


def evaluate_single_model(
    model_name: str,
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict[str, Any]:
    trained_pipeline = _fit_pipeline(pipeline, X_train, y_train)
    predicted_labels = trained_pipeline.predict(X_test)
    predicted_probabilities = trained_pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "model_name": model_name,
        "accuracy": float(accuracy_score(y_test, predicted_labels)),
        "precision": float(precision_score(y_test, predicted_labels, zero_division=0)),
        "recall": float(recall_score(y_test, predicted_labels, zero_division=0)),
        "f1": float(f1_score(y_test, predicted_labels, zero_division=0)),
        "roc_auc": _safe_roc_auc(y_test, predicted_probabilities),
        "confusion_matrix": confusion_matrix(y_test, predicted_labels, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            y_test,
            predicted_labels,
            output_dict=True,
            zero_division=0,
        ),
        "y_test_actual": y_test.to_numpy(),
        "pipeline": trained_pipeline,
        "y_test_probabilities": predicted_probabilities,
        "y_test_predictions": predicted_labels,
    }
    return metrics


def train_and_compare_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[dict[str, Any], pd.DataFrame, list[dict[str, Any]]]:
    candidates = get_model_candidates(X_train.columns.tolist())
    results: list[dict[str, Any]] = []

    for model_name, pipeline in candidates.items():
        results.append(evaluate_single_model(model_name, pipeline, X_train, X_test, y_train, y_test))

    comparison_rows = [
        {
            "model_name": item["model_name"],
            "accuracy": item["accuracy"],
            "precision": item["precision"],
            "recall": item["recall"],
            "f1": item["f1"],
            "roc_auc": item["roc_auc"],
        }
        for item in results
    ]

    comparison_df = pd.DataFrame(comparison_rows).sort_values(
        by=["f1", "roc_auc", "recall", "precision"],
        ascending=False,
        na_position="last",
    )
    best_model_name = comparison_df.iloc[0]["model_name"]
    best_result = next(item for item in results if item["model_name"] == best_model_name)
    return best_result, comparison_df, results


def save_best_model_diagnostics(
    best_result: dict[str, Any],
    output_dir: Path = PLOTS_DIR,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    confusion = np.array(best_result["confusion_matrix"])
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        confusion,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Predicted On-time", "Predicted Delayed"],
        yticklabels=["Actual On-time", "Actual Delayed"],
    )
    plt.title(f"{best_result['model_name']} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_dir / "best_model_confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()

    probabilities = best_result["y_test_probabilities"]
    y_test = best_result["y_test_actual"]
    if len(np.unique(y_test)) > 1:
        fpr, tpr, _ = roc_curve(y_test, probabilities)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f"ROC-AUC = {best_result['roc_auc']:.4f}")
        plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{best_result['model_name']} ROC Curve")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(output_dir / "best_model_roc_curve.png", dpi=300, bbox_inches="tight")
        plt.close()


def get_feature_importance(best_pipeline: Pipeline) -> pd.DataFrame:
    preprocessor = best_pipeline.named_steps["preprocessor"]
    estimator = best_pipeline.named_steps["model"]
    transformed_feature_names = [
        strip_feature_prefix(name) for name in preprocessor.get_feature_names_out()
    ]

    if hasattr(estimator, "feature_importances_"):
        importances = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        importances = np.abs(estimator.coef_).ravel()
    else:
        raise AttributeError("The selected model does not expose feature importance values.")

    feature_importance_df = pd.DataFrame(
        {
            "feature": transformed_feature_names,
            "importance": importances,
        }
    ).sort_values(by="importance", ascending=False, ignore_index=True)
    return feature_importance_df


def save_feature_importance_outputs(
    feature_importance_df: pd.DataFrame,
    best_pipeline: Pipeline,
    X_reference: pd.DataFrame,
    output_dir: Path = PLOTS_DIR,
) -> str:
    """
    Save either a SHAP summary plot or a standard feature importance chart.
    Returns the artifact type used.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    shap_path = output_dir / "shap_summary.png"
    importance_path = output_dir / "feature_importance.png"

    try:
        import shap

        preprocessor = best_pipeline.named_steps["preprocessor"]
        estimator = best_pipeline.named_steps["model"]

        X_transformed = preprocessor.transform(X_reference)
        if hasattr(X_transformed, "toarray"):
            X_transformed = X_transformed.toarray()

        sample_size = min(len(X_transformed), 250)
        X_sample = X_transformed[:sample_size]
        feature_names = [strip_feature_prefix(name) for name in preprocessor.get_feature_names_out()]

        explainer = shap.Explainer(estimator, X_sample, feature_names=feature_names)
        shap_values = explainer(X_sample)
        plt.figure()
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(shap_path, dpi=300, bbox_inches="tight")
        plt.close()
        return "shap"
    except Exception:
        top_features = feature_importance_df.head(15).sort_values("importance", ascending=True)
        plt.figure(figsize=(10, 7))
        plt.barh(top_features["feature"], top_features["importance"], color="#2b7a78")
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.title("Feature Importance")
        plt.tight_layout()
        plt.savefig(importance_path, dpi=300, bbox_inches="tight")
        plt.close()
        return "feature_importance"
