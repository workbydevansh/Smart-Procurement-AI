from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
PLOTS_DIR = OUTPUT_DIR / "plots"


def ensure_project_dirs() -> None:
    """Create the standard project folders when they are missing."""
    for directory in (DATA_DIR, NOTEBOOKS_DIR, OUTPUT_DIR, PLOTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def _normalize_name(name: str) -> str:
    return "".join(character.lower() for character in name if character.isalnum())


def resolve_data_file(data_dir: Path, expected_name: str) -> Path:
    """
    Locate a dataset file even when there are minor spacing or punctuation differences.
    """
    expected_path = data_dir / expected_name
    if expected_path.exists():
        return expected_path

    expected_stem_key = _normalize_name(Path(expected_name).stem)
    expected_suffix = Path(expected_name).suffix.lower()

    candidates = []
    for candidate in data_dir.iterdir():
        if not candidate.is_file():
            continue
        if expected_suffix and candidate.suffix.lower() != expected_suffix:
            continue
        candidate_key = _normalize_name(candidate.stem)
        if candidate_key == expected_stem_key:
            return candidate
        if expected_stem_key in candidate_key or candidate_key in expected_stem_key:
            candidates.append(candidate)

    if len(candidates) == 1:
        return candidates[0]

    available_files = sorted(path.name for path in data_dir.iterdir() if path.is_file())
    raise FileNotFoundError(
        f"Could not locate '{expected_name}' inside {data_dir}. Available files: {available_files}"
    )


def save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def _default_serializer(value: Any) -> Any:
        if isinstance(value, (np.integer, np.floating)):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
        return str(value)

    with path.open("w", encoding="utf-8") as file_pointer:
        json.dump(data, file_pointer, indent=2, default=_default_serializer)


def min_max_normalize(values: pd.Series) -> pd.Series:
    numeric_values = pd.to_numeric(values, errors="coerce")
    minimum = numeric_values.min()
    maximum = numeric_values.max()
    if pd.isna(minimum) or pd.isna(maximum) or np.isclose(minimum, maximum):
        return pd.Series(np.zeros(len(numeric_values)), index=values.index, dtype="float64")
    return (numeric_values - minimum) / (maximum - minimum)


def safe_ratio(
    numerator: pd.Series,
    denominator: pd.Series,
    *,
    fill_strategy: str = "median",
    fallback_value: float = 0.0,
) -> pd.Series:
    denominator = pd.to_numeric(denominator, errors="coerce").replace(0, np.nan)
    ratio = pd.to_numeric(numerator, errors="coerce") / denominator
    ratio = ratio.replace([np.inf, -np.inf], np.nan)

    if fill_strategy == "median":
        fill_value = ratio.median()
        if pd.isna(fill_value):
            fill_value = fallback_value
    else:
        fill_value = fallback_value

    return ratio.fillna(fill_value)


def strip_feature_prefix(feature_name: str) -> str:
    return feature_name.split("__", 1)[-1]


def format_metric(value: float) -> str:
    return f"{value:.4f}"
