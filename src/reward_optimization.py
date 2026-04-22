from __future__ import annotations

import pandas as pd

from src.utils import min_max_normalize


def build_reward_optimized_plan(priority_df: pd.DataFrame) -> pd.DataFrame:
    """
    Score deliveries using the requested reward framework and create an optimized plan.
    """
    reward_df = priority_df.copy()
    reward_df["delay_hours"] = 0.0

    if "actual_time_hours" in reward_df.columns and "expected_time_hours" in reward_df.columns:
        reward_df["delay_hours"] = (
            reward_df["actual_time_hours"] - reward_df["expected_time_hours"]
        ).clip(lower=0)

    reward_df["reward_score"] = 0.0
    reward_df.loc[reward_df["predicted_delay_class"] == 0, "reward_score"] += 10
    reward_df.loc[reward_df["predicted_delay_class"] == 1, "reward_score"] -= 15

    reward_df.loc[reward_df["priority_level"] == "High", "reward_score"] += 5
    reward_df.loc[reward_df["priority_level"] == "Medium", "reward_score"] += 3

    reward_df["reward_score"] -= 2 * reward_df["delay_hours"]
    reward_df.loc[
        (reward_df["priority_level"] == "High") & (reward_df["predicted_delay_class"] == 1),
        "reward_score",
    ] -= 5

    reward_df["normalized_reward_score"] = min_max_normalize(reward_df["reward_score"])
    reward_df["optimized_planning_score"] = (
        reward_df["priority_score"] + reward_df["normalized_reward_score"]
    )

    selected_columns = [
        "delivery_id",
        "factory_id",
        "project_id",
        "date",
        "priority_level",
        "predicted_delay_probability",
        "predicted_delay_class",
        "priority_score",
        "delay_hours",
        "reward_score",
        "optimized_planning_score",
        "recommended_action",
    ]
    return reward_df.sort_values(by="optimized_planning_score", ascending=False)[selected_columns]


def run_reward_simulation(
    reward_df: pd.DataFrame, k_values: tuple[int, ...] = (5, 10, 20)
) -> pd.DataFrame:
    """
    Simulate a daily prioritization capacity limit and aggregate expected outcomes.
    """
    simulation_rows: list[dict[str, float | int]] = []
    sorted_df = reward_df.sort_values(by=["date", "optimized_planning_score"], ascending=[True, False])

    for k_value in k_values:
        selected_deliveries = sorted_df.groupby("date", group_keys=False).head(k_value)
        simulation_rows.append(
            {
                "top_k_per_day": int(k_value),
                "selected_deliveries": int(len(selected_deliveries)),
                "total_expected_reward": float(selected_deliveries["reward_score"].sum()),
                "high_priority_deliveries_served": int(
                    (selected_deliveries["priority_level"] == "High").sum()
                ),
                "average_predicted_delay_probability": float(
                    selected_deliveries["predicted_delay_probability"].mean()
                ),
                "expected_delayed_deliveries": float(
                    selected_deliveries["predicted_delay_probability"].sum()
                ),
            }
        )

    return pd.DataFrame(simulation_rows)
