# Challenge Report: SmartProcure AI

## 1. Executive Summary

SmartProcure AI was built to predict delivery delays, explain the operational causes of delay risk, prioritize which deliveries should receive limited planning attention, and add a reward-based optimization layer for dispatch decision support. The project delivers a practical end-to-end workflow in Python with a trained model, EDA, ranked delivery recommendations, reward simulation, and a Streamlit dashboard.

## 2. Problem Understanding

The challenge is a binary classification problem with target column `delay_flag`:

- `0` = on-time delivery
- `1` = delayed delivery

The core planning question is not just whether a delivery will be delayed, but which deliveries deserve intervention first when time, routes, and backup resources are limited.

## 3. Data Sources

The solution combines four datasets:

- `Factories.csv`: source capacity and stability signals
- `Projects.csv`: destination project urgency and demand
- `Deliveries.csv`: delivery-level movement records and target label
- `External_Factors.csv`: daily weather and traffic conditions

Merged dataset rules:

- deliveries to factories by `factory_id`
- deliveries to projects by `project_id`
- deliveries to external conditions by `date`

## 4. Data Preparation

Preparation steps included:

- automatic source file inspection
- date parsing before merge
- explicit renaming of factory and project coordinate columns
- duplicate checks
- numeric missing value handling with median
- categorical missing value handling with mode
- integer enforcement for `delay_flag`

The merged dataset contains `1,200` deliveries across `5` factories and `199` projects present in the delivery history. No missing values or duplicate merged rows remained after cleaning.

## 5. Model Development

The solution engineered planning-time features only and excluded `actual_time_hours` from model training to prevent leakage.

Engineered features:

- `traffic_weather_risk`
- `distance_time_ratio`
- `demand_to_storage_ratio`
- `production_risk`
- `priority_level_encoded`
- `is_high_priority`
- `is_medium_priority`
- `day_of_week`
- `month`

Models compared:

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- XGBoost

Train/test configuration:

- 80/20 split
- `random_state=42`
- stratified by `delay_flag`

## 6. Model Evaluation

Best model: **Random Forest**

Performance:

- Accuracy: `0.9958`
- Precision: `0.9958`
- Recall: `1.0000`
- F1-score: `0.9979`
- ROC-AUC: `0.9833`

Important evaluation caveat:

- The dataset is extremely imbalanced: `1196` delayed deliveries vs `4` on-time deliveries.
- The test split contains only one on-time example, so the main value of the model is strong delay-risk ranking rather than reliable minority-class generalization.

Model comparison showed Random Forest and XGBoost tied on F1-score, with Random Forest selected because it delivered the stronger ROC-AUC.

## 7. Key Delay Factors

Top signals from the winning model:

1. `traffic_index`
2. `traffic_weather_risk`
3. `day_of_week`
4. `weather_index`
5. `distance_time_ratio`

Interpretation:

- congestion is the single strongest operational risk factor
- the interaction between congestion and bad weather compounds delay exposure
- some weekdays appear structurally harder for delivery execution
- route intensity relative to expected delivery time is an important route-difficulty proxy
- production stability contributes, but it is secondary to external disruption

EDA findings also showed:

- higher-traffic days are more delay-prone than lower-traffic days
- more severe weather days are more delay-prone than milder days
- factory `F1` is the most delay-prone in this sample
- `demand` is constant in the provided data, so it does not provide meaningful separation here

## 8. Delivery Prioritization Logic

The prioritization framework converts model output into an operational ranking.

Priority score formula:

```text
priority_score =
    0.35 * predicted_delay_probability
  + 0.25 * priority_level_score_normalized
  + 0.20 * demand_normalized
  + 0.15 * traffic_weather_risk_normalized
  + 0.05 * distance_normalized
```

Recommended actions are assigned through business rules:

- immediate action for very high-risk, high-priority deliveries
- high-risk dispatch with close monitoring
- early scheduling for priority projects
- alternate route or timing for high external-risk scenarios

The top-ranked deliveries in the current run are concentrated on `2026-04-04`, a day with `weather_index = 0.97` and `traffic_index = 1.00`, which makes the prioritization output intuitive and business-relevant.

## 9. Reward-Based Optimization

The reward layer adds a simple heuristic scoring system:

- +10 for predicted on-time delivery
- -15 for predicted delayed delivery
- +5 for high-priority project service
- +3 for medium-priority project service
- -2 per excess delay hour
- -5 extra penalty for predicted-delayed high-priority deliveries

The final planning signal is:

```text
optimized_planning_score = priority_score + normalized_reward_score
```

Daily top-`K` simulation results:

- `K = 5`: expected reward `-3068.00`, high-priority deliveries served `127`
- `K = 10`: expected reward `-6734.28`, high-priority deliveries served `215`
- `K = 20`: expected reward `-14886.38`, high-priority deliveries served `330`

The absolute reward values are negative because almost every delivery is predicted as delayed and the penalty structure is intentionally harsh. The score still works well as a relative ranking signal for choosing the best limited set of interventions.

## 10. Business Impact

This solution supports:

- early delay detection before dispatch
- better allocation of scarce routing and backup resources
- reduced disruption for high-priority projects
- more disciplined, explainable planning decisions
- a practical framework for daily intervention selection

## 11. Limitations

- severe target imbalance limits confidence in on-time detection
- only one month of external-factor history is available
- demand is constant in the source data and contributes little signal
- no route, supplier, vehicle, staffing, or dispatch-hour features are available
- current reward optimization is heuristic, not a full optimization solver

## 12. Future Scope

- collect more balanced historical delivery outcomes
- add richer route and operational features
- calibrate intervention thresholds by budget and project criticality
- evaluate cost-sensitive learning and probability calibration
- compare heuristic ranking against optimization or reinforcement learning approaches
