# SmartProcure AI

SmartProcure AI is a competition-ready machine learning project for the challenge **"Smart Procurement: Predicting Delivery Delays & Reward-Based Planning Optimization."** It combines delay prediction, interpretable risk analysis, delivery prioritization, and a practical reward-based planning layer for procurement and logistics teams.

## Problem Statement

Procurement teams need to know which deliveries are likely to slip before those delays disrupt downstream projects. This solution predicts `delay_flag`, explains the main causes behind late deliveries, ranks which deliveries should receive scarce operational attention first, and adds a reward-based planning heuristic for bonus optimization points.

## Business Objective

The project is designed to help operations teams:

- detect delay risk early enough to intervene
- prioritize limited dispatch capacity toward the most business-critical deliveries
- reduce downstream project disruption
- allocate backup routes and resources more effectively
- support planning decisions with interpretable, auditable model signals

## Dataset Description

The project uses four challenge datasets:

- `Factories.csv`: factory capacity, production variability, and storage limits
- `Projects.csv`: project demand and priority levels
- `Deliveries.csv`: delivery events, expected time, actual time, and `delay_flag`
- `External_Factors.csv`: daily weather and traffic severity

The pipeline merges the data as follows:

- `Deliveries.csv` + `Factories.csv` on `factory_id`
- `Deliveries.csv` + `Projects.csv` on `project_id`
- `Deliveries.csv` + `External_Factors.csv` on `date`

## Approach

1. Load and inspect all source files automatically.
2. Rename conflicting coordinate columns to `factory_latitude`, `factory_longitude`, `project_latitude`, and `project_longitude`.
3. Merge and clean the full delivery context table.
4. Generate EDA plots and business insights.
5. Engineer planning-time features only.
6. Train multiple classifiers and compare them on F1-score and ROC-AUC.
7. Save predictions, priority recommendations, and reward-optimized plans.
8. Expose the outputs through a Streamlit dashboard.

## Feature Engineering

Core model features include:

- `distance_km`
- `expected_time_hours`
- `weather_index`
- `traffic_index`
- `demand`
- `priority_level_encoded`
- `base_production_per_week`
- `production_variability`
- `max_storage`

Additional engineered features:

- `traffic_weather_risk`
- `distance_time_ratio`
- `demand_to_storage_ratio`
- `production_risk`
- `is_high_priority`
- `is_medium_priority`
- `day_of_week`
- `month`

Important leakage guard:

- `actual_time_hours` is **not** used for model training.
- It is used only for analysis, delay-hour calculation, and reward scoring.

## Models Used

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- XGBoost when installed

## Evaluation Metrics

The project evaluates each model on:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix
- Classification Report

Model selection prioritizes **F1-score and ROC-AUC**, not accuracy alone.

## Results

Current pipeline run results:

- Best model: `Random Forest`
- Accuracy: `0.9958`
- Precision: `0.9958`
- Recall: `1.0000`
- F1-score: `0.9979`
- ROC-AUC: `0.9833`

Important context:

- The dataset is extremely imbalanced, with `1196` delayed deliveries and only `4` on-time deliveries.
- Because of that, headline accuracy is less informative than probability ranking and operational prioritization quality.

## Feature Importance Insights

Top delay drivers from the winning model:

1. `traffic_index`
2. `traffic_weather_risk`
3. `day_of_week`
4. `weather_index`
5. `distance_time_ratio`

Interpretation:

- traffic congestion is the strongest direct delay indicator
- the combined traffic-weather interaction materially increases risk
- some delivery days are structurally more delay-prone
- severe weather still matters even after traffic is considered
- route intensity relative to planned time captures route difficulty

## Prioritization Strategy

Each delivery receives a `priority_score` built from:

- predicted delay probability
- project priority level
- demand
- combined traffic-weather risk
- distance

Higher scores indicate deliveries that should be handled earlier. The system also assigns a `recommended_action`, such as:

- immediate action with fastest route and backup resources
- high-risk dispatch with close monitoring
- early scheduling for priority projects
- alternate route or alternate timing for high external risk

In the current run, the top-ranked deliveries cluster around **2026-04-04**, when both weather and traffic conditions are especially severe.

## Reward-Based Optimization

The bonus optimization layer adds:

- +10 for predicted on-time deliveries
- -15 for predicted delayed deliveries
- +5 for high-priority deliveries
- +3 for medium-priority deliveries
- -2 per excess delay hour
- -5 extra if a high-priority delivery is predicted delayed

The resulting `optimized_planning_score` combines business urgency with expected operational reward. Simulation results are saved for daily top-`K` capacity scenarios (`5`, `10`, and `20` deliveries per day).

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the full pipeline:

```bash
python src/train_model.py
```

Launch the dashboard:

```bash
streamlit run app.py
```

## Folder Structure

```text
smart-procurement-ai/
├── app.py
├── README.md
├── report.md
├── requirements.txt
├── data/
├── notebooks/
├── outputs/
│   ├── cleaned_merged_data.csv
│   ├── trained_model.pkl
│   ├── model_metrics.json
│   ├── model_comparison.csv
│   ├── feature_importance.csv
│   ├── delivery_delay_predictions.csv
│   ├── delivery_priority_recommendations.csv
│   ├── reward_optimized_plan.csv
│   ├── reward_simulation_results.csv
│   └── plots/
└── src/
```

## Future Improvements

- collect more on-time examples to reduce class imbalance risk
- add richer route, vehicle, supplier, and dispatch-time features
- calibrate prediction thresholds for intervention budgets
- compare heuristic prioritization with linear programming or RL scheduling
- introduce cost-sensitive evaluation tied to project disruption cost
