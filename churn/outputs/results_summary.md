# Results summary (auto-generated)

Churn rate: 26.5% | customers: 7043

Final model: **logreg** + sigmoid calibration. Cost assumption 3:1 -> alert threshold **0.25**.

## Validation comparison (threshold 0.5)

| index | roc_auc | pr_auc | precision | recall | f1 | brier (calibration) |
|---|---|---|---|---|---|---|
| logreg | 0.839 | 0.651 | 0.678 | 0.529 | 0.595 | 0.137 |
| logreg_weighted | 0.839 | 0.651 | 0.514 | 0.778 | 0.619 | 0.166 |
| random_forest | 0.836 | 0.642 | 0.663 | 0.5 | 0.57 | 0.138 |
| random_forest_weighted | 0.836 | 0.637 | 0.529 | 0.773 | 0.628 | 0.16 |
| xgboost | 0.834 | 0.646 | 0.661 | 0.5 | 0.569 | 0.14 |
| xgboost_weighted | 0.834 | 0.637 | 0.513 | 0.749 | 0.609 | 0.162 |

## Held-out test metrics (at the alert threshold)

| index | roc_auc | pr_auc | precision | recall | f1 | brier (calibration) |
|---|---|---|---|---|---|---|
| final | 0.846 | 0.654 | 0.499 | 0.813 | 0.619 | 0.136 |

Share of test customers flagged: 43.2%. Brier before calibration 0.136, after 0.136.

## Cost-ratio sensitivity (test set)

| missed-churner : wasted-offer cost | threshold | flagged_pct | precision | recall |
|---|---|---|---|---|
| 1:1 | 0.51 | 19.7 | 0.669 | 0.497 |
| 2:1 | 0.32 | 36.4 | 0.542 | 0.743 |
| 3:1 | 0.25 | 43.2 | 0.499 | 0.813 |
| 5:1 | 0.19 | 50.6 | 0.466 | 0.888 |
| 10:1 | 0.08 | 66.8 | 0.383 | 0.963 |

## Outlier check (training data)

| index | values_outside_1.5xIQR | pct |
|---|---|---|
| tenure | 0.0 | 0.0 |
| MonthlyCharges | 0.0 | 0.0 |
| TotalCharges | 0.0 | 0.0 |
| avg_monthly_spend | 0.0 | 0.0 |
| charge_delta | 323.0 | 7.64 |