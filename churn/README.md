# Customer Churn Prediction

Predicts which telecom customers are likely to leave, so a retention team can contact the highest-risk customers first.

## Data
IBM Telco Customer Churn dataset (Kaggle, `blastchar/telco-customer-churn`): 7,043 customers, 20 columns, churn rate 26.5%.
Churn is defined as the customer leaving within the last month (`Churn = Yes`).

## Approach
1. **Cleaning:** blank `TotalCharges` (brand-new customers) set to 0; target converted to 0/1.
2. **Features:** `num_services`, `avg_monthly_spend`, `charge_delta` (current vs. historical monthly charge), `is_month_to_month`, `is_new_customer`, `auto_pay`, `has_internet`.
3. **Models:** Logistic Regression, Random Forest, XGBoost.
4. **Split:** stratified 60/20/20 train/validation/test. Models are compared on validation; the test set is used once for the final numbers.
5. **Threshold:** chosen on validation data to minimise expected cost, assuming a missed churner costs 5x a wasted retention offer (`COST_FN`, `COST_FP` in `src/churn_pipeline.py`).
6. **Explanations:** SHAP values from XGBoost give the top reasons to review for each customer.

## Results

Validation set (threshold 0.5):

| Model | ROC-AUC | Precision | Recall | F1 | Brier |
|---|---|---|---|---|---|
| Logistic Regression | 0.839 | 0.677 | 0.532 | 0.596 | 0.137 |
| Random Forest | 0.835 | 0.670 | 0.489 | 0.566 | 0.139 |
| XGBoost | 0.836 | 0.661 | 0.505 | 0.573 | 0.140 |

Held-out test set, best model (Logistic Regression) at the cost-optimal threshold of 0.14:
ROC-AUC 0.846, recall 0.91, precision 0.44, F1 0.59, Brier 0.136.

Notes:
- The three models perform almost identically, so the simpler, more interpretable Logistic Regression was selected.
- A low threshold catches most churners (high recall) but flags many customers (about 55%). The 5:1 cost ratio drives this; with a smaller ratio the threshold rises and fewer customers are flagged. Customers are also split into risk tiers (top 10% critical, next 20% high) so a team with limited capacity can start at the top.
- The ranked list scores all customers, including those used for training, so only the test-set metrics should be quoted as performance.
- Strongest churn drivers (SHAP): month-to-month contract, short tenure, and charge-related features.

![ROC](outputs/roc_curve.png)
![Confusion matrix](outputs/confusion_matrix.png)
![SHAP](outputs/shap_summary.png)

## How to run
```bash
pip install -r ../requirements.txt
# place the dataset at data/telco_churn.csv
python src/churn_pipeline.py      # trains, evaluates, writes outputs/
python src/make_plots.py          # ROC, confusion matrix, SHAP plots
streamlit run app.py              # dashboard
```

## Outputs
- `outputs/model_comparison.csv`: validation metrics
- `outputs/high_risk_customers.csv`: ranked customers with probability, tier, and review reasons
- `outputs/churn_model.joblib`: saved model and threshold

## Limitations
- One public dataset with no time dimension, so there is no check for performance drift over time.
- Cost ratio is an assumption, not a measured business figure.
