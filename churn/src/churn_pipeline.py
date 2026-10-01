"""
Customer Churn Prediction - Telco dataset
Run from the churn/ folder:  python src/churn_pipeline.py

Dataset: WA_Fn-UseC_-Telco-Customer-Churn.csv (Kaggle: "Telco Customer Churn")
Put it in churn/data/telco_churn.csv
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "telco_churn.csv"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
# Business costs: missing a churner is assumed 5x worse than a wasted offer.
COST_FN = 5.0
COST_FP = 1.0

SERVICE_COLS = [
    "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]


# ----------------------------------------------------------------------------
# 1. Load + clean
# ----------------------------------------------------------------------------
def load_and_clean(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    # TotalCharges has blank strings for brand-new customers (tenure == 0)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    df["Churn"] = (df["Churn"].str.strip() == "Yes").astype(int)
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)
    return df


# ----------------------------------------------------------------------------
# 2. Feature engineering
# ----------------------------------------------------------------------------
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # How many add-on services the customer actually uses
    df["num_services"] = sum((df[c] == "Yes").astype(int) for c in SERVICE_COLS)
    df["has_internet"] = (df["InternetService"] != "No").astype(int)

    # Billing behaviour
    df["avg_monthly_spend"] = df["TotalCharges"] / df["tenure"].clip(lower=1)
    # Positive = currently paying more than their historical average (price hike signal)
    df["charge_delta"] = df["MonthlyCharges"] - df["avg_monthly_spend"]

    # Commitment / lifecycle
    df["is_month_to_month"] = (df["Contract"] == "Month-to-month").astype(int)
    df["is_new_customer"] = (df["tenure"] <= 6).astype(int)
    df["auto_pay"] = df["PaymentMethod"].str.contains("automatic", case=False).astype(int)
    return df


# ----------------------------------------------------------------------------
# 3. Preprocessing + models
# ----------------------------------------------------------------------------
def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    # pandas 3 stores text as "str" dtype (not "object"), so select "everything non-numeric"
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]
    return ColumnTransformer(
        [
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ]
    )


def build_models(pos_weight: float, preprocessor) -> dict:
    def pipe(clf):
        # clone the preprocessor so each model fits its own copy
        return Pipeline([("prep", clone(preprocessor)), ("clf", clf)])

    return {
        "logreg": pipe(
            LogisticRegression(max_iter=1000)
        ),
        "random_forest": pipe(
            RandomForestClassifier(
                n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=RANDOM_STATE,
            )
        ),
        "xgboost": pipe(
            XGBClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                scale_pos_weight=1,  # handles class imbalance
                eval_metric="logloss", random_state=RANDOM_STATE,
            )
        ),
    }


# ----------------------------------------------------------------------------
# 4. Evaluation helpers
# ----------------------------------------------------------------------------
def evaluate(y_true, proba, threshold=0.5) -> dict:
    pred = (proba >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, proba),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred),
        "f1": f1_score(y_true, pred),
        "brier (calibration)": brier_score_loss(y_true, proba),
    }


def best_cost_threshold(y_true, proba) -> float:
    """Pick the threshold that minimises expected business cost on validation data."""
    best_t, best_cost = 0.5, np.inf
    for t in np.linspace(0.05, 0.95, 91):
        pred = proba >= t
        fn = ((~pred) & (y_true == 1)).sum()
        fp = (pred & (y_true == 0)).sum()
        cost = COST_FN * fn + COST_FP * fp
        if cost < best_cost:
            best_t, best_cost = t, cost
    return float(best_t)


# ----------------------------------------------------------------------------
# 5. SHAP-based "review reasons" for each customer (uses the XGBoost pipeline)
# ----------------------------------------------------------------------------
def top_reasons(xgb_pipe: Pipeline, X: pd.DataFrame, k: int = 3) -> list:
    prep, clf = xgb_pipe.named_steps["prep"], xgb_pipe.named_steps["clf"]
    X_t = prep.transform(X)
    names = prep.get_feature_names_out()
    shap_vals = shap.TreeExplainer(clf).shap_values(X_t)

    reasons = []
    for row in shap_vals:
        idx = np.argsort(row)[::-1][:k]  # features pushing risk up the most
        reasons.append("; ".join(names[i].split("__", 1)[-1] for i in idx if row[i] > 0))
    return reasons


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    df = add_features(load_and_clean(DATA_PATH))
    ids = df["customerID"]
    y = df["Churn"]
    X = df.drop(columns=["customerID", "Churn"])

    print(f"Rows: {len(df)} | churn rate: {y.mean():.1%}")

    # train / validation / test = 60 / 20 / 20, stratified
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.25, stratify=y_tmp, random_state=RANDOM_STATE
    )

    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    models = build_models(pos_weight, build_preprocessor(X_train))

    # Fit + compare on validation set
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_val)[:, 1]
        results[name] = evaluate(y_val, proba)

    comparison = pd.DataFrame(results).T.round(3)
    print("\nValidation comparison (threshold = 0.5):")
    print(comparison)
    comparison.to_csv(OUT_DIR / "model_comparison.csv")

    # Choose best model by ROC-AUC, tune threshold on validation, report on test
    best_name = comparison["roc_auc"].idxmax()
    best_model = models[best_name]
    threshold = best_cost_threshold(
        y_val.values, best_model.predict_proba(X_val)[:, 1]
    )
    test_proba = best_model.predict_proba(X_test)[:, 1]
    test_metrics = evaluate(y_test, test_proba, threshold)
    print(f"\nBest model: {best_name} | cost-optimal threshold: {threshold:.2f}")
    print("Held-out TEST metrics:", {k: round(v, 3) for k, v in test_metrics.items()})

    joblib.dump({"model": best_model, "threshold": threshold}, OUT_DIR / "churn_model.joblib")

    # Ranked high-risk list for the whole customer base, with review reasons
    all_proba = best_model.predict_proba(X)[:, 1]
    ranked = pd.DataFrame(
        {
            "customerID": ids,
            "churn_probability": all_proba.round(3),
            "high_risk": all_proba >= threshold,
        }
    )
    ranked["review_reasons"] = top_reasons(models["xgboost"], X)
    ranked = ranked.sort_values("churn_probability", ascending=False)
    ranked["risk_tier"] = pd.qcut(
    ranked["churn_probability"].rank(method="first", ascending=False),
    q=[0, 0.1, 0.3, 1.0],
    labels=["Critical (top 10%)", "High (next 20%)", "Normal"],
)
    ranked.to_csv(OUT_DIR / "high_risk_customers.csv", index=False)
    print(f"\nSaved ranked list ({ranked['high_risk'].sum()} high-risk customers) to outputs/")
    print(ranked.head(10).to_string(index=False))


if __name__ == "__main__":
    main()