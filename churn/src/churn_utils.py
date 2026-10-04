"""
Shared helpers for the churn project.

These live in their own module (not in churn_pipeline.py) so that the saved
model can be loaded later by make_plots.py and app.py. A class defined inside a
script that is run directly cannot be unpickled from another script.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

SERVICE_COLS = [
    "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]

# Continuous columns where outliers make sense to check / cap.
# (0/1 flags and small counts are excluded: the IQR rule would flatten them.)
CONTINUOUS_COLS = ["tenure", "MonthlyCharges", "TotalCharges", "avg_monthly_spend", "charge_delta"]


class IQRClipper(BaseEstimator, TransformerMixin):
    """Caps values outside [Q1 - k*IQR, Q3 + k*IQR]. Bounds are learned on training data only."""

    def __init__(self, factor: float = 1.5):
        self.factor = factor

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        q1, q3 = np.percentile(X, [25, 75], axis=0)
        iqr = q3 - q1
        lower = q1 - self.factor * iqr
        upper = q3 + self.factor * iqr
        flat = iqr == 0  # no spread -> do not clip (would flatten the column)
        self.lower_ = np.where(flat, -np.inf, lower)
        self.upper_ = np.where(flat, np.inf, upper)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lower_, self.upper_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


def load_and_clean(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    # TotalCharges has blank strings for brand-new customers (tenure == 0)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    df["Churn"] = (df["Churn"].str.strip() == "Yes").astype(int)
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # How many add-on services the customer actually uses
    df["num_services"] = sum((df[c] == "Yes").astype(int) for c in SERVICE_COLS)
    df["has_internet"] = (df["InternetService"] != "No").astype(int)

    # Billing behaviour
    df["avg_monthly_spend"] = df["TotalCharges"] / df["tenure"].clip(lower=1)
    # Positive = currently paying more than their historical average (price-hike signal)
    df["charge_delta"] = df["MonthlyCharges"] - df["avg_monthly_spend"]

    # Commitment / lifecycle
    df["is_month_to_month"] = (df["Contract"] == "Month-to-month").astype(int)
    df["is_new_customer"] = (df["tenure"] <= 6).astype(int)
    df["auto_pay"] = df["PaymentMethod"].str.contains("automatic", case=False).astype(int)
    return df


def top_reasons(xgb_pipe, X: pd.DataFrame, k: int = 3) -> list:
    """Top-k features pushing each customer's churn risk UP (SHAP, XGBoost pipeline)."""
    import shap  # imported here so lightweight imports of this module stay fast

    prep, clf = xgb_pipe.named_steps["prep"], xgb_pipe.named_steps["clf"]
    X_t = prep.transform(X)
    names = prep.get_feature_names_out()
    shap_vals = shap.TreeExplainer(clf).shap_values(X_t)

    reasons = []
    for row in shap_vals:
        idx = np.argsort(row)[::-1][:k]
        reasons.append("; ".join(names[i].split("__", 1)[-1] for i in idx if row[i] > 0))
    return reasons