"""
Customer Churn Prediction - Telco dataset
Run from the churn/ folder:  python src/churn_pipeline.py

Dataset: WA_Fn-UseC_-Telco-Customer-Churn.csv (Kaggle: "Telco Customer Churn")
Save it as churn/data/telco_churn.csv
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from churn_utils import (
    CONTINUOUS_COLS,
    IQRClipper,
    add_features,
    load_and_clean,
    top_reasons,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "telco_churn.csv"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
# Business-cost ASSUMPTION: missing a churner costs 3x a wasted retention offer.
# Change these two numbers if your mentor gives real figures.
COST_FN = 3.0
COST_FP = 1.0
COST_RATIOS_TO_TEST = [1, 2, 3, 5, 10]  # sensitivity table


# ----------------------------------------------------------------------------
# Split (shared with make_plots.py so both use exactly the same data)
# ----------------------------------------------------------------------------
def split_data(X, y) -> dict:
    """Stratified 60 / 20 / 20 train / validation / test."""
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=0.25, stratify=y_tv, random_state=RANDOM_STATE
    )
    return dict(X_train=X_train, X_val=X_val, X_test=X_test, X_tv=X_tv,
                y_train=y_train, y_val=y_val, y_test=y_test, y_tv=y_tv)


# ----------------------------------------------------------------------------
# Preprocessing + models
# ----------------------------------------------------------------------------
def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()
    clip_cols = [c for c in CONTINUOUS_COLS if c in X.columns]
    other_num = [c for c in X.columns if c not in cat_cols and c not in clip_cols]
    return ColumnTransformer(
        [
            # continuous columns: cap outliers (IQR rule, learned on training data) then scale
            ("num_clip", Pipeline([("clip", IQRClipper()), ("scale", StandardScaler())]), clip_cols),
            ("num", StandardScaler(), other_num),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ]
    )


def build_models(pos_weight: float, preprocessor) -> dict:
    """Each model twice: plain, and with class-imbalance weighting (*_weighted)."""
    def pipe(clf):
        return Pipeline([("prep", clone(preprocessor)), ("clf", clf)])

    rf = dict(n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=RANDOM_STATE)
    xgb = dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
               colsample_bytree=0.8, eval_metric="logloss", random_state=RANDOM_STATE)
    return {
        "logreg": pipe(LogisticRegression(max_iter=1000)),
        "logreg_weighted": pipe(LogisticRegression(max_iter=1000, class_weight="balanced")),
        "random_forest": pipe(RandomForestClassifier(**rf)),
        "random_forest_weighted": pipe(RandomForestClassifier(class_weight="balanced", **rf)),
        "xgboost": pipe(XGBClassifier(**xgb)),
        "xgboost_weighted": pipe(XGBClassifier(scale_pos_weight=pos_weight, **xgb)),
    }


# ----------------------------------------------------------------------------
# Evaluation helpers
# ----------------------------------------------------------------------------
def evaluate(y_true, proba, threshold=0.5) -> dict:
    pred = (proba >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred),
        "f1": f1_score(y_true, pred),
        "brier (calibration)": brier_score_loss(y_true, proba),
    }


def best_cost_threshold(y_true, proba, cost_fn=COST_FN, cost_fp=COST_FP) -> float:
    """Threshold that minimises expected business cost."""
    best_t, best_cost = 0.5, np.inf
    for t in np.linspace(0.02, 0.95, 94):
        pred = proba >= t
        fn = ((~pred) & (y_true == 1)).sum()
        fp = (pred & (y_true == 0)).sum()
        cost = cost_fn * fn + cost_fp * fp
        if cost < best_cost:
            best_t, best_cost = t, cost
    return float(best_t)


def md_table(df: pd.DataFrame) -> str:
    """Small markdown table writer (no extra packages needed)."""
    df = df.reset_index()
    head = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "|" + "---|" * len(df.columns)
    rows = ["| " + " | ".join(str(v) for v in r) + " |" for r in df.values]
    return "\n".join([head, sep] + rows)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    df = add_features(load_and_clean(DATA_PATH))
    ids = df["customerID"]
    y = df["Churn"]
    X = df.drop(columns=["customerID", "Churn"])
    print(f"Rows: {len(df)} | churn rate: {y.mean():.1%}")

    S = split_data(X, y)

    # ---- Outlier check (training data) ----
    rows = {}
    for c in CONTINUOUS_COLS:
        v = S["X_train"][c]
        q1, q3 = v.quantile(0.25), v.quantile(0.75)
        n_out = int(((v < q1 - 1.5 * (q3 - q1)) | (v > q3 + 1.5 * (q3 - q1))).sum())
        rows[c] = {"values_outside_1.5xIQR": n_out, "pct": round(100 * n_out / len(v), 2)}
    outliers = pd.DataFrame(rows).T
    print("\nOutlier check (training data, 1.5 x IQR rule; these get capped in the model):")
    print(outliers)
    outliers.to_csv(OUT_DIR / "outlier_check.csv")

    # ---- Stage A: compare 6 model variants on validation ----
    pos_weight = (S["y_train"] == 0).sum() / (S["y_train"] == 1).sum()
    models = build_models(pos_weight, build_preprocessor(S["X_train"]))
    results = {}
    for name, model in models.items():
        model.fit(S["X_train"], S["y_train"])
        results[name] = evaluate(S["y_val"], model.predict_proba(S["X_val"])[:, 1])
    comparison = pd.DataFrame(results).T.round(3)
    print("\nValidation comparison (threshold = 0.5). *_weighted = class-imbalance weighting:")
    print(comparison)
    comparison.to_csv(OUT_DIR / "model_comparison.csv")

    best_name = comparison["roc_auc"].idxmax()
    base = models[best_name]
    print(f"\nBest model by ROC-AUC: {best_name}")

    # ---- Stage B: calibrated final model + cost-optimal threshold ----
    # Calibration makes the predicted probability trustworthy (important for a cost threshold,
    # and for class-weighted models whose raw probabilities are inflated).
    final = CalibratedClassifierCV(clone(base), method="sigmoid", cv=5)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    # Out-of-fold probabilities on train+validation: every customer is scored by a model
    # that never saw them, so the threshold is not tuned on training fits.
    oof = cross_val_predict(clone(final), S["X_tv"], S["y_tv"], cv=skf, method="predict_proba")[:, 1]
    threshold = best_cost_threshold(S["y_tv"].values, oof)

    final.fit(S["X_tv"], S["y_tv"])
    test_proba = final.predict_proba(S["X_test"])[:, 1]
    raw_test_proba = base.predict_proba(S["X_test"])[:, 1]

    test_metrics = evaluate(S["y_test"], test_proba, threshold)
    pct_flagged = float((test_proba >= threshold).mean())
    print(f"\nFinal model: {best_name} + sigmoid calibration | cost ratio {COST_FN:g}:{COST_FP:g} "
          f"| threshold {threshold:.2f}")
    print("Held-out TEST metrics:", {k: round(v, 3) for k, v in test_metrics.items()})
    print(f"Share of test customers flagged: {pct_flagged:.1%}")
    print(f"Brier score before calibration: {brier_score_loss(S['y_test'], raw_test_proba):.3f} "
          f"-> after: {brier_score_loss(S['y_test'], test_proba):.3f}")

    # ---- Cost-ratio sensitivity (how the alert list changes with the assumption) ----
    sens = {}
    for r in COST_RATIOS_TO_TEST:
        t = best_cost_threshold(S["y_tv"].values, oof, cost_fn=float(r), cost_fp=1.0)
        m = evaluate(S["y_test"], test_proba, t)
        sens[f"{r}:1"] = {
            "threshold": round(t, 2),
            "flagged_pct": round(100 * float((test_proba >= t).mean()), 1),
            "precision": round(m["precision"], 3),
            "recall": round(m["recall"], 3),
        }
    sensitivity = pd.DataFrame(sens).T
    sensitivity.index.name = "missed-churner : wasted-offer cost"
    print("\nCost-ratio sensitivity (test set):")
    print(sensitivity)
    sensitivity.to_csv(OUT_DIR / "cost_sensitivity.csv")

    # ---- Ranked list: every customer scored by a model that never saw them ----
    scores = pd.Series(index=X.index, dtype=float)
    scores.loc[S["X_tv"].index] = oof
    scores.loc[S["X_test"].index] = test_proba

    xgb_explain = models["xgboost"]  # plain XGBoost used only to explain risk drivers
    ranked = pd.DataFrame(
        {
            "customerID": ids,
            "churn_probability": scores.round(3),
            "high_risk": scores >= threshold,
        }
    )
    ranked["review_reasons"] = top_reasons(xgb_explain, X)
    ranked = ranked.sort_values("churn_probability", ascending=False)
    ranked["risk_tier"] = pd.qcut(
        ranked["churn_probability"].rank(method="first", ascending=False),
        q=[0, 0.1, 0.3, 1.0],
        labels=["Critical (top 10%)", "High (next 20%)", "Normal"],
    )
    ranked.to_csv(OUT_DIR / "high_risk_customers.csv", index=False)
    print(f"\nRanked list saved ({int(ranked['high_risk'].sum())} high-risk of {len(ranked)}).")
    print(ranked.head(10).to_string(index=False))

    # ---- Save model bundle (used by the dashboard for live scoring) ----
    joblib.dump(
        {"model": final, "threshold": threshold, "xgb_explain": xgb_explain,
         "best_base": best_name, "cost_fn": COST_FN, "cost_fp": COST_FP},
        OUT_DIR / "churn_model.joblib",
    )

    # ---- Summary file you can paste into the README ----
    summary = [
        "# Results summary (auto-generated)",
        f"\nChurn rate: {y.mean():.1%} | customers: {len(df)}",
        f"\nFinal model: **{best_name}** + sigmoid calibration. Cost assumption "
        f"{COST_FN:g}:{COST_FP:g} -> alert threshold **{threshold:.2f}**.",
        "\n## Validation comparison (threshold 0.5)\n", md_table(comparison),
        "\n## Held-out test metrics (at the alert threshold)\n",
        md_table(pd.DataFrame([{k: round(v, 3) for k, v in test_metrics.items()}])
                 .rename(index={0: "final"})),
        f"\nShare of test customers flagged: {pct_flagged:.1%}. "
        f"Brier before calibration {brier_score_loss(S['y_test'], raw_test_proba):.3f}, "
        f"after {brier_score_loss(S['y_test'], test_proba):.3f}.",
        "\n## Cost-ratio sensitivity (test set)\n", md_table(sensitivity),
        "\n## Outlier check (training data)\n", md_table(outliers),
    ]
    (OUT_DIR / "results_summary.md").write_text("\n".join(summary), encoding="utf-8")
    print("\nSaved outputs/results_summary.md (paste its tables into the README).")


if __name__ == "__main__":
    main()