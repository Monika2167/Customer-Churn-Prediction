"""
Saves ROC curve, confusion matrix, calibration curve and SHAP summary into churn/outputs/.
Run AFTER churn_pipeline.py, from the churn/ folder:  python src/make_plots.py
"""
import joblib
import matplotlib

matplotlib.use("Agg")  # save files, no window
import matplotlib.pyplot as plt
import pandas as pd
import shap
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

import churn_pipeline as cp  # same split, preprocessing and models as the pipeline
from churn_utils import add_features, load_and_clean

df = add_features(load_and_clean(cp.DATA_PATH))
y = df["Churn"]
X = df.drop(columns=["customerID", "Churn"])
S = cp.split_data(X, y)

pos_weight = (S["y_train"] == 0).sum() / (S["y_train"] == 1).sum()
models = cp.build_models(pos_weight, cp.build_preprocessor(S["X_train"]))
for m in models.values():
    m.fit(S["X_train"], S["y_train"])

bundle = joblib.load(cp.OUT_DIR / "churn_model.joblib")
final, thr = bundle["model"], bundle["threshold"]
best_name = bundle["best_base"]
X_test, y_test = S["X_test"], S["y_test"]
final_proba = final.predict_proba(X_test)[:, 1]

# 1) ROC curves (test set)
fig, ax = plt.subplots(figsize=(6, 5))
for name in ["logreg", "random_forest", "xgboost"]:
    RocCurveDisplay.from_estimator(models[name], X_test, y_test, ax=ax, name=name)
RocCurveDisplay.from_predictions(y_test, final_proba, ax=ax, name="final (calibrated)")
ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
ax.set_title("ROC curves (test set)")
fig.tight_layout()
fig.savefig(cp.OUT_DIR / "roc_curve.png", dpi=150)
plt.close(fig)

# 2) Confusion matrix at the cost-optimal threshold
fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay.from_predictions(
    y_test, (final_proba >= thr).astype(int),
    display_labels=["Stay", "Churn"], ax=ax, colorbar=False,
)
ax.set_title(f"Confusion matrix (threshold = {thr:.2f})")
fig.tight_layout()
fig.savefig(cp.OUT_DIR / "confusion_matrix.png", dpi=150)
plt.close(fig)

# 3) Calibration curve: predicted probability vs what really happened
fig, ax = plt.subplots(figsize=(5.5, 5))
CalibrationDisplay.from_predictions(
    y_test, models[best_name].predict_proba(X_test)[:, 1],
    n_bins=10, strategy="quantile", ax=ax, name=f"{best_name} (raw)",
)
CalibrationDisplay.from_predictions(
    y_test, final_proba, n_bins=10, strategy="quantile", ax=ax, name="final (calibrated)",
)
ax.set_title("Calibration (closer to the diagonal is better)")
fig.tight_layout()
fig.savefig(cp.OUT_DIR / "calibration_curve.png", dpi=150)
plt.close(fig)

# 4) SHAP summary (plain XGBoost): which features push churn risk up or down
xgb = models["xgboost"]
prep, clf = xgb.named_steps["prep"], xgb.named_steps["clf"]
X_t = prep.transform(X_test)
names = [n.split("__", 1)[-1] for n in prep.get_feature_names_out()]
shap_values = shap.TreeExplainer(clf).shap_values(X_t)
plt.figure()
shap.summary_plot(shap_values, X_t, feature_names=names, max_display=12, show=False)
plt.tight_layout()
plt.savefig(cp.OUT_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()

print("Saved roc_curve.png, confusion_matrix.png, calibration_curve.png, shap_summary.png to outputs/")