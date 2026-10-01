"""
Saves ROC curve, confusion matrix and SHAP summary plot into churn/outputs/.
Run AFTER churn_pipeline.py, from the churn/ folder:  python src/make_plots.py
"""
import joblib
import matplotlib

matplotlib.use("Agg")  # no window needed, just save files
import matplotlib.pyplot as plt
import shap
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.model_selection import train_test_split

import churn_pipeline as cp  # reuses the same cleaning, features and models

df = cp.add_features(cp.load_and_clean(cp.DATA_PATH))
y = df["Churn"]
X = df.drop(columns=["customerID", "Churn"])

# Same split as the pipeline (same random_state) so test data is truly unseen
X_tmp, X_test, y_tmp, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=cp.RANDOM_STATE
)
X_train, X_val, y_train, y_val = train_test_split(
    X_tmp, y_tmp, test_size=0.25, stratify=y_tmp, random_state=cp.RANDOM_STATE
)

pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
models = cp.build_models(pos_weight, cp.build_preprocessor(X_train))
for m in models.values():
    m.fit(X_train, y_train)

# 1) ROC curves for all three models on the test set
fig, ax = plt.subplots(figsize=(6, 5))
for name, m in models.items():
    RocCurveDisplay.from_estimator(m, X_test, y_test, ax=ax, name=name)
ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
ax.set_title("ROC curves (test set)")
fig.tight_layout()
fig.savefig(cp.OUT_DIR / "roc_curve.png", dpi=150)
plt.close(fig)

# 2) Confusion matrix for the saved best model at its cost-optimal threshold
bundle = joblib.load(cp.OUT_DIR / "churn_model.joblib")
proba = bundle["model"].predict_proba(X_test)[:, 1]
pred = (proba >= bundle["threshold"]).astype(int)
fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay.from_predictions(
    y_test, pred, display_labels=["Stay", "Churn"], ax=ax, colorbar=False
)
ax.set_title(f"Confusion matrix (threshold = {bundle['threshold']:.2f})")
fig.tight_layout()
fig.savefig(cp.OUT_DIR / "confusion_matrix.png", dpi=150)
plt.close(fig)

# 3) SHAP summary (XGBoost): which features push churn risk up or down
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

print("Saved roc_curve.png, confusion_matrix.png, shap_summary.png to outputs/")