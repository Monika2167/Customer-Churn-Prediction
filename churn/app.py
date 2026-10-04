"""
Churn dashboard.  Run from the churn/ folder:  streamlit run app.py
(Run python src/churn_pipeline.py and python src/make_plots.py first.)
"""
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
sys.path.insert(0, str(ROOT / "src"))  # so the saved model can find churn_utils
from churn_utils import add_features, top_reasons  # noqa: E402

st.set_page_config(page_title="Churn Risk Dashboard", layout="wide")
st.title("Customer Churn Risk Dashboard")


@st.cache_resource
def load_bundle():
    return joblib.load(OUT / "churn_model.joblib")


bundle = load_bundle()
threshold = bundle["threshold"]
ranked = pd.read_csv(OUT / "high_risk_customers.csv")
comparison = pd.read_csv(OUT / "model_comparison.csv", index_col=0)

# --- Summary numbers -------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Customers scored", f"{len(ranked):,}")
c2.metric("Flagged high-risk", f"{int(ranked['high_risk'].sum()):,}")
c3.metric("Critical tier (top 10%)", f"{(ranked['risk_tier'] == 'Critical (top 10%)').sum():,}")
c4.metric("Alert threshold", f"{threshold:.2f}")

# --- Ranked list with filters ----------------------------------------------
st.subheader("Ranked customers")
tiers = ["Critical (top 10%)", "High (next 20%)", "Normal"]
chosen = st.multiselect("Risk tier", tiers, default=tiers[:1])
min_p = st.slider("Minimum churn probability", 0.0, 1.0, 0.0, 0.01)

view = ranked[ranked["risk_tier"].isin(chosen) & (ranked["churn_probability"] >= min_p)]
st.caption(f"{len(view):,} customers shown. Each customer was scored by a model that never saw them.")
st.dataframe(view, width="stretch", hide_index=True)
st.download_button(
    "Download this list as CSV",
    view.to_csv(index=False).encode("utf-8"),
    file_name="churn_risk_list.csv",
    mime="text/csv",
)

# --- Single customer lookup ------------------------------------------------
st.subheader("Look up an existing customer")
cid = st.text_input("Customer ID (e.g. 5150-ITWWB)").strip()
if cid:
    row = ranked[ranked["customerID"] == cid]
    if row.empty:
        st.warning("Customer ID not found.")
    else:
        r = row.iloc[0]
        st.metric("Churn probability", f"{r['churn_probability']:.1%}")
        st.write(f"**Tier:** {r['risk_tier']}")
        st.write(f"**Top reasons to review:** {r['review_reasons']}")

# --- Score a NEW customer (live model) -------------------------------------
st.subheader("Score a new customer")
YN = ["No", "Yes"]
with st.form("score_new"):
    a, b, c = st.columns(3)
    gender = a.selectbox("Gender", ["Female", "Male"])
    senior = b.selectbox("Senior citizen", YN)
    partner = c.selectbox("Partner", YN)
    dependents = a.selectbox("Dependents", YN)
    tenure = b.slider("Tenure (months)", 0, 72, 12)
    contract = c.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    internet = a.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
    phone = b.selectbox("Phone service", ["Yes", "No"])
    lines = c.selectbox("Multiple lines", ["No", "Yes", "No phone service"])
    sec = a.selectbox("Online security", ["No", "Yes", "No internet service"])
    backup = b.selectbox("Online backup", ["No", "Yes", "No internet service"])
    protect = c.selectbox("Device protection", ["No", "Yes", "No internet service"])
    support = a.selectbox("Tech support", ["No", "Yes", "No internet service"])
    tv = b.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    movies = c.selectbox("Streaming movies", ["No", "Yes", "No internet service"])
    paperless = a.selectbox("Paperless billing", ["Yes", "No"])
    payment = b.selectbox("Payment method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    monthly = c.number_input("Monthly charges", 15.0, 130.0, 70.0, 0.5)
    total = a.number_input("Total charges so far (0 = estimate as monthly x tenure)", 0.0, 10000.0, 0.0)
    submitted = st.form_submit_button("Score customer")

if submitted:
    new = pd.DataFrame([{
        "gender": gender, "SeniorCitizen": int(senior == "Yes"), "Partner": partner,
        "Dependents": dependents, "tenure": tenure, "PhoneService": phone,
        "MultipleLines": lines, "InternetService": internet, "OnlineSecurity": sec,
        "OnlineBackup": backup, "DeviceProtection": protect, "TechSupport": support,
        "StreamingTV": tv, "StreamingMovies": movies, "Contract": contract,
        "PaperlessBilling": paperless, "PaymentMethod": payment,
        "MonthlyCharges": monthly, "TotalCharges": total if total > 0 else monthly * tenure,
    }])
    feats = add_features(new)
    p = float(bundle["model"].predict_proba(feats)[0, 1])
    st.metric("Churn probability", f"{p:.1%}")
    if p >= threshold:
        st.error("HIGH RISK: above the alert threshold, review for retention.")
    else:
        st.success("Below the alert threshold.")
    reasons = top_reasons(bundle["xgb_explain"], feats)[0]
    st.write(f"**Main risk drivers:** {reasons if reasons else 'none stand out'}")

# --- Model comparison + plots ----------------------------------------------
st.subheader("Model comparison (validation set; *_weighted = class-imbalance weighting)")
st.dataframe(comparison, width="stretch")

for fname, title in [("cost_sensitivity.csv", "How the cost assumption changes the alert list (test set)"),
                     ("outlier_check.csv", "Outlier check (training data)")]:
    if (OUT / fname).exists():
        st.subheader(title)
        st.dataframe(pd.read_csv(OUT / fname), width="stretch", hide_index=True)

plots = [p for p in ["roc_curve.png", "confusion_matrix.png", "calibration_curve.png", "shap_summary.png"]
         if (OUT / p).exists()]
if plots:
    st.subheader("Plots")
    cols = st.columns(2)
    for i, p in enumerate(plots):
        cols[i % 2].image(str(OUT / p), caption=p.replace("_", " ").replace(".png", "").title())