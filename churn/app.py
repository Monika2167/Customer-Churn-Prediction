"""
Churn dashboard.  Run from the churn/ folder:  streamlit run app.py
(Run python src/churn_pipeline.py first so outputs/ exists.)
"""
from pathlib import Path

import pandas as pd
import streamlit as st

OUT = Path(__file__).resolve().parent / "outputs"

st.set_page_config(page_title="Churn Risk Dashboard", layout="wide")
st.title("Customer Churn Risk Dashboard")

ranked = pd.read_csv(OUT / "high_risk_customers.csv")
comparison = pd.read_csv(OUT / "model_comparison.csv", index_col=0)

# --- Summary numbers -------------------------------------------------------
c1, c2, c3 = st.columns(3)
c1.metric("Customers scored", f"{len(ranked):,}")
c2.metric("Flagged high-risk", f"{int(ranked['high_risk'].sum()):,}")
c3.metric("Critical tier (top 10%)", f"{(ranked['risk_tier'] == 'Critical (top 10%)').sum():,}")

# --- Ranked list with filters ----------------------------------------------
st.subheader("Ranked customers")
tiers = ["Critical (top 10%)", "High (next 20%)", "Normal"]
chosen = st.multiselect("Risk tier", tiers, default=tiers[:2])
min_p = st.slider("Minimum churn probability", 0.0, 1.0, 0.0, 0.01)

view = ranked[ranked["risk_tier"].isin(chosen) & (ranked["churn_probability"] >= min_p)]
st.caption(f"{len(view):,} customers shown")
st.dataframe(view, width="stretch", hide_index=True)
st.download_button(
    "Download this list as CSV",
    view.to_csv(index=False).encode("utf-8"),
    file_name="churn_risk_list.csv",
    mime="text/csv",
)

# --- Single customer lookup ------------------------------------------------
st.subheader("Look up a customer")
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

# --- Model comparison + plots ----------------------------------------------
st.subheader("Model comparison (validation set)")
st.dataframe(comparison, use_container_width=True)

plots = [p for p in ["roc_curve.png", "confusion_matrix.png", "shap_summary.png"] if (OUT / p).exists()]
if plots:
    st.subheader("Plots")
    cols = st.columns(len(plots))
    for col, p in zip(cols, plots):
        col.image(str(OUT / p), caption=p.replace("_", " ").replace(".png", "").title())
