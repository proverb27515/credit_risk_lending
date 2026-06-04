import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Risk Predictor",
    page_icon="🏦",
    layout="wide",
)

# ── Load artifacts ─────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model    = joblib.load("lgbm_model.pkl")
    columns  = joblib.load("feature_columns.pkl")
    medians  = joblib.load("feature_medians.pkl")
    encoders = joblib.load("label_encoders.pkl")
    return model, columns, medians, encoders

model, feature_columns, feature_medians, encoders = load_artifacts()

OPT_THRESHOLD = 0.538

# ── Header ─────────────────────────────────────────────────────────────────
st.title("🏦 Credit Risk Predictor")
st.markdown(
    "Lending Club loan default probability — **LightGBM + Optuna**, "
    "trained on 1.1M loans (2007–2016), validated on 2017–2018 out-of-time holdout.  \n"
    "Decision threshold **t\\* = 0.538** optimised for expected portfolio P&L."
)
st.divider()

# ── Input form ─────────────────────────────────────────────────────────────
st.subheader("Loan Application Details")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Loan Terms**")
    loan_amnt  = st.slider("Loan Amount ($)", 500, 40000, 15000, step=500)
    int_rate   = st.slider("Interest Rate (%)", 5.0, 30.0, 13.5, step=0.25)
    term_label = st.selectbox("Term", ["36 months", "60 months"])
    purpose    = st.selectbox("Loan Purpose", sorted([
        "debt_consolidation", "credit_card", "home_improvement",
        "car", "medical", "small_business", "major_purchase",
        "moving", "vacation", "other",
    ]))

with col2:
    st.markdown("**Borrower Profile**")
    annual_inc = st.number_input("Annual Income ($)", 10_000, 500_000, 65_000, step=1_000)
    dti        = st.slider("Debt-to-Income (%)", 0.0, 50.0, 18.0, step=0.5)
    fico       = st.slider("FICO Score", 580, 850, 700, step=5)
    emp_length = st.selectbox("Employment Length", [
        "< 1 year", "1 year", "2 years", "3 years", "4 years", "5 years",
        "6 years", "7 years", "8 years", "9 years", "10+ years",
    ])

with col3:
    st.markdown("**Credit Profile**")
    grade          = st.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])
    home_ownership = st.selectbox("Home Ownership", ["RENT", "MORTGAGE", "OWN", "OTHER"])
    delinq_2yrs    = st.slider("Delinquencies (past 2 yrs)", 0, 10, 0)
    open_acc       = st.slider("Open Credit Lines", 1, 40, 10)

# ── Feature engineering ────────────────────────────────────────────────────
emp_map = {
    "< 1 year": 0, "1 year": 1, "2 years": 2, "3 years": 3,
    "4 years": 4,  "5 years": 5, "6 years": 6, "7 years": 7,
    "8 years": 8,  "9 years": 9, "10+ years": 10,
}
term_months    = 36 if "36" in term_label else 60
emp_length_yrs = emp_map[emp_length]

# Monthly payment (amortisation formula)
r = (int_rate / 100) / 12
installment = loan_amnt * r / (1 - (1 + r) ** (-term_months)) if r > 0 else loan_amnt / term_months

loan_to_income        = loan_amnt / (annual_inc + 1)
installment_to_income = installment / (annual_inc / 12 + 1)

# ── Build feature vector ───────────────────────────────────────────────────
row = {col: feature_medians.get(col, 0) for col in feature_columns}

row.update({
    "loan_amnt":             loan_amnt,
    "int_rate":              int_rate,
    "annual_inc":            annual_inc,
    "dti":                   dti,
    "fico_range_low":        fico,
    "fico_range_high":       fico + 4,
    "term_months":           term_months,
    "emp_length_yrs":        emp_length_yrs,
    "installment":           installment,
    "loan_to_income":        loan_to_income,
    "installment_to_income": installment_to_income,
    "delinq_2yrs":           delinq_2yrs,
    "open_acc":              open_acc,
})

# Categorical encoding
for col, val in [
    ("purpose",        purpose),
    ("home_ownership", home_ownership),
    ("grade",          grade),
]:
    if col in encoders:
        try:
            row[col] = int(encoders[col].transform([val])[0])
        except ValueError:
            pass  # unseen label → keep median

input_df = pd.DataFrame([row])[feature_columns]

# ── Predict ────────────────────────────────────────────────────────────────
proba    = float(model.predict_proba(input_df)[0][1])
approved = proba < OPT_THRESHOLD

st.divider()
st.subheader("Credit Decision")

res_col, gauge_col = st.columns([1, 2])

with res_col:
    if approved:
        st.success("### ✅ APPROVED")
        st.markdown(f"Default probability: **{proba:.1%}**  \nBelow threshold t\\* = {OPT_THRESHOLD}")
    else:
        st.error("### ❌ DECLINED")
        st.markdown(f"Default probability: **{proba:.1%}**  \nExceeds threshold t\\* = {OPT_THRESHOLD}")

    # Risk tier
    if proba < 0.15:
        tier, tier_color = "Low Risk", "🟢"
    elif proba < 0.25:
        tier, tier_color = "Medium Risk", "🟡"
    elif proba < 0.40:
        tier, tier_color = "High Risk", "🟠"
    else:
        tier, tier_color = "Very High Risk", "🔴"
    st.markdown(f"**Risk Tier**: {tier_color} {tier}")

with gauge_col:
    fig, ax = plt.subplots(figsize=(6, 1.2))
    bar_color = "#4CAF50" if approved else "#F44336"
    ax.barh([""], [proba], color=bar_color, height=0.5)
    ax.barh([""], [1 - proba], left=[proba], color="#E0E0E0", height=0.5)
    ax.axvline(OPT_THRESHOLD, color="black", linewidth=2, linestyle="--",
               label=f"t* = {OPT_THRESHOLD}")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Predicted Default Probability", fontsize=10)
    ax.set_title(f"Score: {proba:.1%}", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.set_yticks([])
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ── SHAP explanation ───────────────────────────────────────────────────────
st.divider()
st.subheader("Why this decision? (SHAP explanation)")

with st.spinner("Computing feature contributions…"):
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)
    if isinstance(shap_values, list):
        sv = shap_values[1][0]
    else:
        sv = shap_values[0]

    shap_df = (
        pd.DataFrame({"feature": feature_columns, "shap": sv})
        .assign(abs_shap=lambda d: d["shap"].abs())
        .sort_values("abs_shap", ascending=False)
        .head(10)
        .sort_values("shap")
    )

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    colors = ["#F44336" if v > 0 else "#4CAF50" for v in shap_df["shap"]]
    ax2.barh(shap_df["feature"], shap_df["shap"], color=colors, edgecolor="white")
    ax2.axvline(0, color="black", linewidth=0.8)
    ax2.set_xlabel("SHAP value (red = increases default risk)", fontsize=10)
    ax2.set_title("Top 10 Feature Contributions to This Prediction", fontsize=11, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

# ── Footer ─────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Model: LightGBM + Optuna (50 trials Bayesian HPO) · "
    "Train: 1,122,448 loans 2007–2016 · "
    "Test AUC: 0.7505 · KS: 0.3571 · "
    "Threshold t\\* optimised for expected portfolio P&L"
)
