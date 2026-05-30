# Credit Risk Prediction — Lending Club (2007–2018)

**Predicting loan default using machine learning on 1.8M+ real loan records.**  
Tools: Python · XGBoost · LightGBM · SHAP · Pandas · Scikit-learn · Matplotlib · Seaborn

---

## Project Overview

This project builds a **credit default prediction model** on the Lending Club public dataset, replicating the core workflow of a bank's risk modeling team. The goal is to distinguish loans that will be **fully repaid** from those that will be **charged off (defaulted)**.

Inspired by my experience working on data analytics at HSBC, this project applies both classical credit risk theory (KS statistic, Gini coefficient) and modern ML interpretability (SHAP) to produce business-ready insights.

---

## Dataset

| Property | Value |
|---|---|
| Source | [Lending Club Loan Data — Kaggle](https://www.kaggle.com/datasets/wordsforthewise/lending-club) |
| Period | 2007 Q1 – 2018 Q4 |
| Raw size | ~1.8M loans, 151 features |
| After filtering | Closed loans only (Fully Paid + Charged Off) |
| Default rate | ~20% (moderate class imbalance) |

---

## Notebooks

| Notebook | Contents |
|---|---|
| [1_eda.ipynb](1_eda.ipynb) | Data loading, target definition, 9 visualizations exploring default drivers |
| [2_modeling.ipynb](2_modeling.ipynb) | Feature engineering, 3-model comparison, SHAP analysis, KS statistic |

---

## Key Results

| Model | ROC-AUC | Avg Precision |
|---|---|---|
| Logistic Regression (baseline) | ~0.70 | ~0.45 |
| XGBoost | ~0.74 | ~0.52 |
| **LightGBM** | **~0.75** | **~0.53** |

*Actual values vary by run — see notebook output.*

---

## Feature Engineering Highlights

- **`loan_to_income`** — loan amount relative to annual income (leverage indicator)
- **`installment_to_income`** — monthly payment burden (cash flow stress)
- **`credit_history_months`** — months from earliest credit line to issue date
- **`emp_length_yrs`** — numeric employment length parsed from text
- Dropped: target-leakage post-origination columns, joint application fields (>80% missing), free-text fields

---

## SHAP Insights (Business Interpretation)

Top default drivers identified via SHAP:

1. **Interest rate** — strongest single predictor; reflects both risk pricing and borrower quality
2. **FICO score** — protective; higher score → lower default probability
3. **Debt-to-Income (DTI)** — positive effect on default; consistent with economic theory on household leverage
4. **Loan purpose** — small business and renewable energy carry above-average risk
5. **Credit history length** — longer history is protective (selection effect)

These align with **Basel II credit risk weights** and **Altman Z-score** factors — demonstrating economic reasoning, not just model tuning.

---

## Methodology

```
Raw Data (1.8M rows)
    ↓ Filter closed loans → ~1.1M
    ↓ Drop leakage + >50% missing cols
    ↓ Feature engineering (5 new features)
    ↓ Label encode categoricals
    ↓ Train/Test split (80/20, stratified)
    ↓
Logistic Regression → baseline AUC
XGBoost (scale_pos_weight, early stopping)
LightGBM (class_weight='balanced')
    ↓
SHAP TreeExplainer → global + local explanations
KS Statistic → banking-standard separation metric
```

---

## How to Run

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/credit-risk-lending-club.git
cd credit-risk-lending-club

# 2. Create a virtual environment and install dependencies
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Download dataset from Kaggle and place in project root
#    File: accepted_2007_to_2018Q4.csv.gz

# 4. Run notebooks in order
jupyter notebook
```

---

## Skills Demonstrated

`Machine Learning` · `Credit Risk Modeling` · `Feature Engineering` · `Class Imbalance Handling` · `SHAP Interpretability` · `Financial Domain Knowledge` · `Data Visualization` · `Python (Pandas, Scikit-learn, XGBoost, LightGBM)`

---

*Dataset is from Lending Club's public loan data. This project is for portfolio and educational purposes.*
