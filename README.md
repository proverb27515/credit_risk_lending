# Credit Risk Prediction — Lending Club (2007–2018)

Predicting consumer loan default on **1.34 million real loan records** using interpretable machine learning.  
**Stack**: Python · XGBoost · LightGBM · Optuna · SHAP · Scikit-learn · Pandas · Matplotlib

---

## Business Context

The 2008 Global Financial Crisis demonstrated the systemic cost of mispriced credit risk. In its aftermath, regulators (Basel III, Dodd-Frank) and lenders alike shifted toward data-driven underwriting — replacing static rules with models that can price risk at the individual borrower level.

This project builds a **pre-origination credit default model** using Lending Club's public loan data. Every feature used is available *at the time of application* — no post-disbursement payment behavior is used, ensuring the model is deployment-realistic.

The model addresses three practical goals:
1. **Discriminate** good borrowers from bad before a loan is issued
2. **Interpret** predictions using SHAP to satisfy regulatory explainability requirements
3. **Validate** model logic against established economic theory — confirming the model has learned real credit risk structure, not statistical artifacts

---

## Dataset

| Property | Value |
|---|---|
| Source | [Lending Club Loan Data — Kaggle](https://www.kaggle.com/datasets/wordsforthewise/lending-club) |
| Period | 2007 Q1 – 2018 Q4 |
| Raw rows | ~1.8M loans, 151 features |
| After filtering | Closed loans only (Fully Paid + Charged Off): **1,345,310 loans** |
| Default rate | 19.96% — moderate class imbalance |
| Target | `1` = Charged Off (default), `0` = Fully Paid |

---

## EDA: What Drives Default?

### The Class Imbalance Problem

![Class Distribution](fig_01_class_distribution.png)

~20% of closed loans were charged off. This moderate imbalance guided key modeling decisions: `scale_pos_weight` in XGBoost, `class_weight='balanced'` in LightGBM, and the use of **ROC-AUC and Average Precision** — rather than accuracy — as evaluation metrics (accuracy is misleading when classes are unequal).

---

### Default Rate Across Economic Cycles

![Default Rate Over Time](fig_02_default_rate_over_time.png)

Default rates peaked sharply in **2007–2009** during the Global Financial Crisis, then declined as Lending Club tightened underwriting post-crisis. Loan volume grew dramatically from 2012 onward as the platform scaled.

The cyclical pattern carries an important modeling lesson: a credit model trained only on boom-period data will underestimate systemic risk during downturns. Our dataset spans a **full economic cycle** — including the crisis period — giving the model exposure to both stressed and benign credit environments, producing a more robust estimator.

---

### Lending Club's Grade System: Useful, But Insufficient

![Grade Analysis](fig_03_grade_analysis.png)

Lending Club's internal grade system (A = lowest risk → G = highest) correctly ranks default rates from ~5% (Grade A) to ~35% (Grade G). The monotonic increase validates the grade system's directional accuracy.

However, even Grade A carries a ~5% default rate — and Grade B (~10%) still represents substantial risk for a lender at scale. This demonstrates that **the grade system alone is insufficient** for fine-grained risk pricing. A borrower-level model that captures within-grade variation is exactly where machine learning adds value over traditional scorecard approaches.

---

### Loan Purpose as a Structural Risk Signal

![Purpose vs Default Rate](fig_05_purpose_default.png)

Loan purpose is a strong categorical risk signal:

- **Small business loans** carry the highest default rate (~27%), consistent with U.S. SBA data showing roughly 50% of small businesses fail within 5 years. Credit extended to businesses inherits the business's survival risk.
- **Debt consolidation** — the most common purpose — sits near the average, reflecting a heterogeneous pool of borrowers.
- **Credit card refinancing** borrowers show relatively lower risk, likely due to self-selection: only credit-aware borrowers seek to consolidate high-rate balances at lower rates.

This pattern illustrates **adverse selection by loan purpose**: borrowers with the most urgent need for credit (small business, medical) are systematically higher risk.

---

## Modeling

### Why These Three Models

Three models were selected to represent a deliberate progression from interpretability to predictive power:

| Model | Rationale |
|---|---|
| **Logistic Regression** | The traditional credit scorecard model. Interpretable, auditable, and preferred by regulators under Basel II. Sets a principled performance baseline — any gain from more complex models must be justified against this floor. |
| **XGBoost** | Captures non-linear risk interactions that logistic regression cannot (e.g., DTI risk is convex — low DTI is safe, but risk accelerates disproportionately above a threshold). Handles missing values natively. Industry standard for tabular credit data. |
| **LightGBM** | Leaf-wise splitting is 3–5× faster than XGBoost on datasets with 1M+ rows, with comparable AUC. In production, model training and retraining frequency are engineering constraints — LightGBM's speed advantage is a real operational consideration, not just benchmark performance. |

### Hyperparameter Tuning with Optuna

Hyperparameters were optimized using **Optuna** (Bayesian optimization, TPE sampler) rather than manual selection or exhaustive grid search — which would be computationally prohibitive on 1.34M records.

| Decision | Rationale |
|---|---|
| Search on 10% subsample (~107K loans) | Reduces search time from hours to minutes while preserving the training distribution |
| 50 trials per model | TPE sampler converges efficiently; diminishing returns beyond 50 trials at this scale |
| Retrain on full data | Best parameters from subsample search applied to the full 1.07M-loan training set for final evaluation |

**Parameters tuned**: `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda` + model-specific: `min_child_weight`, `gamma` (XGBoost); `num_leaves`, `min_child_samples` (LightGBM).

### Results

Evaluated on a held-out test set of **269,062 loans** (20%, stratified).

| Model | ROC-AUC | Avg Precision |
|---|---|---|
| Logistic Regression (baseline) | 0.7534 | 0.5122 |
| XGBoost + Optuna | 0.7626 | 0.5257 |
| **LightGBM + Optuna** | **0.7663** | **0.5304** |

**KS Statistic (LightGBM): 0.3802**

**Interpreting the numbers:**

- **LR → tree model gap (+0.013 AUC)**: Confirms that credit default has non-linear structure. DTI and FICO interact with other variables in ways a linear model cannot capture.
- **XGBoost vs LightGBM (0.004 AUC)**: Nearly identical predictive power. At this margin, **LightGBM's 3–5× training speed** is the decisive factor for any production deployment requiring frequent retraining.
- **KS = 0.3802**: Approaches the industry benchmark of 0.40. The remaining gap suggests that additional data sources — bank transaction history, employment verification, rent payment records — could close it. This is a known limitation of public-source credit data.

![ROC Curves](fig_10_roc_curves.png)

The ROC curve shows that tree models meaningfully outperform logistic regression across all operating thresholds, not just at a single cutoff — indicating robust, structurally superior discrimination rather than threshold-specific overfitting.

![Score Distribution](fig_17_score_distribution.png)

The score distribution confirms the model's separation power: charged-off loans (red) are assigned higher predicted default probabilities than fully-paid loans (green). The overlap region defines the irreducible error — borrowers whose observable characteristics are similar regardless of outcome.

---

## Business & Economic Insights

SHAP (SHapley Additive exPlanations) decomposes each prediction into the contribution of individual features, satisfying the explainability requirements that regulators increasingly demand for credit decisions.

![SHAP Beeswarm](fig_14_shap_beeswarm.png)

### Interest Rate: Adverse Selection in Action

Interest rate is the strongest default predictor in the model — but the mechanism is not direct causation. Lending Club **sets rates based on perceived borrower risk**, so interest rate already encodes the platform's own risk assessment. High-rate borrowers also face higher monthly burdens, increasing cash flow stress.

This reflects a classic **adverse selection loop** (Stiglitz & Weiss, 1981): riskier borrowers are willing to accept high-rate loans that lower-risk borrowers would reject, confirming their riskiness, which drives rates even higher. The model correctly learns this signal, but practitioners should be aware that using interest rate as a feature creates a dependency on Lending Club's internal rating — which may not be available in an independent credit model.

### FICO Score: Validated Against Regulatory Credit Tiers

FICO has a strongly negative SHAP effect across all borrowers — higher scores reduce default probability monotonically. Crucially, the model's behavior aligns with established regulatory tiers:

- **FICO < 620** (subprime): Sharp SHAP spike — highest marginal risk
- **FICO 620–679** (near-prime): Transitional, moderate risk
- **FICO ≥ 720** (prime): Near-zero or negative SHAP contribution

This alignment with regulatory definitions is an important model validation signal: the model has not just fit statistical patterns but has learned **economically meaningful credit risk structure**.

### DTI: Empirical Support for the Dodd-Frank Threshold

Debt-to-Income ratio is a top-3 positive risk driver. Borrowers with high DTI are stretched thin — each additional dollar of loan commitment increases the probability of cash flow failure.

Under **Dodd-Frank's Ability-to-Repay rule**, DTI > 43% is the regulatory threshold for "qualified mortgage" status. Our SHAP analysis confirms that the same threshold region is where DTI contributions to default risk accelerate — providing empirical validation of regulatory intuition using market data rather than prescribed rules.

### Sub-Grade: The Value of Granularity

Lending Club's sub-grade (A1–G5, representing 35 risk buckets) carries substantial SHAP values beyond what the main grade captures. This means the **finest level of Lending Club's internal grading contains real predictive signal** — lenders who price only at the grade level are leaving risk information on the table.

### Credit History Length: A Survivorship Effect

Longer credit history reduces default probability. The economic mechanism is a **survivorship effect**: borrowers who have maintained credit accounts for many years without defaulting have already passed an implicit endurance test. The model correctly interprets credit history length as a proxy for demonstrated financial discipline, not merely as a demographic variable.

---

## Feature Engineering

Raw features were cleaned and augmented with five engineered variables that encode economic relationships:

| Feature | Construction | Economic rationale |
|---|---|---|
| `loan_to_income` | `loan_amnt / (annual_inc + 1)` | Leverage ratio — how large is this obligation relative to annual earnings |
| `installment_to_income` | `installment / (annual_inc / 12 + 1)` | Monthly cash flow burden — a more direct measure of repayment stress than DTI alone |
| `credit_history_months` | Months from `earliest_cr_line` to `issue_d` | Duration of demonstrated credit management |
| `term_months` | Parsed from `term` string | Numeric loan duration (36 or 60 months) |
| `emp_length_yrs` | Parsed from `emp_length` string | Income stability proxy |

**Dropped features**: 18 post-origination columns (payment history, recovery amounts) to prevent target leakage; 40+ joint application fields (>50% missing); free-text fields (url, desc, emp_title).

---

## Methodology

```
Raw Data: 1.8M rows, 151 features
    ↓ Filter: keep Fully Paid + Charged Off → 1,345,310 loans
    ↓ Drop: leakage columns, >50% missing fields, free-text
    ↓ Engineer: 5 new features (loan_to_income, installment_to_income, etc.)
    ↓ Encode: 12 categorical features via Label Encoding
    ↓ Impute: remaining NaN → column median
    ↓ Split: 80/20 stratified train/test
    ↓
    Logistic Regression (balanced weights) → baseline AUC: 0.7534
    Optuna (50 trials, TPE, 10% subsample) → XGBoost best params
    Optuna (50 trials, TPE, 10% subsample) → LightGBM best params
    ↓ Retrain both on full training set with best params
    ↓
    SHAP TreeExplainer → global importance (beeswarm) + dependence plots
    KS Statistic → banking-standard score separation metric
```

---

## How to Run

```bash
# 1. Clone the repo
git clone https://github.com/proverb27515/credit-risk-lending-club.git
cd credit-risk-lending-club

# 2. Install dependencies (Anaconda recommended on Apple Silicon)
pip install -r requirements.txt

# 3. Download dataset from Kaggle and place in project root
#    File: accepted_2007_to_2018Q4.csv.gz
#    https://www.kaggle.com/datasets/wordsforthewise/lending-club

# 4. Run notebooks in order
jupyter notebook
# → 1_eda.ipynb        (EDA + visualizations)
# → 2_modeling.ipynb   (Feature engineering + Optuna + SHAP)
```

> **Apple Silicon (M1/M2) note**: XGBoost and LightGBM require `libomp`. Use Anaconda Python with the `anaconda-m1` kernel — both libraries work natively on arm64.

---

## Skills Demonstrated

`Machine Learning` · `Credit Risk Modeling` · `Hyperparameter Optimization (Optuna)` · `SHAP Interpretability` · `Feature Engineering` · `Class Imbalance Handling` · `Economic Theory Application` · `Data Visualization` · `Python` · `XGBoost` · `LightGBM` · `Pandas` · `Scikit-learn`
