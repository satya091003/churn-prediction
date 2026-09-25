# Customer Churn Prediction & Retention System

## Business Problem
Subscription and telecom businesses lose 15–30% of customers every year, and it costs
5–7x more to acquire a new customer than to retain an existing one. Most churn only
becomes visible *after* the customer has already cancelled — by then it's too late to
act. This project builds a system that identifies **which customers are likely to
churn, why, and what the business should do about each one**, before they leave.

This is deliberately not "just a model." The output is a prioritized, explainable,
actionable customer list a retention team could use today.

## What this project does
1. **Predicts** churn probability per customer (Logistic Regression / Random Forest /
   XGBoost, trained on an imbalanced-aware pipeline).
2. **Explains** each prediction — which factors are pushing a customer toward churn
   (SHAP, with a feature-importance fallback if SHAP isn't installed).
3. **Segments** customers into Critical Risk / High Risk / Watch / Healthy.
4. **Recommends** a specific retention action per segment, and quantifies monthly
   revenue at risk.
5. **Deploys** as an interactive Streamlit dashboard a non-technical user could use.

## Project Structure
```
churn-prediction/
├── data/
│   └── generate_data.py     # synthetic dataset generator (see note below)
├── src/
│   ├── preprocessing.py     # load + clean
│   ├── features.py          # feature engineering
│   ├── train.py             # train/evaluate/save best model
│   └── explain.py           # feature importance + business risk report
├── app/
│   └── streamlit_app.py     # interactive dashboard
├── models/                  # saved model, scaler, feature list (generated)
├── reports/                 # churn risk report, feature importance (generated)
├── requirements.txt
└── README.md
```

## About the dataset
This project ships with `data/generate_data.py`, which creates a **realistic
synthetic dataset** (7,000 customers) with the same schema as the well-known
[Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
(IBM/Kaggle), plus a few added behavioral columns (support tickets, login frequency,
late payments) to make it feel closer to what a real company's internal data would
look like.

**For a stronger, more credible project, swap in the real dataset:**
1. Download it from Kaggle: search "Telco Customer Churn" (IBM dataset).
2. Save it as `data/telco_churn.csv`.
3. Everything downstream (`preprocessing.py`, `features.py`, `train.py`, `explain.py`,
   the Streamlit app) works unchanged — the columns match.

Mention clearly in your report/presentation which dataset (synthetic vs. real) you
ultimately used, and why.

## Setup
```bash
git clone <your-repo>
cd churn-prediction
pip install -r requirements.txt
```

## How to run
```bash
# 1. Generate the sample dataset (skip this if using the real Kaggle CSV)
python data/generate_data.py

# 2. Train the models and save the best one
python src/train.py

# 3. Generate the business risk report
python src/explain.py

# 4. Launch the interactive dashboard
streamlit run app/streamlit_app.py
```

## Methodology notes (for your report / viva)

**Why PR-AUC over accuracy?**
Churn is imbalanced (~20% positive class). A model that predicts "no churn" for
everyone gets 80% accuracy while being useless. PR-AUC and recall on the churn class
are the metrics that actually matter here — missing a churner (false negative) costs
the business a lost customer; a false positive just costs one unnecessary retention
email.

**Why SMOTE / class_weight?**
Without addressing imbalance, models default to predicting the majority class. SMOTE
oversamples the minority (churn) class in the training set only — never on the test
set, to avoid leakage. If `imbalanced-learn` isn't installed, the pipeline falls back
to `class_weight="balanced"`, which reweights the loss function instead.

**Why SHAP?**
A churn score alone isn't actionable — a retention agent needs to know *why* a
customer is at risk (high monthly charges? month-to-month contract? recent support
tickets?) to choose the right intervention. SHAP decomposes each prediction into
per-feature contributions.

**Business translation**
The point of this project isn't the model — it's connecting model output to a
retention strategy: segment customers by risk level, estimate revenue at stake per
segment, and recommend a differentiated action (a $60/month month-to-month customer
with 3 recent support tickets needs a different intervention than a $20/month
two-year-contract customer).

## Extending this project
- Add a cost-benefit simulation: compare cost of retention offers vs. cost of churn
  (customer lifetime value).
- Add model monitoring: track prediction drift over time as new data comes in.
- Try a survival-analysis approach (e.g., Cox proportional hazards) to predict *when*
  a customer will churn, not just *whether*.
- A/B test retention interventions and feed the outcomes back into the model.

## Tech Stack
pandas, numpy, scikit-learn, XGBoost, imbalanced-learn, SHAP, Streamlit
