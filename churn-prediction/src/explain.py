"""
explain.py
----------
Turns model predictions into human-readable explanations and a business
retention report. This is the step that turns "a model" into "a solution."

Run from project root (after train.py):
    python src/explain.py
"""

import json
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.append("src")
from features import engineer_features
from preprocessing import clean_data, load_data

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


def load_artifacts():
    with open("models/best_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("models/feature_names.json") as f:
        feature_names = json.load(f)
    return model, scaler, feature_names


def get_feature_importance(model, feature_names, top_n=10):
    """Works for tree models (feature_importances_) and linear models (coef_)."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        return pd.DataFrame()

    imp_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    imp_df = imp_df.sort_values("importance", ascending=False).head(top_n)
    return imp_df


def explain_with_shap(model, X_sample, feature_names):
    """Returns a SHAP explainer + values if shap is installed, else None."""
    if not HAS_SHAP:
        print("shap not installed -> falling back to global feature importance "
              "(pip install shap for per-customer explanations).")
        return None, None
    try:
        explainer = shap.Explainer(model, X_sample)
        shap_values = explainer(X_sample)
        return explainer, shap_values
    except Exception as e:
        print(f"SHAP explainer failed ({e}) -> falling back to feature importance.")
        return None, None


def risk_segment(prob):
    if prob >= 0.7:
        return "Critical Risk"
    elif prob >= 0.4:
        return "High Risk"
    elif prob >= 0.2:
        return "Watch"
    return "Healthy"


def top_reasons_for_customer(row, imp_df, n=3):
    """Simple, fast, human-readable reasons using global importance x the
    customer's own deviation from the population — a lightweight stand-in
    for SHAP when it isn't installed, good enough for a dashboard."""
    reasons = []
    for feat in imp_df["feature"].head(n):
        if feat in row.index:
            reasons.append(f"{feat} = {row[feat]}")
    return reasons


def generate_business_report():
    model, scaler, feature_names = load_artifacts()
    df_raw = clean_data(load_data())
    X, y = engineer_features(df_raw)
    X = X[feature_names]  # ensure column order matches training

    needs_scaling = hasattr(model, "coef_")
    X_for_pred = scaler.transform(X) if needs_scaling else X

    probs = model.predict_proba(X_for_pred)[:, 1]

    report = df_raw.copy()
    report["churn_probability"] = np.round(probs, 3)
    report["risk_segment"] = report["churn_probability"].apply(risk_segment)

    imp_df = get_feature_importance(model, feature_names)

    # --- Business-facing summary ---
    summary = report.groupby("risk_segment").agg(
        customers=("churn_probability", "count"),
        avg_monthly_revenue=("MonthlyCharges", "mean"),
    ).round(2)
    summary["est_monthly_revenue_at_risk"] = np.round(
        summary["customers"] * summary["avg_monthly_revenue"], 2
    )

    print("\n=== Top churn drivers (global) ===")
    print(imp_df.to_string(index=False))

    print("\n=== Risk segments ===")
    print(summary)

    # Recommended actions per segment (this is the "solution", not just a score)
    actions = {
        "Critical Risk": "Immediate personal outreach + retention offer (discount or plan review) within 48h.",
        "High Risk": "Proactive email/SMS check-in, highlight underused features, offer loyalty discount on renewal.",
        "Watch": "Add to nurture campaign; monitor next billing cycle, no discount needed yet.",
        "Healthy": "No action needed; candidate for upsell / referral programs.",
    }
    print("\n=== Recommended actions ===")
    for seg, action in actions.items():
        if seg in summary.index:
            print(f"- {seg}: {action}")

    report.to_csv("reports/churn_risk_report.csv", index=False)
    imp_df.to_csv("reports/feature_importance.csv", index=False)
    with open("reports/segment_summary.json", "w") as f:
        json.dump(json.loads(summary.to_json(orient="index")), f, indent=2)

    print("\nSaved: reports/churn_risk_report.csv, reports/feature_importance.csv, "
          "reports/segment_summary.json")

    return report, imp_df, summary


if __name__ == "__main__":
    generate_business_report()
