"""
train.py
--------
End-to-end training pipeline:
    1. Load + clean + engineer features
    2. Train/test split (stratified)
    3. Handle class imbalance (SMOTE if available, else class_weight)
    4. Train Logistic Regression (baseline) + Random Forest + XGBoost (if available)
    5. Evaluate with metrics that matter for imbalanced churn data (PR-AUC, F1, recall)
    6. Save the best model + the feature list to models/

Run from the project root:
    python src/train.py
"""

import json
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.append("src")
from features import engineer_features
from preprocessing import clean_data, load_data

# Optional dependencies - the project degrades gracefully if these aren't
# installed locally, but we STRONGLY recommend installing xgboost + imbalanced-learn
# (see requirements.txt) for the best results.
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False


def evaluate(model, X_test, y_test, name):
    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)
    metrics = {
        "model": name,
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
        "pr_auc": round(average_precision_score(y_test, proba), 4),
        "f1": round(f1_score(y_test, preds), 4),
    }
    print(f"\n=== {name} ===")
    print(json.dumps(metrics, indent=2))
    print(classification_report(y_test, preds, target_names=["No Churn", "Churn"]))
    return metrics, proba


def main():
    print("Loading data...")
    df = clean_data(load_data())
    X, y = engineer_features(df)
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Scale numeric features for Logistic Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- Handle class imbalance ---
    if HAS_SMOTE:
        print("Balancing classes with SMOTE...")
        sm = SMOTE(random_state=42)
        X_train_bal, y_train_bal = sm.fit_resample(X_train, y_train)
        X_train_scaled_bal, y_train_scaled_bal = sm.fit_resample(X_train_scaled, y_train)
    else:
        print("imbalanced-learn not installed -> using class_weight='balanced' instead "
              "(pip install imbalanced-learn for SMOTE oversampling).")
        X_train_bal, y_train_bal = X_train, y_train
        X_train_scaled_bal, y_train_scaled_bal = X_train_scaled, y_train

    class_weight = None if HAS_SMOTE else "balanced"

    results = []

    # --- Baseline: Logistic Regression ---
    lr = LogisticRegression(max_iter=1000, class_weight=class_weight, random_state=42)
    lr.fit(X_train_scaled_bal, y_train_scaled_bal)
    m, _ = evaluate(lr, X_test_scaled, y_test, "Logistic Regression")
    results.append((m, lr, "logreg"))

    # --- Random Forest ---
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=8, class_weight=class_weight,
        random_state=42, n_jobs=-1
    )
    rf.fit(X_train_bal, y_train_bal)
    m, _ = evaluate(rf, X_test, y_test, "Random Forest")
    results.append((m, rf, "rf"))

    # --- Gradient Boosting (sklearn fallback if xgboost isn't installed) ---
    if HAS_XGB:
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum() if not HAS_SMOTE else 1
        xgb = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            scale_pos_weight=scale_pos_weight, eval_metric="logloss",
            random_state=42
        )
        xgb.fit(X_train_bal, y_train_bal)
        m, _ = evaluate(xgb, X_test, y_test, "XGBoost")
        results.append((m, xgb, "xgb"))
    else:
        print("\nxgboost not installed -> training sklearn GradientBoostingClassifier "
              "instead (pip install xgboost for the real thing).")
        gb = GradientBoostingClassifier(n_estimators=300, max_depth=3, random_state=42)
        gb.fit(X_train_bal, y_train_bal)
        m, _ = evaluate(gb, X_test, y_test, "Gradient Boosting (sklearn)")
        results.append((m, gb, "gb"))

    # --- Pick best model by PR-AUC (most meaningful metric for imbalanced churn) ---
    best_metrics, best_model, best_tag = max(results, key=lambda r: r[0]["pr_auc"])
    print(f"\nBest model: {best_metrics['model']} (PR-AUC={best_metrics['pr_auc']})")

    # --- Save everything needed to reproduce predictions later ---
    with open("models/best_model.pkl", "wb") as f:
        pickle.dump(best_model, f)
    with open("models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open("models/feature_names.json", "w") as f:
        json.dump(feature_names, f)
    with open("models/metrics.json", "w") as f:
        json.dump({"best_model": best_tag, "results": [r[0] for r in results]}, f, indent=2)

    print("\nSaved: models/best_model.pkl, models/scaler.pkl, "
          "models/feature_names.json, models/metrics.json")


if __name__ == "__main__":
    main()
