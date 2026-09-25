"""
features.py
------------
Feature engineering. Takes the cleaned dataframe and returns:
    X (encoded feature matrix), y (target), feature_names
"""

import pandas as pd


def engineer_features(df):
    df = df.copy()

    # --- Tenure buckets: captures the well-known "new customer risk" pattern
    df["tenure_bucket"] = pd.cut(
        df["tenure"], bins=[-1, 6, 12, 24, 48, 100],
        labels=["0-6mo", "7-12mo", "13-24mo", "25-48mo", "49mo+"]
    )

    # --- Spend efficiency: charges relative to tenure
    df["avg_monthly_spend"] = df["TotalCharges"] / df["tenure"].replace(0, 1)

    # --- Count of add-on services subscribed to (engagement proxy)
    addon_cols = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    existing_addons = [c for c in addon_cols if c in df.columns]
    df["num_addon_services"] = (df[existing_addons] == "Yes").sum(axis=1)

    # --- Contract risk flag: month-to-month is the single strongest churn driver
    df["is_month_to_month"] = (df["Contract"] == "Month-to-month").astype(int)

    # --- Behavioral risk score, only if the behavioral columns exist
    if "SupportTicketsLast6Months" in df.columns:
        df["support_ticket_rate"] = df["SupportTicketsLast6Months"] / df["tenure"].replace(0, 1)

    y = df["Churn"]
    X = df.drop(columns=["Churn"])

    # One-hot encode categoricals
    X = pd.get_dummies(X, drop_first=True)

    return X, y


if __name__ == "__main__":
    from preprocessing import load_data, clean_data
    df = clean_data(load_data())
    X, y = engineer_features(df)
    print(X.shape, y.shape)
    print(X.columns.tolist())
