"""
generate_data.py
-----------------
Creates a realistic, synthetic telecom customer-churn dataset so the whole
project can be run immediately without downloading anything.

The columns intentionally mirror the popular "Telco Customer Churn" dataset
(IBM / Kaggle), so once you're ready, you can drop in the real CSV
(https://www.kaggle.com/datasets/blastchar/telco-customer-churn) in place of
this file's output and every downstream script keeps working unchanged.

Run:
    python data/generate_data.py
Produces:
    data/telco_churn.csv
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N = 7000


def generate(n=N):
    df = pd.DataFrame()
    df["customerID"] = [f"C{100000+i}" for i in range(n)]

    df["gender"] = RNG.choice(["Male", "Female"], n)
    df["SeniorCitizen"] = RNG.choice([0, 1], n, p=[0.84, 0.16])
    df["Partner"] = RNG.choice(["Yes", "No"], n, p=[0.48, 0.52])
    df["Dependents"] = RNG.choice(["Yes", "No"], n, p=[0.3, 0.7])

    # Tenure in months - skewed towards short tenure (realistic churn pattern)
    df["tenure"] = RNG.integers(0, 73, n)

    df["PhoneService"] = RNG.choice(["Yes", "No"], n, p=[0.9, 0.1])
    df["MultipleLines"] = np.where(
        df["PhoneService"] == "No", "No phone service",
        RNG.choice(["Yes", "No"], n)
    )

    df["InternetService"] = RNG.choice(
        ["DSL", "Fiber optic", "No"], n, p=[0.35, 0.44, 0.21]
    )

    def dependent_service(has_internet):
        return np.where(
            has_internet == "No", "No internet service",
            RNG.choice(["Yes", "No"], len(has_internet))
        )

    df["OnlineSecurity"] = dependent_service(df["InternetService"])
    df["OnlineBackup"] = dependent_service(df["InternetService"])
    df["DeviceProtection"] = dependent_service(df["InternetService"])
    df["TechSupport"] = dependent_service(df["InternetService"])
    df["StreamingTV"] = dependent_service(df["InternetService"])
    df["StreamingMovies"] = dependent_service(df["InternetService"])

    df["Contract"] = RNG.choice(
        ["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.24, 0.21]
    )
    df["PaperlessBilling"] = RNG.choice(["Yes", "No"], n, p=[0.59, 0.41])
    df["PaymentMethod"] = RNG.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)",
         "Credit card (automatic)"],
        n, p=[0.34, 0.23, 0.22, 0.21]
    )

    # Monthly charges depend loosely on services selected
    base = 18 + RNG.normal(0, 3, n)
    internet_cost = np.select(
        [df["InternetService"] == "DSL", df["InternetService"] == "Fiber optic"],
        [20, 45], default=0
    )
    extra_services = sum(
        (df[col] == "Yes").astype(int) * RNG.uniform(4, 8, n)
        for col in ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                    "TechSupport", "StreamingTV", "StreamingMovies"]
    )
    df["MonthlyCharges"] = np.round(base + internet_cost + extra_services, 2).clip(18, 120)
    df["TotalCharges"] = np.round(df["MonthlyCharges"] * df["tenure"] +
                                   RNG.normal(0, 20, n), 2).clip(0, None)

    # --- Behavioral / engagement features (not in the original dataset,
    # added to make the project feel closer to a real production system) ---
    df["SupportTicketsLast6Months"] = RNG.poisson(1.2, n)
    df["AvgMonthlyLoginDays"] = np.round(RNG.normal(14, 6, n).clip(0, 30), 1)
    df["LatePaymentsLast12Months"] = RNG.poisson(0.6, n)

    # --- Build churn probability from a realistic combination of drivers ---
    logit = (
        -1.8
        + 1.4 * (df["Contract"] == "Month-to-month")
        - 0.9 * (df["Contract"] == "Two year")
        + 0.015 * (df["MonthlyCharges"] - 60)
        - 0.03 * df["tenure"]
        + 0.25 * df["SupportTicketsLast6Months"]
        + 0.35 * df["LatePaymentsLast12Months"]
        - 0.02 * df["AvgMonthlyLoginDays"]
        + 0.5 * (df["InternetService"] == "Fiber optic")
        + 0.4 * (df["PaymentMethod"] == "Electronic check")
        - 0.3 * (df["TechSupport"] == "Yes")
        + RNG.normal(0, 0.6, n)
    )
    prob = 1 / (1 + np.exp(-logit))
    df["Churn"] = (RNG.uniform(0, 1, n) < prob).map({True: "Yes", False: "No"})

    return df


if __name__ == "__main__":
    data = generate()
    out_path = "data/telco_churn.csv"
    data.to_csv(out_path, index=False)
    print(f"Saved {len(data)} rows to {out_path}")
    print(f"Churn rate: {(data['Churn'] == 'Yes').mean():.2%}")
