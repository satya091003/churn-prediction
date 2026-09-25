"""
eda.py
------
Quick exploratory data analysis. Generates the charts you'll want for your
report/presentation and saves them to reports/.

Run from the project root:
    python notebooks/eda.py
"""

import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sys.path.append("src")
from preprocessing import clean_data, load_data

sns.set_style("whitegrid")


def run_eda():
    df = clean_data(load_data())
    df["Churn_label"] = df["Churn"].map({1: "Yes", 0: "No"})

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # 1. Overall churn rate
    df["Churn_label"].value_counts().plot(
        kind="bar", ax=axes[0, 0], color=["#4C72B0", "#DD8452"]
    )
    axes[0, 0].set_title("Overall Churn Distribution")
    axes[0, 0].set_xlabel("")

    # 2. Churn rate by contract type
    contract_churn = df.groupby("Contract")["Churn"].mean().sort_values()
    contract_churn.plot(kind="barh", ax=axes[0, 1], color="#55A868")
    axes[0, 1].set_title("Churn Rate by Contract Type")
    axes[0, 1].set_xlabel("Churn Rate")

    # 3. Tenure distribution by churn
    sns.histplot(
        data=df, x="tenure", hue="Churn_label", bins=30, kde=True,
        ax=axes[1, 0], palette=["#4C72B0", "#DD8452"]
    )
    axes[1, 0].set_title("Tenure Distribution by Churn")

    # 4. Monthly charges by churn
    sns.boxplot(
        data=df, x="Churn_label", y="MonthlyCharges", ax=axes[1, 1],
        palette=["#4C72B0", "#DD8452"]
    )
    axes[1, 1].set_title("Monthly Charges by Churn")

    plt.tight_layout()
    plt.savefig("reports/eda_overview.png", dpi=150)
    print("Saved reports/eda_overview.png")

    # Print key stats for the report
    print("\nChurn rate:", round(df["Churn"].mean(), 3))
    print("\nChurn rate by contract:\n", contract_churn.round(3))
    print("\nChurn rate by payment method:\n",
          df.groupby("PaymentMethod")["Churn"].mean().sort_values(ascending=False).round(3))


if __name__ == "__main__":
    run_eda()
