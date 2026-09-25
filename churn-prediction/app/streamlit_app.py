"""
streamlit_app.py
-----------------
Interactive dashboard: upload a customer CSV -> get churn risk scores,
segments, top drivers, and recommended retention actions.

Run from the project root:
    streamlit run app/streamlit_app.py
"""

import json
import pickle
import sys

import numpy as np
import pandas as pd
import streamlit as st

sys.path.append("src")
from features import engineer_features
from preprocessing import clean_data

st.set_page_config(page_title="Churn Risk Dashboard", layout="wide")


@st.cache_resource
def load_artifacts():
    with open("models/best_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    with open("models/feature_names.json") as f:
        feature_names = json.load(f)

    return model, scaler, feature_names


def fix_model_compatibility(model):
    """
    Fix compatibility issue with older LogisticRegression
    models loaded using newer scikit-learn versions.
    """

    if type(model).__name__ == "LogisticRegression":

        # Older saved models may not contain this attribute.
        if not hasattr(model, "multi_class"):
            model.multi_class = "auto"

        # Make sure required LogisticRegression attributes exist
        if not hasattr(model, "solver"):
            model.solver = "lbfgs"

    return model


def risk_segment(prob):
    if prob >= 0.7:
        return "Critical Risk"
    elif prob >= 0.4:
        return "High Risk"
    elif prob >= 0.2:
        return "Watch"
    return "Healthy"


ACTIONS = {
    "Critical Risk": "Immediate personal outreach + retention offer within 48h.",
    "High Risk": "Proactive check-in, highlight underused features, loyalty discount.",
    "Watch": "Add to nurture campaign; monitor next billing cycle.",
    "Healthy": "No action needed; upsell / referral candidate.",
}


# ---------------------------------------------------------
# APP TITLE
# ---------------------------------------------------------

st.title("📉 Customer Churn Risk Dashboard")
st.caption(
    "Upload customer data to score churn risk and get retention recommendations."
)


# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------

model, scaler, feature_names = load_artifacts()

# Fix old LogisticRegression pickle compatibility
model = fix_model_compatibility(model)


# ---------------------------------------------------------
# FILE UPLOAD
# ---------------------------------------------------------

uploaded = st.file_uploader(
    "Upload customer CSV (same schema as telco_churn.csv)",
    type="csv"
)

if uploaded is None:
    st.info(
        "No file uploaded — using the sample dataset "
        "(data/telco_churn.csv) for demo purposes."
    )

    df_raw = pd.read_csv("data/telco_churn.csv")

else:
    df_raw = pd.read_csv(uploaded)


# ---------------------------------------------------------
# DATA CLEANING
# ---------------------------------------------------------

has_labels = "Churn" in df_raw.columns

if has_labels:
    df_clean = clean_data(df_raw)
else:
    df_clean = clean_data(
        df_raw.assign(Churn="No")
    )


# ---------------------------------------------------------
# FEATURE ENGINEERING
# ---------------------------------------------------------

X, _ = engineer_features(df_clean)

# Make sure columns match the columns used during training
X = X.reindex(
    columns=feature_names,
    fill_value=0
)


# ---------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------

needs_scaling = hasattr(model, "coef_")

if needs_scaling:
    X_for_pred = scaler.transform(X)
else:
    X_for_pred = X


# Predict churn probability
probs = model.predict_proba(X_for_pred)[:, 1]


# ---------------------------------------------------------
# RESULTS
# ---------------------------------------------------------

results = df_raw.copy()

results["churn_probability"] = np.round(
    probs,
    3
)

results["risk_segment"] = results[
    "churn_probability"
].apply(risk_segment)

results["recommended_action"] = results[
    "risk_segment"
].map(ACTIONS)


# ---------------------------------------------------------
# KPI ROW
# ---------------------------------------------------------

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Customers",
    len(results)
)

col2.metric(
    "Critical Risk",
    int(
        (
            results["risk_segment"] == "Critical Risk"
        ).sum()
    )
)

col3.metric(
    "High Risk",
    int(
        (
            results["risk_segment"] == "High Risk"
        ).sum()
    )
)

at_risk_revenue = results.loc[
    results["risk_segment"].isin(
        ["Critical Risk", "High Risk"]
    ),
    "MonthlyCharges"
].sum()

col4.metric(
    "Monthly Revenue at Risk",
    f"${at_risk_revenue:,.0f}"
)


st.divider()


# ---------------------------------------------------------
# SEGMENT BREAKDOWN
# ---------------------------------------------------------

seg_counts = (
    results["risk_segment"]
    .value_counts()
    .reindex(
        [
            "Critical Risk",
            "High Risk",
            "Watch",
            "Healthy"
        ]
    )
    .fillna(0)
)

st.subheader("Customers by Risk Segment")

st.bar_chart(seg_counts)


st.divider()


# ---------------------------------------------------------
# CUSTOMER TABLE
# ---------------------------------------------------------

st.subheader("Customer Risk Table")

segment_filter = st.multiselect(
    "Filter by segment",

    options=seg_counts.index.tolist(),

    default=[
        "Critical Risk",
        "High Risk"
    ]
)

filtered = (
    results[
        results["risk_segment"].isin(
            segment_filter
        )
    ]
    .sort_values(
        "churn_probability",
        ascending=False
    )
)

show_cols = [
    "customerID",
    "tenure",
    "Contract",
    "MonthlyCharges",
    "churn_probability",
    "risk_segment",
    "recommended_action"
]

show_cols = [
    c for c in show_cols
    if c in filtered.columns
]

st.dataframe(
    filtered[show_cols],
    use_container_width=True,
    height=400
)


# ---------------------------------------------------------
# DOWNLOAD RESULTS
# ---------------------------------------------------------

st.download_button(
    "Download full scored dataset (CSV)",

    data=results.to_csv(
        index=False
    ).encode("utf-8"),

    file_name="churn_risk_scored.csv",

    mime="text/csv",
)


st.divider()


# ---------------------------------------------------------
# SINGLE CUSTOMER EXPLANATION
# ---------------------------------------------------------

st.subheader("🔍 Inspect a Single Customer")

if "customerID" in results.columns:

    selected_id = st.selectbox(
        "Select customer",
        results["customerID"].tolist()
    )

    cust = results[
        results["customerID"] == selected_id
    ].iloc[0]

    c1, c2 = st.columns([1, 2])

    with c1:

        st.metric(
            "Churn Probability",
            f"{cust['churn_probability']:.1%}"
        )

        st.write(
            f"**Segment:** {cust['risk_segment']}"
        )

        st.write(
            f"**Recommended action:** "
            f"{cust['recommended_action']}"
        )

    with c2:

        st.write("**Key profile details**")

        details = cust[
            [
                "tenure",
                "Contract",
                "MonthlyCharges",
                "InternetService",
                "TechSupport",
                "PaymentMethod"
            ]
        ]

        st.table(details)

else:

    st.caption(
        "Upload a dataset with a `customerID` column "
        "to inspect individual customers."
    )