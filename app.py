import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="UPI Fraud Detection",
    page_icon="💳",
    layout="wide"
)

MODEL_PATH = Path("models/fraud_model.pkl")
FEATURES_PATH = Path("models/feature_columns.pkl")

st.title("💳 UPI Fraud Detection & Risk Scoring")
st.caption("Educational/demo decision-support system — not a banking fraud decision engine.")

if not MODEL_PATH.exists() or not FEATURES_PATH.exists():
    st.error("Model files are missing. Run: python train_model.py")
    st.stop()

model = joblib.load(MODEL_PATH)
features = joblib.load(FEATURES_PATH)

st.sidebar.header("Transaction Details")

amount = st.sidebar.number_input("Transaction Amount (₹)", 20.0, 100000.0, 1000.0)
hour = st.sidebar.slider("Transaction Hour", 0, 23, 14)
transactions_last_1h = st.sidebar.slider("Transactions in Last 1 Hour", 0, 15, 1)
failed_attempts = st.sidebar.slider("Failed Attempts", 0, 6, 0)
account_age_days = st.sidebar.number_input("Account Age (days)", 1, 2500, 365)
avg_amount_30d = st.sidebar.number_input("Average Amount (30 days)", 50.0, 20000.0, 800.0)
new_device = st.sidebar.selectbox("New Device", ["No", "Yes"])
new_beneficiary = st.sidebar.selectbox("New Beneficiary", ["No", "Yes"])
location_change = st.sidebar.selectbox("Location Change", ["No", "Yes"])
is_international_ip = st.sidebar.selectbox("International IP", ["No", "Yes"])
device_change_24h = st.sidebar.selectbox("Device Change in 24h", ["No", "Yes"])
velocity_10m = st.sidebar.slider("Transactions in Last 10 Minutes", 0, 10, 0)

if st.button("🔍 Check Transaction", type="primary"):
    row = {
        "amount": amount,
        "hour": hour,
        "transactions_last_1h": transactions_last_1h,
        "failed_attempts": failed_attempts,
        "account_age_days": account_age_days,
        "avg_amount_30d": avg_amount_30d,
        "new_device": int(new_device == "Yes"),
        "new_beneficiary": int(new_beneficiary == "Yes"),
        "location_change": int(location_change == "Yes"),
        "is_international_ip": int(is_international_ip == "Yes"),
        "device_change_24h": int(device_change_24h == "Yes"),
        "velocity_10m": velocity_10m,
        "amount_ratio": amount / (avg_amount_30d + 1),
        "night_transaction": int(hour <= 5 or hour >= 23)
    }

    X = pd.DataFrame([row])[features]
    probability = float(model.predict_proba(X)[0, 1])
    prediction = int(probability >= 0.50)

    if probability >= 0.75:
        risk = "HIGH"
    elif probability >= 0.40:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    c1, c2, c3 = st.columns(3)
    c1.metric("Prediction", "⚠️ Potential Fraud" if prediction else "✅ Likely Legitimate")
    c2.metric("Fraud Probability", f"{probability * 100:.2f}%")
    c3.metric("Risk Level", risk)

    st.progress(min(max(probability, 0.0), 1.0))

    st.subheader("Transaction Features")
    st.dataframe(X.T.rename(columns={0: "Value"}), use_container_width=True)

    st.info(
        "A model score is a risk signal, not proof of fraud. "
        "Real payment systems require additional authentication, rules, "
        "human review, and bank-side controls."
    )

st.divider()
st.subheader("How the model works")
st.write(
    "The system uses transaction amount, transaction velocity, device changes, "
    "beneficiary changes, location changes, failed attempts, account age, "
    "and related behavioral features to estimate fraud probability."
)
