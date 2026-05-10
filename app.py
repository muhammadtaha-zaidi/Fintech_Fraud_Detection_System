
# ==========================================
# 💳 FINTECH FRAUD DETECTION SYSTEM
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="💳",
    layout="wide"
)

# ==========================================
# LOAD MODEL & FEATURES
# ==========================================

@st.cache_resource
def load_artifacts():

    model = joblib.load("RandomForest_fraud_model.pkl")

    features = joblib.load("feature_names.pkl")

    return model, features

try:

    model, features = load_artifacts()

except Exception as e:

    st.error(f"❌ Error loading model files: {e}")

    st.stop()

# ==========================================
# PREPROCESSING FUNCTION
# ==========================================

def preprocess_input(raw):

    # --------------------------------------
    # Feature Engineering
    # --------------------------------------

    sender_balance_diff = (
        raw["oldbalanceOrg"] - raw["newbalanceOrig"]
    )

    receiver_balance_diff = (
        raw["newbalanceDest"] - raw["oldbalanceDest"]
    )

    amount_balance_ratio = (
        raw["amount"] / (raw["oldbalanceOrg"] + 1)
    )

    amount_log = np.log1p(raw["amount"])

    # --------------------------------------
    # Time Features
    # --------------------------------------

    hour = raw["step"] % 24

    day = raw["step"] // 24

    is_night = 1 if hour >= 22 or hour <= 6 else 0

    # --------------------------------------
    # Balance Features
    # --------------------------------------

    is_sender_zero_balance = (
        1 if raw["newbalanceOrig"] == 0 else 0
    )

    is_receiver_zero_balance = (
        1 if raw["newbalanceDest"] == 0 else 0
    )

    is_large_transaction = (
        1 if raw["amount"] > 200000 else 0
    )

    # --------------------------------------
    # Transaction Type Encoding
    # --------------------------------------

    transaction_types = {
        "type_CASH_OUT": 0,
        "type_DEBIT": 0,
        "type_PAYMENT": 0,
        "type_TRANSFER": 0
    }

    selected_type = f"type_{raw['type']}"

    if selected_type in transaction_types:
        transaction_types[selected_type] = 1

    # --------------------------------------
    # Final DataFrame
    # --------------------------------------

    row = {

        "step": raw["step"],

        "amount": raw["amount"],

        "oldbalanceOrg": raw["oldbalanceOrg"],

        "newbalanceOrig": raw["newbalanceOrig"],

        "oldbalanceDest": raw["oldbalanceDest"],

        "newbalanceDest": raw["newbalanceDest"],

        "sender_balance_diff": sender_balance_diff,

        "receiver_balance_diff": receiver_balance_diff,

        "amount_balance_ratio": amount_balance_ratio,

        "amount_log": amount_log,

        "hour": hour,

        "day": day,

        "is_night": is_night,

        "is_sender_zero_balance": is_sender_zero_balance,

        "is_receiver_zero_balance": is_receiver_zero_balance,

        "is_large_transaction": is_large_transaction,

        **transaction_types
    }

    return pd.DataFrame([row])[features]

# ==========================================
# UI
# ==========================================

st.title("💳 Fintech Fraud Detection System")

st.write(
    "Enter transaction details below to predict whether "
    "a transaction is fraudulent or legitimate."
)

st.divider()

# ==========================================
# INPUT SECTION
# ==========================================

col1, col2 = st.columns(2)

with col1:

    st.subheader("💰 Transaction Information")

    transaction_type = st.selectbox(
        "Transaction Type",
        ["CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
    )

    amount = st.number_input(
        "Transaction Amount",
        min_value=0.0,
        value=1000.0,
        step=100.0
    )

    step = st.number_input(
        "Transaction Step (Hour)",
        min_value=1,
        value=1,
        step=1
    )

with col2:

    st.subheader("🏦 Account Balances")

    oldbalanceOrg = st.number_input(
        "Sender Old Balance",
        min_value=0.0,
        value=5000.0
    )

    newbalanceOrig = st.number_input(
        "Sender New Balance",
        min_value=0.0,
        value=4000.0
    )

    oldbalanceDest = st.number_input(
        "Receiver Old Balance",
        min_value=0.0,
        value=1000.0
    )

    newbalanceDest = st.number_input(
        "Receiver New Balance",
        min_value=0.0,
        value=2000.0
    )

st.divider()

# ==========================================
# PREDICTION
# ==========================================

if st.button(
    "🔍 Predict Fraud",
    type="primary",
    use_container_width=True
):

    raw_data = {

        "step": step,

        "type": transaction_type,

        "amount": amount,

        "oldbalanceOrg": oldbalanceOrg,

        "newbalanceOrig": newbalanceOrig,

        "oldbalanceDest": oldbalanceDest,

        "newbalanceDest": newbalanceDest
    }

    # Preprocess Input
    X = preprocess_input(raw_data)

    # Prediction
    prediction = model.predict(X)[0]

    probability = model.predict_proba(X)[0][1]

    st.divider()

    # ======================================
    # RESULT DISPLAY
    # ======================================

    if prediction == 1:

        st.error(
            f"⚠️ Fraudulent Transaction Detected\n\n"
            f"Fraud Probability: {probability*100:.2f}%"
        )

    else:

        st.success(
            f"✅ Legitimate Transaction\n\n"
            f"Fraud Probability: {probability*100:.2f}%"
        )

    st.progress(
        float(probability),
        text=f"Fraud Risk Score: {probability*100:.2f}%"
    )

# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "📌 This ML system predicts potentially fraudulent "
    "financial transactions using a Random Forest model "
    "trained on the PaySim dataset."
)
