import streamlit as st
import pandas as pd
import joblib
import warnings
import os

warnings.filterwarnings("ignore")

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Telecom Churn Platform",
    page_icon="📡",
    layout="wide"
)

# =========================================================
# BASE DIR
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# =========================================================
# LOAD MODEL ONLY (NO features.pkl NEEDED)
# =========================================================

@st.cache_resource
def load_model():

    model_path = os.path.join(BASE_DIR, "telecom_churn_model.pkl")

    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found: {model_path}")
        st.stop()

    model = joblib.load(model_path)

    return model


model = load_model()

# =========================================================
# DEFINE FEATURES DIRECTLY (FIX)
# =========================================================

features = [
    "gender", "SeniorCitizen", "Partner", "Dependents",
    "tenure", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling",
    "PaymentMethod", "MonthlyCharges", "TotalCharges"
]

# =========================================================
# SIDEBAR INPUTS
# =========================================================

st.sidebar.header("Customer Information")

tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 12)

monthly_charges = st.sidebar.number_input("Monthly Charges", 10.0, 500.0, 80.0)

total_charges = st.sidebar.number_input("Total Charges", 0.0, 10000.0, 1500.0)

contract = st.sidebar.selectbox("Contract Type",
                                ["Month-to-month", "One year", "Two year"])

internet_service = st.sidebar.selectbox("Internet Service",
                                        ["DSL", "Fiber optic", "No"])

payment_method = st.sidebar.selectbox("Payment Method",
                                      ["Electronic check", "Mailed check",
                                       "Bank transfer", "Credit card"])

# =========================================================
# ENCODING
# =========================================================

contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
internet_map = {"DSL": 0, "Fiber optic": 1, "No": 2}
payment_map = {
    "Electronic check": 0,
    "Mailed check": 1,
    "Bank transfer": 2,
    "Credit card": 3
}

# =========================================================
# INPUT VALUES
# =========================================================

default_values = {
    "gender": 1,
    "SeniorCitizen": 0,
    "Partner": 1,
    "Dependents": 0,
    "tenure": tenure,
    "PhoneService": 1,
    "MultipleLines": 0,
    "InternetService": internet_map[internet_service],
    "OnlineSecurity": 0,
    "OnlineBackup": 1,
    "DeviceProtection": 1,
    "TechSupport": 0,
    "StreamingTV": 1,
    "StreamingMovies": 1,
    "Contract": contract_map[contract],
    "PaperlessBilling": 1,
    "PaymentMethod": payment_map[payment_method],
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges
}

# =========================================================
# BUILD INPUT DATA
# =========================================================

input_data = pd.DataFrame([{f: default_values.get(f, 0) for f in features}])

# =========================================================
# UI
# =========================================================

st.title("📡 Telecom Churn Prediction")

c1, c2, c3 = st.columns(3)

c1.metric("Monthly Charges", f"${monthly_charges}")
c2.metric("Tenure", f"{tenure} months")
c3.metric("Total Charges", f"${total_charges}")

# =========================================================
# PREDICTION
# =========================================================

if st.button("Predict Churn"):

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1] * 100

    st.subheader("Results")

    st.metric("Churn Probability", f"{probability:.2f}%")
    st.progress(int(probability))

    if prediction == 1:
        st.error("Customer likely to churn")
    else:
        st.success("Customer likely to stay")
