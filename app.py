import streamlit as st
import pandas as pd
import joblib
import os
import warnings

warnings.filterwarnings("ignore")

# =========================================================
# CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Telecom Churn Dashboard",
    page_icon="📡",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():
    model_path = os.path.join(BASE_DIR, "telecom_churn_model.pkl")

    if not os.path.exists(model_path):
        st.error("❌ Model file not found!")
        st.stop()

    return joblib.load(model_path)

model = load_model()

# =========================================================
# FEATURES (NO PICKLE NEEDED)
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
# CSS (PROFESSIONAL UI)
# =========================================================

st.markdown("""
<style>

body {
    background-color: #0f172a;
}

.main {
    background: linear-gradient(to right, #0f172a, #111827);
    color: white;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0b1220;
}

/* Cards */
.card {
    background: rgba(255,255,255,0.05);
    padding: 20px;
    border-radius: 16px;
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}

/* Buttons */
.stButton>button {
    width: 100%;
    background: linear-gradient(135deg, #2563eb, #06b6d4);
    color: white;
    padding: 0.6rem;
    border-radius: 10px;
    font-size: 18px;
    font-weight: bold;
    border: none;
    transition: 0.3s;
}

.stButton>button:hover {
    transform: scale(1.02);
    background: linear-gradient(135deg, #06b6d4, #2563eb);
}

/* Title */
.title {
    font-size: 40px;
    font-weight: bold;
    text-align: center;
    margin-bottom: 10px;
}

/* Footer */
.footer {
    text-align: center;
    color: gray;
    margin-top: 30px;
    font-size: 13px;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# HEADER
# =========================================================

st.markdown("<div class='title'>📡 Telecom Churn AI Dashboard</div>", unsafe_allow_html=True)

st.markdown("### Predict customer churn using Machine Learning in real-time dashboard style UI")
st.markdown("---")

# =========================================================
# SIDEBAR INPUTS
# =========================================================

st.sidebar.header("Customer Profile")

tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 12)

monthly_charges = st.sidebar.number_input("Monthly Charges", 10.0, 500.0, 80.0)

total_charges = st.sidebar.number_input("Total Charges", 0.0, 10000.0, 1500.0)

contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

internet = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

payment = st.sidebar.selectbox("Payment Method",
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
# INPUT DATA
# =========================================================

values = {
    "gender": 1,
    "SeniorCitizen": 0,
    "Partner": 1,
    "Dependents": 0,
    "tenure": tenure,
    "PhoneService": 1,
    "MultipleLines": 0,
    "InternetService": internet_map[internet],
    "OnlineSecurity": 0,
    "OnlineBackup": 1,
    "DeviceProtection": 1,
    "TechSupport": 0,
    "StreamingTV": 1,
    "StreamingMovies": 1,
    "Contract": contract_map[contract],
    "PaperlessBilling": 1,
    "PaymentMethod": payment_map[payment],
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges
}

input_df = pd.DataFrame([{f: values.get(f, 0) for f in features}])

# =========================================================
# DASHBOARD METRICS
# =========================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.metric("Monthly Charges", f"${monthly_charges}")
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.metric("Tenure", f"{tenure} Months")
    st.markdown("</div>", unsafe_allow_html=True)

with col3:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.metric("Total Charges", f"${total_charges}")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

# =========================================================
# PREDICTION BUTTON
# =========================================================

if st.button("🔍 Predict Churn"):

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1] * 100

    st.subheader("📊 Prediction Result")

    colA, colB = st.columns(2)

    with colA:
        st.metric("Churn Probability", f"{probability:.2f}%")
        st.progress(int(probability))

    with colB:
        st.metric("Risk Level", "HIGH" if probability > 70 else "LOW")

    if prediction == 1:
        st.error("⚠️ Customer is likely to CHURN")
    else:
        st.success("✅ Customer is likely to STAY")

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.markdown("""
<div class='footer'>
Built with ❤️ using Streamlit • Telecom AI Dashboard
</div>
""", unsafe_allow_html=True)
