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
    page_title="Telecom AI Dashboard",
    page_icon="📡",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# =========================================================
# MODEL LOAD
# =========================================================

@st.cache_resource
def load_model():
    path = os.path.join(BASE_DIR, "telecom_churn_model.pkl")
    if not os.path.exists(path):
        st.error("Model file missing!")
        st.stop()
    return joblib.load(path)

model = load_model()

# =========================================================
# FEATURES
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
# ADVANCED CSS (PRO LEVEL UI)
# =========================================================

st.markdown("""
<style>

/* BACKGROUND */
.stApp {
    background: radial-gradient(circle at top left, #0f172a, #020617);
    color: white;
    font-family: 'Segoe UI', sans-serif;
}

/* REMOVE STREAMLIT DEFAULT PADDING */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    padding-left: 2rem;
    padding-right: 2rem;
}

/* TITLE */
.main-title {
    font-size: 42px;
    font-weight: 800;
    text-align: center;
    background: linear-gradient(90deg, #38bdf8, #6366f1);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 5px;
}

.sub-title {
    text-align: center;
    font-size: 16px;
    color: #94a3b8;
    margin-bottom: 25px;
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background: #0b1220;
    border-right: 1px solid #1f2937;
}

section[data-testid="stSidebar"] * {
    color: #e2e8f0;
}

/* CARD STYLE */
.card {
    background: rgba(255, 255, 255, 0.06);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 18px;
    padding: 18px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.4);
    transition: 0.3s ease;
}

.card:hover {
    transform: translateY(-4px);
    border: 1px solid rgba(56, 189, 248, 0.4);
}

/* METRICS */
div[data-testid="metric-container"] {
    background: rgba(255,255,255,0.04);
    border-radius: 14px;
    padding: 14px;
    border: 1px solid rgba(255,255,255,0.08);
}

/* BUTTON */
.stButton>button {
    width: 100%;
    padding: 0.75rem;
    font-size: 18px;
    font-weight: 600;
    border-radius: 12px;
    background: linear-gradient(135deg, #2563eb, #06b6d4);
    border: none;
    color: white;
    transition: 0.3s ease;
}

.stButton>button:hover {
    transform: scale(1.02);
    background: linear-gradient(135deg, #06b6d4, #2563eb);
}

/* PROGRESS BAR */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #22c55e, #06b6d4);
}

/* FOOTER */
.footer {
    text-align: center;
    color: #64748b;
    font-size: 13px;
    margin-top: 30px;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# HEADER
# =========================================================

st.markdown("<div class='main-title'>📡 Telecom AI Churn Dashboard</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Predict customer churn using Machine Learning & AI analytics</div>", unsafe_allow_html=True)

st.markdown("---")

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("👤 Customer Profile")

tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 12)
monthly = st.sidebar.number_input("Monthly Charges", 10.0, 500.0, 80.0)
total = st.sidebar.number_input("Total Charges", 0.0, 10000.0, 1500.0)

contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
internet = st.sidebar.selectbox("Internet", ["DSL", "Fiber optic", "No"])
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
    "MonthlyCharges": monthly,
    "TotalCharges": total
}

input_df = pd.DataFrame([{f: values.get(f, 0) for f in features}])

# =========================================================
# TOP METRICS (PRO GRID)
# =========================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.metric("💰 Monthly Charges", f"${monthly}")
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.metric("⏳ Tenure", f"{tenure} months")
    st.markdown("</div>", unsafe_allow_html=True)

with col3:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.metric("💳 Total Charges", f"${total}")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("##")

# =========================================================
# PREDICTION SECTION
# =========================================================

if st.button("🚀 Predict Churn Risk"):

    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1] * 100

    st.markdown("## 📊 Prediction Dashboard")

    colA, colB = st.columns([1.2, 1])

    with colA:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.metric("Churn Probability", f"{prob:.2f}%")
        st.progress(int(prob))
        st.markdown("</div>", unsafe_allow_html=True)

    with colB:
        risk = "🔴 HIGH" if prob > 70 else "🟠 MEDIUM" if prob > 40 else "🟢 LOW"
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.metric("Risk Level", risk)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("### Result")

    if pred == 1:
        st.error("⚠️ Customer is likely to CHURN")
    else:
        st.success("✅ Customer is likely to STAY")

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")
st.markdown("<div class='footer'>Built with Streamlit • AI Powered Telecom Analytics Dashboard</div>", unsafe_allow_html=True)
