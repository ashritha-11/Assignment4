

import streamlit as st
import pandas as pd
import joblib
import warnings

warnings.filterwarnings("ignore")

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Telecom Churn Platform",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# LOAD FILES
# =========================================================

@st.cache_resource
def load_model():

    model = joblib.load("telecom_churn_model.pkl")

    features = joblib.load("features.pkl")

    return model, features

model, features = load_model()

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main {
    background: linear-gradient(to right, #0f172a, #111827);
    color: white;
}

section[data-testid="stSidebar"] {
    background: #0B1120;
}

section[data-testid="stSidebar"] * {
    color: white !important;
}

.stButton>button {
    width: 100%;
    background: linear-gradient(to right, #2563eb, #06b6d4);
    color: white;
    border-radius: 12px;
    height: 3.2em;
    font-size: 18px;
    font-weight: bold;
    border: none;
}

.stButton>button:hover {
    background: linear-gradient(to right, #06b6d4, #2563eb);
}

.card {
    background: #111827;
    padding: 25px;
    border-radius: 18px;
    border: 1px solid #1f2937;
    text-align: center;
}

.footer {
    text-align: center;
    color: #9ca3af;
    padding-top: 20px;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# HEADER
# =========================================================

st.title("📡 AI Telecom Customer Churn Prediction Platform")

st.markdown(
    "Predict customer churn, revenue loss, and retention risk using Machine Learning."
)

st.markdown("---")

# =========================================================
# SIDEBAR INPUTS
# =========================================================

st.sidebar.header("Customer Information")

tenure = st.sidebar.slider(
    "Tenure (Months)",
    0,
    72,
    12
)

monthly_charges = st.sidebar.number_input(
    "Monthly Charges",
    10.0,
    500.0,
    80.0
)

total_charges = st.sidebar.number_input(
    "Total Charges",
    0.0,
    10000.0,
    1500.0
)

contract = st.sidebar.selectbox(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"]
)

internet_service = st.sidebar.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

payment_method = st.sidebar.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer",
        "Credit card"
    ]
)

# =========================================================
# ENCODING
# =========================================================

contract_map = {
    "Month-to-month": 0,
    "One year": 1,
    "Two year": 2
}

internet_map = {
    "DSL": 0,
    "Fiber optic": 1,
    "No": 2
}

payment_map = {
    "Electronic check": 0,
    "Mailed check": 1,
    "Bank transfer": 2,
    "Credit card": 3
}

# =========================================================
# DEFAULT INPUT VALUES
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
# MATCH FEATURES
# =========================================================

final_input = {}

for feature in features:

    if feature in default_values:

        final_input[feature] = default_values[feature]

    else:

        final_input[feature] = 0

input_data = pd.DataFrame([final_input])

input_data = input_data[features]

# =========================================================
# DASHBOARD
# =========================================================

c1, c2, c3 = st.columns(3)

with c1:

    st.metric(
        "Monthly Charges",
        f"$ {monthly_charges}"
    )

with c2:

    st.metric(
        "Tenure",
        f"{tenure} Months"
    )

with c3:

    st.metric(
        "Total Charges",
        f"$ {total_charges}"
    )

st.markdown("---")

# =========================================================
# PREDICTION
# =========================================================

if st.button("Predict Churn"):

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(
        input_data
    )[0][1] * 100

    # =====================================================
    # RISK LEVEL
    # =====================================================

    if probability >= 75:

        risk = "🔴 HIGH RISK"

        recommendation = """
        Offer discounts and loyalty benefits immediately.
        """

    elif probability >= 45:

        risk = "🟠 MEDIUM RISK"

        recommendation = """
        Improve customer engagement and support quality.
        """

    else:

        risk = "🟢 LOW RISK"

        recommendation = """
        Customer appears stable and satisfied.
        """

    # =====================================================
    # REVENUE LOSS
    # =====================================================

    revenue_loss = monthly_charges * 12

    # =====================================================
    # RESULTS
    # =====================================================

    st.subheader("📊 Prediction Results")

    r1, r2 = st.columns(2)

    with r1:

        st.metric(
            "Churn Probability",
            f"{probability:.2f}%"
        )

        st.progress(int(probability))

    with r2:

        st.metric(
            "Expected Revenue Loss",
            f"$ {revenue_loss:.2f}"
        )

        st.info(risk)

    if prediction == 1:

        st.error(
            "Customer likely to churn."
        )

    else:

        st.success(
            "Customer likely to stay."
        )

    # =====================================================
    # RETENTION RECOMMENDATION
    # =====================================================

    st.markdown("## 💡 Retention Recommendation")

    st.info(recommendation)

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.markdown(
    """
    <div class="footer">
    Built with Streamlit • Random Forest • Telecom Analytics AI
    </div>
    """,
    unsafe_allow_html=True
)

