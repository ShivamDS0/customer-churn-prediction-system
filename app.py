# ============================================
# Customer Churn Prediction System
# Master's Final Project
# Developed by: Shivam
# ============================================

import streamlit as st
import pandas as pd
import joblib
import time
import plotly.graph_objects as go

# ============================================
# PAGE CONFIGURATION
# ============================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# ============================================
# CUSTOM CSS
# ============================================

# ============================================
# CUSTOM CSS - Enhanced Styling
# ============================================

st.markdown("""
<style>
/* Background gradient */
.main {
    background: linear-gradient(135deg, #fdfcfb, #e2d1c3);
}

/* Container padding */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Headings */
h1, h2, h3 {
    font-family: 'Segoe UI', sans-serif;
    color: #0F4C81;
    font-weight: 700;
}

/* Metrics cards */
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #ffffff, #f3f6f9);
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 6px 15px rgba(0,0,0,0.1);
    transition: transform 0.2s ease-in-out;
}
div[data-testid="stMetric"]:hover {
    transform: scale(1.05);
}

/* Buttons */
.stButton>button {
    width: 100%;
    height: 55px;
    background: linear-gradient(90deg, #0F4C81, #1565C0);
    color: white;
    border-radius: 12px;
    font-size: 18px;
    font-weight: bold;
    border: none;
    transition: 0.3s;
}
.stButton>button:hover {
    background: linear-gradient(90deg, #1565C0, #1E88E5);
    transform: scale(1.03);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0F4C81, #1565C0);
}
[data-testid="stSidebar"] * {
    color: white;
    font-family: 'Segoe UI', sans-serif;
}

/* Expander */
.streamlit-expanderHeader {
    font-weight: bold;
    color: #0F4C81;
}

/* Footer */
footer {
    text-align: center;
    padding: 20px;
    font-size: 14px;
    color: #444;
}
</style>
""", unsafe_allow_html=True)


# ============================================
# LOAD FILES
# ============================================

model = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")
label_encoders = joblib.load("models/label_encoders.pkl")
feature_names = joblib.load("models/feature_names.pkl")

# ============================================
# HEADER
# ============================================

st.title("📊 Customer Churn Prediction System")
st.markdown("### 🚀 AI Powered Customer Retention Analytics")

st.info("Predict whether a customer is likely to churn using Machine Learning.\n\n**Developed by:** Shivam\n\n**Model:** XGBoost Classifier")

st.divider()

# ============================================
# DASHBOARD METRICS
# ============================================

m1, m2, m3, m4 = st.columns(4)
m1.metric("📂 Dataset", "64,000+")
m2.metric("🧠 Features", "10")
m3.metric("🤖 Model", "XGBoost")
m4.metric("🎯 Accuracy", "99.992%")

st.divider()

# ============================================
# SIDEBAR
# ============================================

st.sidebar.title("📌 Project Overview")
st.sidebar.success("""
### Customer Churn Prediction

✔ Machine Learning Project  
✔ XGBoost Classifier  
✔ Customer Behaviour Analysis  
✔ Predict Customer Retention  

Developed by Shivam
""")

# ============================================
# FORM
# ============================================

st.header("📝 Customer Information")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("🎂 Age", 18, 100, 30)
        gender = st.selectbox("👤 Gender", ["Male", "Female"])
        tenure = st.number_input("📅 Tenure (Months)", 0, 60, 12)
        usage = st.number_input("📈 Usage Frequency", 0, 30, 10)
        support = st.number_input("☎ Support Calls", 0, 10, 2)

    with col2:
        payment_delay = st.number_input("💰 Payment Delay", 0, 30, 5)
        subscription = st.selectbox("📦 Subscription Type", ["Basic", "Standard", "Premium"])
        contract = st.selectbox("📄 Contract Length", ["Monthly", "Quarterly", "Annual"])
        total_spend = st.number_input("💵 Total Spend", min_value=0.0, value=500.0)
        last_interaction = st.number_input("📞 Last Interaction (Days)", 0, 30, 5)

    submitted = st.form_submit_button("🔍 Predict Customer Churn")

# ============================================
# PREDICTION
# ============================================

# ============================================
# PREDICTION
# ============================================

if submitted:

    try:

        with st.spinner("Analyzing customer information..."):
            time.sleep(1)

            # Encode categorical variables
            gender_encoded = label_encoders["Gender"].transform([gender])[0]
            subscription_encoded = label_encoders["Subscription Type"].transform([subscription])[0]
            contract_encoded = label_encoders["Contract Length"].transform([contract])[0]

            # Create Input DataFrame
            input_data = pd.DataFrame({
                "Age": [age],
                "Gender": [gender_encoded],
                "Tenure": [tenure],
                "Usage Frequency": [usage],
                "Support Calls": [support],
                "Payment Delay": [payment_delay],
                "Subscription Type": [subscription_encoded],
                "Contract Length": [contract_encoded],
                "Total Spend": [total_spend],
                "Last Interaction": [last_interaction]
            })

            # Arrange columns
            input_data = input_data[feature_names]

            # Scale
            scaled_input = scaler.transform(input_data)

            # Prediction
            prediction = model.predict(scaled_input)[0]
            probability = model.predict_proba(scaled_input)[0][1]

        st.divider()
        st.header("📊 Prediction Result")

        # -------------------------------
        # Risk Level
        # -------------------------------
        if probability >= 0.70:
            risk = "🔴 High Risk"

        elif probability >= 0.30:
            risk = "🟠 Medium Risk"

        else:
            risk = "🟢 Low Risk"

        # -------------------------------
        # Prediction Output
        # -------------------------------

        if prediction == 1:

            st.error("⚠️ Customer is Likely to Churn")

            st.metric(
                "Churn Probability",
                f"{probability*100:.2f}%"
            )

            st.metric(
                "Risk Level",
                risk
            )

            st.progress(float(probability))

            st.warning("""
### Recommended Actions

✅ Offer Loyalty Discounts

✅ Assign Dedicated Support Executive

✅ Recommend Annual Subscription

✅ Resolve Customer Complaints Quickly

✅ Follow-up Through Customer Success Team
""")

        else:

            retention = 1 - probability

            st.success("✅ Customer is Likely to Stay")

            st.metric(
                "Retention Probability",
                f"{retention*100:.2f}%"
            )

            st.metric(
                "Risk Level",
                risk
            )

            st.progress(float(retention))

            st.info("""
### Recommendation

Customer is likely to remain with the company.

Continue providing:

- Excellent Customer Support
- Personalized Offers
- Regular Engagement
- High Service Quality
""")

        # -------------------------------
        # Why this prediction?
        # -------------------------------

        st.divider()

        st.subheader("🧠 Why was this Prediction Made?")

        st.write("""
The model primarily considers the following important features:

- **Payment Delay**
- **Support Calls**
- **Tenure**
- **Usage Frequency**

Customers with higher payment delays, frequent support calls and lower engagement are more likely to churn.
""")

        # -------------------------------
        # Customer Summary
        # -------------------------------

        st.divider()

        st.subheader("📋 Customer Summary")

        summary = pd.DataFrame({
            "Feature": [
                "Age",
                "Gender",
                "Tenure",
                "Subscription",
                "Contract",
                "Total Spend"
            ],
            "Value": [
                age,
                gender,
                tenure,
                subscription,
                contract,
                total_spend
            ]
        })

        st.dataframe(
            summary.astype(str),
            width="stretch"
            )

        # -------------------------------
        # Gauge Chart
        # -------------------------------

        import plotly.graph_objects as go

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            title={'text': "Customer Churn Probability (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'steps': [
                    {'range': [0, 50], 'color': "lightgreen"},
                    {'range': [50, 80], 'color': "gold"},
                    {'range': [80, 100], 'color': "salmon"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'value': probability * 100
                }
            }
        ))

        st.plotly_chart(fig, width="stretch")

        # -------------------------------
        # Model Comparison
        # -------------------------------

        st.divider()

        with st.expander("📈 Model Comparison"):

            results = pd.DataFrame({

                "Model": [
                    "Logistic Regression",
                    "Decision Tree",
                    "Random Forest",
                    "KNN",
                    "XGBoost"
                ],

                "Accuracy (%)": [
                    83.06,
                    99.88,
                    99.95,
                    91.02,
                    99.992
                ]

            })

            st.dataframe(
                results,
                width="stretch"
                )

    except Exception as e:

        st.error(f"❌ Error: {e}")
