import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Page Config
st.set_page_config(page_title="Telco Churn Predictor", layout="wide", page_icon="📡")

# Custom CSS for aesthetics
st.markdown("""
<style>
    .stProgress > div > div > div > div {
        background-color: #ff4b4b;
    }
    .main-header {
        font-family: 'Inter', sans-serif;
        color: #1E3A8A;
        font-weight: 800;
        text-align: center;
        margin-bottom: 30px;
    }
    div[data-testid="stSidebar"] {
        background-color: #f8fafc;
        border-right: 1px solid #e2e8f0;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 class='main-header'>📡 Telco Customer Churn Predictor</h1>", unsafe_allow_html=True)
st.markdown("Predict the likelihood of a customer leaving based on their demographics, services, and billing.")

@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

try:
    pipeline = load_model()
except Exception as e:
    st.warning("Model version mismatch or missing. Retraining automatically... please wait ~5 seconds.")
    import train_model
    st.cache_resource.clear()
    pipeline = load_model()
    st.success("Model retrained successfully for this environment!")

# Layout
st.sidebar.header("Customer Profile")
st.sidebar.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Demographics")
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])

with col2:
    st.subheader("Services Subscribed")
    phone = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    
    if internet != "No":
        security = st.selectbox("Online Security", ["Yes", "No"])
        backup = st.selectbox("Online Backup", ["Yes", "No"])
        protection = st.selectbox("Device Protection", ["Yes", "No"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No"])
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No"])
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No"])
    else:
        security = backup = protection = tech_support = streaming_tv = streaming_movies = "No internet service"

with col3:
    st.subheader("Billing & Account")
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    
    tenure = st.slider("Tenure (Months)", 0, 72, 12)
    monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 150.0, 70.0)
    total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, float(tenure * monthly_charges))

# Predict Button
st.markdown("---")
col_btn, _ = st.columns([1, 4])
with col_btn:
    predict_btn = st.button("🔮 Predict Churn Probability", type="primary", use_container_width=True)

if predict_btn:
    # Build dataframe
    input_data = pd.DataFrame([{
        'gender': gender,
        'SeniorCitizen': senior,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone,
        'MultipleLines': multiple_lines,
        'InternetService': internet,
        'OnlineSecurity': security,
        'OnlineBackup': backup,
        'DeviceProtection': protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless,
        'PaymentMethod': payment,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }])
    
    try:
        prob = pipeline.predict_proba(input_data)[0][1]
        
        st.markdown("### Prediction Results")
        res_col1, res_col2 = st.columns([1, 2])
        
        with res_col1:
            if prob > 0.5:
                st.error("⚠️ HIGH CHURN RISK")
            elif prob > 0.3:
                st.warning("⚠️ MEDIUM CHURN RISK")
            else:
                st.success("✅ LOW CHURN RISK")
                
        with res_col2:
            st.markdown(f"**Churn Probability:** {prob*100:.1f}%")
            st.progress(float(prob))
            
    except Exception as e:
        st.error(f"Error making prediction: {e}")
