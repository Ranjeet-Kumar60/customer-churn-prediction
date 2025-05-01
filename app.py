import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# Load artifacts
model = joblib.load("final_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")
top_10_features = joblib.load("top_10_features.pkl")


# Load training data to get monthly charge range
train_data = pd.read_csv("data/train.csv")
min_charge = float(train_data["monthlycharges"].min())
max_charge = float(train_data["monthlycharges"].max())

st.set_page_config(page_title="Customer Churn Prediction", layout="centered")
st.title("💡 Customer Churn Prediction App")
st.markdown("Enter customer details to predict if they are likely to churn.")

# Input Fields
gender = st.selectbox("Gender", ["Male", "Female"])
senior_citizen = st.selectbox("Senior Citizen", ["Yes", "No"])
partner = st.selectbox("Partner", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["Yes", "No"])
tenure = st.number_input("Tenure (in months)", min_value=0, max_value=72, step=1)
phoneservice = st.selectbox("Phone Service", ["Yes", "No"], key="phoneservice")
multiplelines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"], key="multiplelines")
internetservice = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
onlinesecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
onlinebackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
deviceprotection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
techsupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
streamingtv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
streamingmovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperlessbilling = st.selectbox("Paperless Billing", ["Yes", "No"])
paymentmethod = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
monthlycharges = st.number_input("Monthly Charges", min_value=min_charge, max_value=max_charge, step=1.0)
totalcharges = tenure * monthlycharges
st.markdown(f"**Automatically Calculated Total Charges:** ₹{totalcharges:.2f}")

# Input Dictionary
input_data = {
    'gender': gender,
    'seniorcitizen': senior_citizen,
    'partner': partner,
    'dependents': dependents,
    'tenure': tenure,
    'phoneservice': phoneservice,
    'multiplelines': multiplelines,
    'internetservice': internetservice,
    'onlinesecurity': onlinesecurity,
    'onlinebackup': onlinebackup,
    'deviceprotection': deviceprotection,
    'techsupport': techsupport,
    'streamingtv': streamingtv,
    'streamingmovies': streamingmovies,
    'contract': contract,
    'paperlessbilling': paperlessbilling,
    'paymentmethod': paymentmethod,
    'monthlycharges': monthlycharges,
    'totalcharges': totalcharges
}

# Validate Inputs
def is_valid_combination(contract, tenure, monthlycharges, totalcharges):
    if tenure <= 0:
        return False, "Tenure must be greater than 0."
    if monthlycharges <= 0:
        return False, "Monthly Charges must be positive."
    if totalcharges <= 0:
        return False, "Total Charges must be positive."
    if contract == "Two year" and tenure < 24:
        return False, "Two-year contract requires at least 24 months of tenure."
    if contract == "One year" and tenure < 12:
        return False, "One-year contract requires at least 12 months of tenure."

    expected_total = monthlycharges * tenure
    if abs(expected_total - totalcharges) > 50:
        return False, f"Total Charges should roughly equal Monthly Charges × Tenure. Expected around ₹{expected_total:.2f}."

    return True, None

# Save invalid inputs to log
def log_invalid_input(data, error):
    row = data.copy()
    row["error"] = error
    row["timestamp"] = datetime.now().isoformat()
    df_log = pd.DataFrame([row])
    df_log.to_csv("invalid_inputs_log.csv", mode="a", header=False, index=False, encoding="utf-8")

# Predict
if st.button("Predict Churn"):
    valid, error = is_valid_combination(contract, tenure, monthlycharges, totalcharges)
    if not valid:
        log_invalid_input(input_data, error)
        st.warning(f"⚠️ {error}\n\nPlease enter realistic values that match a real customer's data.")
    else:
        input_df = pd.DataFrame([input_data])
        input_df['seniorcitizen'] = input_df['seniorcitizen'].map({"Yes": 1, "No": 0})

        X_transformed = preprocessor.transform(input_df)
        X_transformed_df = pd.DataFrame(X_transformed, columns=preprocessor.get_feature_names_out())
        X_selected = X_transformed_df[top_10_features]

        churn_prob = model.predict_proba(X_selected)[0][1]
        churn_pred = model.predict(X_selected)[0]

        st.markdown("---")
        st.subheader("Prediction Result")
        st.write("🔍 **Churn Probability:**", f"{churn_prob:.2f}")
        if churn_pred == 1:
            st.error("⚠️ This customer might leave the service.")
        else:
            st.success("✅ This customer is likely to stay.")





