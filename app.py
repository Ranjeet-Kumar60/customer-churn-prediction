# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and preprocessor
model = joblib.load("final_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")
top_features = joblib.load("top_10_features.pkl")

# UI title
st.title("📉 Customer Churn Prediction App")

st.markdown("Enter customer details to predict if they are likely to churn.")

# Input fields (20 original features)
gender = st.selectbox("Gender", ["Male", "Female"])
seniorcitizen = st.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.selectbox("Partner", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["Yes", "No"])
tenure = st.number_input("Tenure (in months)", min_value=0, max_value=100, value=1)
phoneservice = st.selectbox("Phone Service", ["Yes", "No"])
multiplelines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
internetservice = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
onlinesecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
onlinebackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
deviceprotection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
techsupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
streamingtv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
streamingmovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperlessbilling = st.selectbox("Paperless Billing", ["Yes", "No"])
paymentmethod = st.selectbox("Payment Method", ["Electronic check", "Mailed check", 
                                                 "Bank transfer (automatic)", "Credit card (automatic)"])
monthlycharges = st.number_input("Monthly Charges", min_value=0.0, value=50.0)
totalcharges = st.number_input("Total Charges", min_value=0.0, value=100.0)


# Create input DataFrame
input_dict = {
    "gender": gender,
    "seniorcitizen": seniorcitizen,
    "partner": partner,
    "dependents": dependents,
    "tenure": tenure,
    "phoneservice": phoneservice,
    "multiplelines": multiplelines,
    "internetservice": internetservice,
    "onlinesecurity": onlinesecurity,
    "onlinebackup": onlinebackup,
    "deviceprotection": deviceprotection,
    "techsupport": techsupport,
    "streamingtv": streamingtv,
    "streamingmovies": streamingmovies,
    "contract": contract,
    "paperlessbilling": paperlessbilling,
    "paymentmethod": paymentmethod,
    "monthlycharges": monthlycharges,
    "totalcharges": totalcharges,
}

input_df = pd.DataFrame([input_dict])

# Predict button
if st.button("Predict Churn"):
    try:
        # Preprocess input
        transformed = preprocessor.transform(input_df)
        input_transformed_df = pd.DataFrame(transformed, columns=preprocessor.get_feature_names_out())

        # Select only top 10 features
        selected_features = input_transformed_df[top_features]

        # Predict
        churn_prob = model.predict_proba(selected_features)[0][1]
        churn_label = "Yes" if churn_prob > 0.5 else "No"

        # Show result
        st.subheader("Prediction Result")
        st.write(f"🔮 **Will the customer churn?**: **{churn_label}**")
        st.write(f"🧮 **Churn Probability**: `{churn_prob:.2%}`")

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
