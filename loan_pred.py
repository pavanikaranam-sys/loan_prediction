import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model
model = pickle.load(open('loan_model.pkl', 'rb'))

# Title and styling
st.set_page_config(page_title="Loan Prediction", layout="centered")
st.markdown(
    "<h1 style='text-align: center; color: #f4c430;'>Loan Approval Prediction</h1>",
    unsafe_allow_html=True
)

st.markdown("Enter applicant details below to check loan eligibility.")

# Input form
with st.form("loan_form"):
    Gender = st.selectbox("Gender", ["Male", "Female"])
    Married = st.selectbox("Married", ["Yes", "No"])
    Education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    Self_Employed = st.selectbox("Self Employed", ["Yes", "No"])
    ApplicantIncome = st.number_input("Applicant Income", min_value=0)
    CoapplicantIncome = st.number_input("Coapplicant Income", min_value=0)
    LoanAmount = st.number_input("Loan Amount (in thousands)", min_value=0)
    Loan_Amount_Term = st.selectbox("Loan Term (in months)", [360, 120, 180, 240, 300, 60, 84, 36, 12])
    Credit_History = st.selectbox("Credit History", [1.0, 0.0])
    Property_Area = st.selectbox("Property Area", ["Urban", "Rural", "Semiurban"])
    
    submit = st.form_submit_button("Predict")

# Preprocess input and predict
if submit:
    # Mapping
    gender_map = {'Male': 1, 'Female': 0}
    married_map = {'Yes': 1, 'No': 0}
    education_map = {'Graduate': 1, 'Not Graduate': 0}
    self_employed_map = {'Yes': 1, 'No': 0}
    property_map = {'Urban': 2, 'Rural': 0, 'Semiurban': 1}

    # Convert input to model format
    input_data = np.array([[
        gender_map[Gender],
        married_map[Married],
        education_map[Education],
        self_employed_map[Self_Employed],
        ApplicantIncome,
        CoapplicantIncome,
        LoanAmount,
        Loan_Amount_Term,
        Credit_History,
        property_map[Property_Area]
    ]])

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")

