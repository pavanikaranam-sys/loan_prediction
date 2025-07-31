import streamlit as st
import pandas as pd
import pickle

# Load model and encoder
with open("loan_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

# Set page config
st.set_page_config(page_title="Loan Prediction", layout="centered")

# Title
st.markdown(
    "<h1 style='text-align: center; color: #f4c430;'>Loan Approval Prediction</h1>",
    unsafe_allow_html=True
)

# User input form
with st.form("loan_form"):
    person_age = st.slider("Age", 18, 100, 30)
    person_income = st.number_input("Monthly Income ($)", min_value=0)
    person_home_ownership = st.selectbox("Home Ownership", ['RENT', 'OWN', 'MORTGAGE', 'OTHER'])
    person_emp_length = st.slider("Employment Length (years)", 0, 40, 5)
    loan_intent = st.selectbox("Loan Intent", ['PERSONAL', 'EDUCATION', 'MEDICAL', 'VENTURE', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION'])
    loan_grade = st.selectbox("Loan Grade", ['A', 'B', 'C', 'D', 'E', 'F', 'G'])
    loan_amnt = st.number_input("Loan Amount ($)", min_value=500, value=10000)
    loan_int_rate = st.slider("Interest Rate (%)", 0.0, 30.0, 12.5)
    cb_person_default_on_file = st.selectbox("Previous Default", ['Y', 'N'])
    cb_person_cred_hist_length = st.slider("Credit History Length (years)", 1, 50, 10)
    person_education = st.selectbox("Education", ['Master', 'High School', 'Bachelor', 'Associate', 'Doctorate'])

    submit = st.form_submit_button("Predict")

# Process and predict
if submit:
    input_dict = {
        'person_age': [person_age],
        'person_income': [person_income],
        'person_home_ownership': [person_home_ownership],
        'person_emp_length': [person_emp_length],
        'loan_intent': [loan_intent],
        'loan_grade': [loan_grade],
        'loan_amnt': [loan_amnt],
        'loan_int_rate': [loan_int_rate],
        'cb_person_default_on_file': [cb_person_default_on_file],
        'cb_person_cred_hist_length': [cb_person_cred_hist_length],
        'person_education': [person_education]
    }

    input_df = pd.DataFrame(input_dict)

    # One-hot encode the input using the loaded encoder
    input_encoded = encoder.transform(input_df).toarray()

    # Predict
    prediction = model.predict(input_encoded)[0]

    # Output
    if prediction == 1:
        st.success("✅ Loan is likely to be approved.")
    else:
        st.error("❌ Loan is likely to be rejected.")

