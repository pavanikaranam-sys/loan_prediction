# Email Spam Detection

## 📌 Project Overview

This machine learning project that can predict whether a loan application will be approved based on customer details.

## ✅ Steps Performed

1. **Data Collection**  
   Loaded the dataset (`load_pred.csv`) containing email features and labels.

2. **Data Cleaning**  
   - Checked for null values  
   - Handled outliers using the IQR method
   - Encoded columns like person_gender,previous_loan_defaults_on_file using label encoder and person_education,person_home_ownership,loan_intent using get_dummies

3. **Data Visualization**  
   - Used box plots and histograms to explore distributions 

4. **Feature engineering**
   - Removed unnecessary features like `loan_status`
   - Split the dataset into training and testing sets

5. **Model Selection & Training**
   - Used **Random Forest Classification** with `class_weight='balanced'`  
   - Evaluated the model with classification metrics and a confusion matrix
   - Achieved a classification accuracy of **~93%**, with:
       - High precision and recall for approved applications
       - Balanced model avoiding overfitting

##Dataset

    The dataset used contains fields like:
    - person_age
    - person_gender
    - person_income
    - person_emp_exp
    - loan_amnt
    - loan_int_rate
    - loan_percent_income
    - cb_person_cred_hist_length
    - credit_score
    - previous_loan_defaults_on_file
    - loan_status
    - Bachelor
    - Doctorate
    - High School
    - Master
    - MORTGAGE
    - OWN
    - RENT
    - EDUCATION
    - HOMEIMPROVEMENT
    - MEDICAL
    - PERSONAL
    - VENTURE

## 📊 Libraries Used

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
