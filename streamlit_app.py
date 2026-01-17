import streamlit as st
import os
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import accuracy_score

######################## TO RUN:
# 1 INSTALL STREAMLIT
# 2 streamlit run streamlit_app.py
MODEL_PATH = "loan_model.joblib"

st.set_page_config(
    page_title="Loan Approval Checker",
    page_icon="🏦",
    layout="centered"
)

# Title
st.title("🏦 Loan Approval Checker")



# If model missing -> show waiting message and stop
if not os.path.exists(MODEL_PATH):
    st.warning("The system is initializing, please wait")
    st.error("The model file was not found. Please train the model and save it before running the app.")
    st.stop()


@st.cache_resource
def load_model(path: str):
    return joblib.load(path)

model = load_model(MODEL_PATH)


# ---- Input form (screenshot-friendly UI) ----
with st.form("loan_form"):
    st.subheader("Loan Application Details")

    applicant_income = st.number_input(
        "Applicant Income (monthly)",
        min_value=0.0,
        value=5000.0,
        step=100.0
    )

    coapplicant_income = st.number_input(
        "Coapplicant Income (monthly)",
        min_value=0.0,
        value=0.0,
        step=100.0
    )

    loan_amount = st.number_input(
        "Requested Loan Amount",
        min_value=0.0,
        value=150.0,
        step=10.0
    )

    loan_term = st.number_input(
        "Loan Term (months)",
        min_value=1.0,
        value=360.0,
        step=12.0
    )

    credit_history = st.selectbox(
        "Credit History",
        options=[1, 0],
        index=0,
        format_func=lambda x: "Exists (1)" if x == 1 else "Does not exist (0)"
    )
    dict_opt = {"Yes": "Married", "No": "Single"}
    married = st.selectbox(
        "Marital Status",
        options=["Yes", "No"],
        index=0,
        format_func=lambda x: dict_opt[x]
    )

    submitted = st.form_submit_button("Check Loan Eligibility")


# Runs only after clicking the button
if submitted:
    # Convert Married to dummy (drop_first=True → keep only Married_Yes)
    married_yes = 1 if married == "Yes" else 0

    # IMPORTANT:
    # Column names must exactly match those used during model training
    X = pd.DataFrame([{
        "ApplicantIncome": float(applicant_income),
        "CoapplicantIncome": float(coapplicant_income),
        "LoanAmount": float(loan_amount),
        "Loan_Amount_Term": float(loan_term),
        "Credit_History": int(credit_history),
        "Married_Yes": int(married_yes),
    }])
    print(X)

    predict = model.predict(X)[0]
    probability = model.predict_proba(X)[0][1]

    if predict == 1:
        st.success(f"loan approved with probability of: {probability:.2f})")
    else:
        st.error(f"loan rejected with probability of: {probability:.2f})")

if st.button("model accuracy"):
    loan_df = pd.read_csv("train.csv")
    loan_df["Loan_Status"] = np.where(loan_df["Loan_Status"] == "Y", 1, 0)
    loan_df = loan_df.fillna(loan_df.mean(numeric_only=True))
    loan_df = pd.get_dummies(loan_df, drop_first=True)

    X = loan_df[[
        "ApplicantIncome",
        "CoapplicantIncome",
        "LoanAmount",
        "Loan_Amount_Term",
        "Credit_History",
        "Married_Yes"
    ]]
    Y = loan_df["Loan_Status"]
    y_pred = model.predict(X)
    accuracy = accuracy_score(Y, y_pred)

    st.info(f"model accuracy is: {accuracy*100:.2f}%")