import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

# loading the data into the python file
loan_df = pd.read_csv("train.csv")

# choosing the features for the dataframe
features = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
    "Married"
]

# creating the right features and adding a "label" column
loan_df = loan_df[features + ["Loan_Status"]]

# changing the label from string to numeric
loan_df["Loan_Status"] = np.where(loan_df["Loan_Status"] == "Y", 1, 0)

# filling all numeric NaN with column mean
loan_df = loan_df.fillna(loan_df.mean(numeric_only=True))

# changing non-numeric values with 0/1 by creating more columns
loan_df = pd.get_dummies(loan_df, drop_first=True)

# creating X and Y values to train the model
X_train = loan_df[[
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
    "Married_Yes"
]]
Y_train = loan_df["Loan_Status"]

# creating pipeline with scaler and SVC
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svc", SVC(probability=True))
])

# training the model
pipeline.fit(X_train, Y_train)

# saving the trained model and pipeline
joblib.dump(pipeline, "loan_model.joblib")


















