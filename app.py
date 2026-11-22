

import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Credit Scoring System",
    page_icon="💳",
    layout="centered"
)

model = joblib.load("models/random_forest.pkl")
df = pd.read_csv("dataset/credit_data.csv")

target_column = "class"
X = df.drop(columns=[target_column])

# Categorical columns used during model training
categorical_columns = [
    "checking_account",
    "credit_history",
    "purpose",
    "savings",
    "employment",
    "personal_status",
    "other_debtors",
    "property",
    "other_installment",
    "housing",
    "job",
    "telephone",
    "foreign_worker"
]

st.title("💳 Credit Scoring Prediction System")
st.write("Enter customer financial details below.")

customer_data = {}

for column in X.columns:

    if column in categorical_columns:
        options = sorted(
            X[column].dropna().astype(str).unique().tolist()
        )

        customer_data[column] = st.selectbox(
            column.replace("_", " ").title(),
            options
        )

    else:
        numeric_values = pd.to_numeric(X[column], errors="coerce")

        customer_data[column] = st.number_input(
            column.replace("_", " ").title(),
            min_value=float(numeric_values.min()),
            max_value=float(numeric_values.max()),
            value=float(numeric_values.median())
        )

if st.button("Predict Credit Class"):

    input_df = pd.DataFrame([customer_data])

    for column in categorical_columns:
        if column in input_df.columns:
            input_df[column] = input_df[column].astype(str)

    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]

    st.subheader("Prediction Result")
    st.success(f"Predicted Class: {prediction}")

    st.subheader("Prediction Probabilities")

    for class_name, probability in zip(model.classes_, probabilities):
        st.write(f"{class_name}: {probability * 100:.2f}%")
