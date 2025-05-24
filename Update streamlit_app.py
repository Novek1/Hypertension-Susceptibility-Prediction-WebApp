import streamlit as st
import pandas as pd 
import numpy as np
import joblib

# Title and description
st.markdown("<h1 style='text-align: center;'>💊🩸 Hypertension Susceptibility Prediction WebApp</h1>", unsafe_allow_html=True)
st.info("This web application predicts an individual's risk of developing hypertension using machine learning models. By analyzing key health indicators like age, BMI, blood pressure, glucose levels, and lifestyle factors (e.g., smoking, diabetes, medication use), the app provides an instant risk assessment.")
st.write("Please enter the patient's information and medical measurements below:")

# Load model and scaler
model = joblib.load("svm_model.pkl")
scaler = joblib.load("scaler.pkl")

# (Optional) hardcoded or loaded model accuracy
model_accuracy = 0.87  # change this if you have a different one

# Input form
male = st.selectbox("Gender (0: Female, 1: Male)", [0, 1])
age = st.slider("Age", 1, 100, 30)
currentSmoker = st.selectbox("Current Smoker? (0 = No, 1 = Yes)", [0.0, 1.0])
cigsPerDay = st.slider("Cigarettes per Day", 0.0, 60.0, 5.0, step=1.0)
BPMeds = st.selectbox("On BP Medication? (0 = No, 1 = Yes)", [0, 1])
diabetes = st.selectbox("Diabetic? (0 = No, 1 = Yes)", [0, 1])
totChol = st.slider("Total Cholesterol (mg/dL)", 100.0, 500.0, 200.0, step=0.1)
sysBP = st.slider("Systolic BP (mmHg)", 90.0, 250.0, 120.0, step=0.1)
diaBP = st.slider("Diastolic BP (mmHg)", 60.0, 150.0, 80.0, step=0.5)
BMI = st.slider("Body Mass Index (BMI)", 10.0, 50.0, 25.0, step=0.1)
heartRate = st.slider("Heart Rate (bpm)", 40.0, 150.0, 75.0, step=1.0)
glucose = st.slider("Glucose Level (mg/dL)", 50.0, 300.0, 100.0, step=1.0)

# Prediction
if st.button("Predict Hypertension Stage"):
    user_input = pd.DataFrame([[male, age, currentSmoker, cigsPerDay, BPMeds, diabetes,
                                totChol, sysBP, diaBP, BMI, heartRate, glucose]],
                              columns=['male', 'age', 'currentSmoker', 'cigsPerDay', 'BPMeds',
                                       'diabetes', 'totChol', 'sysBP', 'diaBP', 'BMI',
                                       'heartRate', 'glucose'])

    input_scaled = scaler.transform(user_input)
    prediction = model.predict(input_scaled)[0]  # this is a string like "Stage 1 Hypertension"
    probs = model.predict_proba(input_scaled)[0]
    confidence = np.max(probs)

    st.success(f"Predicted Hypertension Stage: **{prediction}**")
    st.success(f"Prediction Confidence: **{confidence:.2%}**")
    st.info(f"Model Accuracy: **{model_accuracy:.2%}**")

