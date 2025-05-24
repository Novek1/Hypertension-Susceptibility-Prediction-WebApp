import streamlit as st
import pandas as pd 
import numpy as np
import joblib
st.title('💊🩸 Hypertension Susceptibility Prediction WebApp')

st.info("This web application predicts an individual's risk of developing hypertension using machine learning models. By analyzing key health indicators like age, BMI, blood pressure, glucose levels, and lifestyle factors (e.g., smoking, diabetes, medication use), the app provides an instant risk assessment.")


# Load model and scaler
model = joblib.load("svm_model.pkl")
scaler = joblib.load("scaler.pkl")

# Define feature inputs
st.title("Hypertension Stage Predictor")

male = st.selectbox("Gender (0: Female, 1: Male)", [0, 1])
age = st.slider("Age", 1, 100, 30)
currentSmoker = st.selectbox("Current Smoker? (0 = No, 1 = Yes)", [0, 1])
cigsPerDay = st.slider("Cigarettes per Day", 0.0, 60.0, 5.0)
BPMeds = st.selectbox("On BP Medication? (0 = No, 1 = Yes)", [0.0, 1.0])
diabetes = st.selectbox("Diabetic? (0 = No, 1 = Yes)", [0, 1])
totChol = st.slider("Total Cholesterol", 100.0, 500.0, 200.0)
sysBP = st.slider("Systolic BP", 90.0, 250.0, 120.0)
diaBP = st.slider("Diastolic BP", 60.0, 150.0, 80.0)
BMI = st.slider("Body Mass Index", 10.0, 50.0, 25.0)
heartRate = st.slider("Heart Rate", 40.0, 150.0, 75.0)
glucose = st.slider("Glucose Level", 50.0, 300.0, 100.0)

if st.button("Predict Hypertension Stage"):
    # Input formatting
    user_input = pd.DataFrame([[male, age, currentSmoker, cigsPerDay, BPMeds, diabetes,
                                totChol, sysBP, diaBP, BMI, heartRate, glucose]],
                              columns=['male', 'age', 'currentSmoker', 'cigsPerDay', 'BPMeds',
                                       'diabetes', 'totChol', 'sysBP', 'diaBP', 'BMI',
                                       'heartRate', 'glucose'])
    
    # Scale input
    input_scaled = scaler.transform(user_input)
    
    # Make prediction
    prediction = model.predict(input_scaled)[0]
    
    # Label mapping
    labels = ["Normal", "Elevated", "Stage 1 Hypertension", "Stage 2 Hypertension", "Hypertension Crisis"]
    st.success(f"Predicted Hypertension Stage: **{labels[prediction]}**")

streamlit run app.py

