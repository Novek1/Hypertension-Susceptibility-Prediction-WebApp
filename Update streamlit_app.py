import streamlit as st
import pandas as pd 
import numpy as np

st.title('💊🩸 Hypertension Susceptibility Prediction WebApp')

st.info("This web application predicts an individual's risk of developing hypertension using machine learning models. By analyzing key health indicators like age, BMI, blood pressure, glucose levels, and lifestyle factors (e.g., smoking, diabetes, medication use), the app provides an instant risk assessment.")
df= pd.read_csv("/workspaces/Hypertension-Susceptibility-Prediction-WebApp/Hypertension_Cleaned(2).csv")
df=df.drop('Unnamed: 0', axis=1)# delete unwanted index column
df= df.drop('Risk', axis=1) # delete the binary risk column
df
