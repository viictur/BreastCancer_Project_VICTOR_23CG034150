import streamlit as st
import numpy as np
import joblib


checkpoint = joblib.load('model/breast_cancer_model.pkl')
model = checkpoint['model']
scaler = checkpoint['scaler']

st.title("Breast Cancer Prediction System")
st.write("Enter the tumor features to predict diagnosis (Benign or Malignant)")

radius = st.number_input("radius_mean", min_value=0.0, format="%.4f")
texture = st.number_input("texture_mean", min_value=0.0, format="%.4f")
perimeter = st.number_input("perimeter_mean", min_value=0.0, format="%.4f")
area = st.number_input("area_mean", min_value=0.0, format="%.4f")
smoothness = st.number_input("smoothness_mean", min_value=0.0, format="%.4f")

if st.button("Predict"):
    features = np.array([[radius, texture, perimeter, area, smoothness]])
    features_scaled = scaler.transform(features)
    
    prediction = model.predict(features_scaled)
    
    result = "Benign" if prediction[0] == 1 else "Malignant"
    
    if result == "Malignant":
        st.error(f"The result is: {result}")
    else:
        st.success(f"The result is: {result}")

st.caption("Disclaimer: This system is for educational purposes and not a medical tool.")