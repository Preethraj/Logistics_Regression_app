import streamlit as st
import joblib
import numpy as np

# Load your trained model
model = joblib.load("Logistic_Model.pkl")

# App title
st.title("Logistic Regression Predictor")

st.write("""
This app uses a trained Logistic Regression model to predict outcomes based on user input.
""")

# Example input fields – replace or add as per your actual model
feature1 = st.number_input("Enter Feature 1", value=0.0)
feature2 = st.number_input("Enter Feature 2", value=0.0)
feature3 = st.number_input("Enter Feature 3", value=0.0)

# Predict button
if st.button("Predict"):
    input_data = np.array([[feature1, feature2, feature3]])  # Adjust number of features
    prediction = model.predict(input_data)
    st.success(f"Predicted Class: {prediction[0]}")
