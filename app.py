import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("Logistic_Model.pkl")
st.write("Model expects", model.n_features_in_, "features")

# Title
st.title("Logistic Regression Prediction App")

# Input features (make sure these match the features used in training)
age = st.number_input("Age", min_value=0)
fare = st.number_input("Fare", min_value=0.0)
pclass = st.selectbox("Pclass", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])

# Convert categorical to numerical
sex = 1 if sex == "male" else 0

# Create input array
input_data = np.array([[age, fare, pclass, sex]])

# Predict
if st.button("Predict"):
    prediction = model.predict(input_data)
    st.write("Prediction:", "Survived" if prediction[0] == 1 else "Did not survive")
