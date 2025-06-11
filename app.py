import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load("Logistic_Model.pkl")

st.title("🚢 Titanic Survival Predictor")

# Input fields
passenger_id = st.number_input("Passenger ID", min_value=1)
pclass = st.selectbox("Passenger Class", [1, 2, 3])
age = st.number_input("Age", min_value=0.0, max_value=100.0, value=25.0)
sibsp = st.number_input("Siblings/Spouses Aboard", min_value=0, max_value=10)
parch = st.number_input("Parents/Children Aboard", min_value=0, max_value=10)
fare = st.number_input("Fare", min_value=0.0, value=30.0)

# Categorical encoding (must match your one-hot training data)
sex = st.selectbox("Sex", ["male", "female"])
sex_male = 1 if sex == "male" else 0  # Only one dummy column used: Sex_male

embarked = st.selectbox("Embarked (Port)", ["S", "C", "Q"])
embarked_s = 1 if embarked == "S" else 0  # Only one dummy column: Embarked_S

# Create input vector (must match model training order)
input_data = np.array([[passenger_id, pclass, age, sibsp, parch, fare, sex_male, embarked_s]])

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_data)
    result = "🟢 Survived" if prediction[0] == 1 else "🔴 Did Not Survive"
    st.success(f"Prediction: **{result}**")
