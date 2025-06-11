import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load("Logistic_Model.pkl")

st.title("🚢 Titanic Survival Predictor")

# Inputs
passenger_id = st.number_input("Passenger ID", min_value=1)
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.number_input("Age", min_value=0.0, max_value=100.0, value=25.0)
sibsp = st.number_input("Siblings/Spouses Aboard (SibSp)", min_value=0, max_value=10)
parch = st.number_input("Parents/Children Aboard (Parch)", min_value=0, max_value=10)
fare = st.number_input("Fare", min_value=0.0, value=30.0)
embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"])

# Encode categorical values
sex_encoded = 1 if sex == "male" else 0
embarked_map = {"S": 0, "C": 1, "Q": 2}
embarked_encoded = embarked_map[embarked]

# Create input data in correct order
input_data = np.array([[passenger_id, pclass, sex_encoded, age, sibsp, parch, fare, embarked_encoded]])

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_data)
    result = "🟢 Survived" if prediction[0] == 1 else "🔴 Did Not Survive"
    st.success(f"Prediction: **{result}**")
