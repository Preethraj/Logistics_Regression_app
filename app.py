import streamlit as st
import numpy as np
import joblib

# Load your trained model
model = joblib.load("logistic_model.pkl")

# App title
st.title("Titanic Survival Prediction")

# User Inputs
pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.number_input("Age", min_value=0.0, max_value=100.0, value=25.0)
sibsp = st.number_input("Number of Siblings/Spouses Aboard (SibSp)", min_value=0, max_value=10, value=0)
parch = st.number_input("Number of Parents/Children Aboard (Parch)", min_value=0, max_value=10, value=0)
fare = st.number_input("Fare", min_value=0.0, value=50.0)
embarked = st.selectbox("Port of Embarkation (Embarked)", ["C", "Q", "S"])

# Encode categorical inputs
sex_encoded = 1 if sex == "male" else 0
embarked_map = {"C": 0, "Q": 1, "S": 2}
embarked_encoded = embarked_map[embarked]

# Prepare input for model
input_data = np.array([[pclass, sex_encoded, age, sibsp, parch, fare, embarked_encoded]])

# Predict
if st.button("Predict"):
    prediction = model.predict(input_data)
    result = "Survived" if prediction[0] == 1 else "Did not survive"
    st.success(f"The passenger likely **{result}**.")
