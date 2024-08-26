import streamlit as st
import pickle

# Load the trained model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Title and introduction
st.title("Diabetes Prediction App")
st.write("This app predicts the likelihood of diabetes based on user input using a trained model.")

# User input fields
Pregnancies = st.number_input('Pregnancies', min_value=0, max_value=17)
Glucose = st.number_input('Glucose', min_value=0, max_value=199)
BMI = st.number_input('BMI', min_value=0.0, max_value=67.1)

# Prepare input data
input_data = [[Pregnancies, Glucose, BMI]]

# Make prediction
output = model.predict(input_data)

# Display the result
st.write(f"The prediction is: {'Diabetes' if output[0] == 1 else 'No Diabetes'}")

# Optional: Show prediction probability if the model supports it
if hasattr(model, 'predict_proba'):
    prediction_proba = model.predict_proba(input_data)
    st.write("Prediction Probability:")
    st.write(prediction_proba)
