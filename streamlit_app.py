import streamlit as st
import pickle

with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

st.title("Diabetes Prediction App")

#inputs
Pregnancies = st.number_input('Pregnancies' , min_value=0.0 , max_value=10.0,value=1.0)
Glucose = st.number_input('Glucose' , min_value=0.0 , max_value=10.0,value=1.0)
BMI =  st.number_input(' BMI' , min_value=0.0 , max_value=100.0,value=1.0)

output = model.predict([[Pregnancies,Glucose,BMI]])
st.write("the predict : ",output[0])

