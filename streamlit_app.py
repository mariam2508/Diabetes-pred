import streamlit as st
import pickle

with open('model.pkl', 'rb') as file:
    model = pickle.load(file)


# Title and introduction
st.title('Diabetes Prediction App')
st.write('This app predicts the likelihood of diabetes based on user input using an SVM model.')

# Sidebar for user input
st.sidebar.header('Input Features')

def user_input_features():
    Pregnancies = st.sidebar.number_input('Pregnancies', min_value=0, max_value=20, value=0)
    Glucose = st.sidebar.slider('Glucose', 0, 200, 120)
    BMI = st.sidebar.slider('BMI', 0.0, 50.0, 25.0)

    data = {
        'Pregnancies': Pregnancies,
        'Glucose': Glucose,
        'BMI': BMI
    }
    
    # Create the DataFrame
    features = pd.DataFrame(data, index=[0])
    return features

# Display input features
input_df = user_input_features()
st.write('Input features')
st.write(input_df)

# Make predictions using the loaded model
prediction = model.predict(input_df)

# Use predict_proba only if the model supports it
if hasattr(model, 'predict_proba'):
    prediction_proba = model.predict_proba(input_df)
    st.write('Prediction Probability')
    st.write(prediction_proba)
else:
    st.write("Model does not support probability prediction.")

# Display results
st.write(f'Prediction (0: No Diabetes, 1: Diabetes)')
st.write(prediction[0])
