import streamlit as st
import joblib
import numpy as np
import warnings
warnings.filterwarnings("ignore")

model = joblib.load('diabetes_prediction_model.pkl')
st.title('Diabetes Prediction')

Pregnancies = st.number_input('Pregnancies')
Glucose = st.number_input('Glucose')
BloodPressure = st.number_input('BloodPressure')
SkinThickness = st.number_input('SkinThickness')
Insulin = st.number_input('Insulin')
BMI = st.number_input('BMI')
DiabetesPedigreeFunction = st.number_input('DiabetesPedigreeFunction')
Age = st.number_input('Age')

if st.button('Predict'):
    features = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
    prediction = model.predict(features)
    result = int(prediction[0])
    st.write('Prediction:', 'Diabetic' if result == 1 else 'Not Diabetic')


