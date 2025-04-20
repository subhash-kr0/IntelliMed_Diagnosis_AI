import streamlit as st
import numpy as np
import pickle  # Assuming you're using a trained model

# Load your trained model (optional if already trained)
# model = pickle.load(open('diabetes_model.pkl', 'rb'))

st.set_page_config(page_title="Diabetes Predictor", page_icon="💉")

st.title("💉 Diabetes Diagnosis Form")

# Define the input fields (same as in your Flask template)
features = [
    "pregnancies", "glucose", "blood_pressure", "skin_thickness",
    "insulin", "bmi", "diabetes_pedigree_function", "age"
]

user_input = {}

with st.form(key='diabetes_form'):
    for feature in features:
        user_input[feature] = st.number_input(
            label=feature.replace('_', ' ').title(),
            min_value=0.0, format="%.2f", key=feature
        )
    
    submit_button = st.form_submit_button(label='Predict')

# Prediction logic after form submission
if submit_button:
    input_values = np.array(list(user_input.values())).reshape(1, -1)
    
    # You can replace this dummy logic with your model's prediction
    # prediction = model.predict(input_values)
    prediction = [1 if user_input["glucose"] > 125 else 0]  # Dummy logic

    if prediction[0] == 1:
        st.error("⚠️ The model predicts that you might have diabetes.")
    else:
        st.success("✅ The model predicts that you are not diabetic.")
