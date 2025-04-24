import streamlit as st
import joblib
import numpy as np

# Load models
breast_model = joblib.load('models/breastCancer_randomForest_model.pkl')
diabetes_model = joblib.load('models/diabetes_model.pkl')
liver_model = joblib.load('models/kidneyDisease_model.pkl')

st.set_page_config(page_title="Medical Diagnosis AI", layout="wide")
st.title("🩺 IntelliMed: Disease Diagnosis & Chatbot")

menu = st.sidebar.radio("Select Option", ["🏥 Breast Cancer", "💉 Diabetes", "🧬 Liver Disease", "🤖 Chatbot"])

# --- Breast Cancer ---
if menu == "🏥 Breast Cancer":
    st.header("Breast Cancer Prediction")

    features = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
                'smoothness_mean', 'compactness_mean', 'concavity_mean',
                'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean']
    
    inputs = [st.number_input(f"{feat.replace('_', ' ').title()}", value=1.0) for feat in features]
    
    if st.button("Predict Breast Cancer"):
        data = np.array(inputs).reshape(1, -1)
        pred = breast_model.predict(data)[0]
        prob = breast_model.predict_proba(data)[0][1]
        diagnosis = "Malignant" if pred == 1 else "Benign"
        st.success(f"Diagnosis: {diagnosis} (Confidence: {prob:.2%})")

# --- Diabetes ---
elif menu == "💉 Diabetes":
    st.header("Diabetes Prediction")

    features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    
    inputs = [st.number_input(f"{feat}", value=1.0) for feat in features]

    if st.button("Predict Diabetes"):
        data = np.array(inputs).reshape(1, -1)
        pred = diabetes_model.predict(data)[0]
        st.success("Diabetic" if pred == 1 else "Non-Diabetic")

# --- Liver Disease ---
elif menu == "🧬 Liver Disease":
    st.header("Liver Disease Prediction")

    features = ['Age', 'Total_Bilirubin', 'Direct_Bilirubin', 'Alkaline_Phosphotase',
                'Alamine_Aminotransferase', 'Aspartate_Aminotransferase', 'Total_Proteins',
                'Albumin', 'Albumin_and_Globulin_Ratio']
    
    inputs = [st.number_input(f"{feat.replace('_', ' ').title()}", value=1.0) for feat in features]

    if st.button("Predict Liver Disease"):
        data = np.array(inputs).reshape(1, -1)
        pred = liver_model.predict(data)[0]
        st.success("Liver Disease Detected" if pred == 1 else "Healthy")

# --- Chatbot ---
elif menu == "🤖 Chatbot":
    st.header("Chat with MedBot 💬")
    user_input = st.text_input("Ask anything medical...")

    if user_input:
        # Dummy response – replace with real API/LLM
        response = f"🤖: I received your question: '{user_input}'"
        st.write(response)
