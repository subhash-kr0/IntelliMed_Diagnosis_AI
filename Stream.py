import streamlit as st
import numpy as np
import joblib
import pandas as pd
import base64
import google.generativeai as genai
import os
import json
from openai import OpenAI

# ---------------------- Setup Section ----------------------
st.set_page_config(
    layout='wide',
    page_icon='🧬'
)

# Fetch the API keys securely
gemini_key = st.secrets["api_keys"]["gemini"]

# Configure Gemini and OpenAI
genai.configure(api_key=gemini_key)

# Load Models & Scalers
diabetes_model = joblib.load('./models/diabetes_model.pkl')
diabetes_scaler = joblib.load('./models/diabetes_scaler.pkl')

kidney_model = joblib.load('./models/kidneyDisease_model.pkl')
heart_model = joblib.load('models/heartDisease_randomForest_model.pkl')
heart_scaler = joblib.load('models/heartDisease_scaler.pkl')
hypertension_model = joblib.load('models/hypertension_model.pkl')
breast_model = joblib.load('./models/breastCancer_randomForest_model.pkl')
lung_model = joblib.load('./models/lungCancer_XGBClassifier_model.pkl')
liver_model = joblib.load('./models/liverDisease_rf_model.pkl')

# Feature Lists
DIABETES_FEATURES = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
KIDNEY_FEATURES = ['age', 'bp', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 'bu', 'sc', 'pot', 'wc', 'htn', 'dm', 'cad', 'pe', 'ane']
HEART_FEATURES = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach','exang','oldpeak','slope','ca','thal']
HYPERTENSION_FEATURES = ['age', 'bmi', 'smoking', 'exercise', 'alcohol']
BREAST_FEATURES = ['mean_radius', 'mean_texture', 'mean_perimeter', 'mean_area', 'mean_smoothness', 'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean']
LUNG_FEATURES = ['GENDER', 'AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE', 'CHRONIC_DISEASE', 'FATIGUE', 'ALLERGY', 'WHEEZING', 'ALCOHOL_CONSUMING', 'COUGHING', 'SHORTNESS_OF_BREATH', 'SWALLOWING_DIFFICULTY', 'CHEST_PAIN']
LIVER_FEATURES = ['Age', 'Gender', 'Total_Bilirubin', 'Alkaline_Phosphotase', 'Alamine_Aminotransferace', 'Aspartate_Amino', 'Protien', 'Albumin', 'Albumin_Globulin_ratio']

# ---------------------- Helper Functions ----------------------

def get_base64(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

def load_chat_history():
    if os.path.exists("chat_history.json"):
        with open("chat_history.json", "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        return [("bot", "Welcome to IntelliMed AI Chat! How can I assist you today?")]

def save_chat_history(chat_history):
    with open("chat_history.json", "w", encoding="utf-8") as f:
        json.dump(chat_history, f)

# AI Response Functions
def get_gemini_response(message):
    gemini_model = genai.GenerativeModel(model_name="models/gemini-2.5-flash-preview-04-17")
    response = gemini_model.generate_content(message)
    return response.text

# Master AI Selector
def get_ai_response(selected_ai, user_message):
    if selected_ai == "Gemini":
        return get_gemini_response(user_message)
    else:
        return "Error: AI model not recognized!"
    


# -- Prediction Form
def prediction_form(disease_name, features, model, scaler=None):
    st.subheader(f"🩺 {disease_name} Prediction Form")

    with st.form(f"{disease_name.lower()}_form", clear_on_submit=False):
        st.markdown("### Fill the patient details:")
        cols = st.columns(3)
        inputs = {}

        for i, feature in enumerate(features):
            with cols[i % 3]:
                f_key = feature.strip()
                if f_key in FEATURE_INPUTS:
                    ftype, *params = FEATURE_INPUTS[f_key]

                    if ftype == 'slider':
                        min_val, max_val, step = params

                        # Convert to uniform types (int or float)
                        if isinstance(step, float) or isinstance(min_val, float) or isinstance(max_val, float):
                            inputs[f_key] = st.slider(f_key, float(min_val), float(max_val), float(min_val), step=float(step))
                        else:
                            inputs[f_key] = st.slider(f_key, int(min_val), int(max_val), int(min_val), step=int(step))

                    elif ftype == 'select':
                        inputs[f_key] = st.selectbox(f_key, params[0])

                else:
                    # Fallback to text input
                    inputs[f_key] = st.text_input(f_key)

        submitted = st.form_submit_button("🔍 Predict")

    if submitted:
        # Convert 'yes'/'no'/'male'/'female' etc. to numeric values
        def convert(val):
            if isinstance(val, str):
                val = val.lower()
                if val in ['yes', 'present', 'male', 'normal']: return 1
                if val in ['no', 'notpresent', 'female', 'abnormal']: return 0
            return val

        input_values = [convert(inputs[feature.strip()]) for feature in features]

        # data = np.array([input_values], dtype=float, )
        data = pd.DataFrame([input_values], columns=features)
        if scaler:
            data = scaler.transform(data)

        prediction = model.predict(data)[0]
        confidence = model.predict_proba(data)[0][1] if hasattr(model, 'predict_proba') else 0.0
        diagnosis = "Positive" if prediction == 1 else "Negative"

        st.session_state['prediction_result'] = {
            'title': disease_name,
            'inputs': inputs,
            'diagnosis': diagnosis,
            'confidence': f"{confidence:.2%}"
        }

        st.success(f"Diagnosis: {diagnosis} (Confidence: {confidence:.2%})")
        st.balloons()

        # Call your report display logic
        show_medical_report()


# -- Medical Report Page
def show_medical_report():
    result = st.session_state.get('prediction_result', {})
    if not result:
        st.error("No diagnosis found. Please submit the form first.")
        return

    st.title("🏥 Medical Diagnosis Report")
    st.write("Diagnosed by **IntelliMed AI System**")

    st.markdown("---")
    st.subheader(f"🧾 Disease: {result['title']}")
    st.subheader(f"Diagnosis Result: {'✅' if result['diagnosis'] == 'Negative' else '⚠️'} {result['diagnosis']}")
    st.subheader(f"Confidence Score: {result['confidence']}")
    st.markdown("---")

    st.markdown("### Patient Provided Details:")
    for k, v in result['inputs'].items():
        st.write(f"**{k}**: {v}")

    st.markdown("---")

    col1, col2, col3 = st.columns([1,1,1])
    if col1.button("🔁 Predict Again"):
        st.session_state['page'] = "form"
    if col3.button("🏠 Go Home"):
        st.session_state['page'] = "home"



# ---------------------- UI Section ----------------------

# Inject Sidebar Theme
st.markdown(open('sidebar.html').read(), unsafe_allow_html=True)

# Load Logo
img_base64 = get_base64("./static/logo.png")

query_params = st.query_params
page = query_params.get("page", ["home"])[0]  # Default to "home" if not present


# ---- Navbar ----
st.markdown(f"""
<div class="topnav">
    <h1>🩺 IntelliMed</h1>
    <a href="?page=home">Home</a>
    <a href="?page=about">About</a>

</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.sidebar.image("./static/logo.png", width=150)
    st.sidebar.markdown("---")

model_choice = ""
radio_button = st.sidebar.radio("", ["🤖 ChatBot", "🩺 Disease Diagnose"])


# Define all feature input types and constraints
FEATURE_INPUTS = {
    # Diabetes
    'Pregnancies': ('slider', 0, 20, 1),
    'Glucose': ('slider', 0, 300, 1),
    'BloodPressure': ('slider', 0, 200, 1),
    'SkinThickness': ('slider', 0, 100, 1),
    'Insulin': ('slider', 0, 900, 1),
    'BMI': ('slider', 0.0, 70.0, 0.1),
    'DiabetesPedigreeFunction': ('slider', 0.0, 2.5, 0.01),
    'Age': ('slider', 0, 120, 1),

    # Kidney
    'age': ('slider', 0, 100, 1),
    'bp': ('slider', 0, 200, 1),
    'al': ('slider', 0, 5, 1),
    'su': ('slider', 0, 5, 1),
    'rbc': ('select', ['normal', 'abnormal']),
    'pc': ('select', ['normal', 'abnormal']),
    'pcc': ('select', ['present', 'notpresent']),
    'ba': ('select', ['present', 'notpresent']),
    'bgr': ('slider', 0, 500, 1),
    'bu': ('slider', 0, 200, 1),
    'sc': ('slider', 0.0, 20.0, 0.1),
    'pot': ('slider', 0.0, 10.0, 0.1),
    'wc': ('slider', 0, 20000, 100),
    'htn': ('select', ['yes', 'no']),
    'dm': ('select', ['yes', 'no']),
    'cad': ('select', ['yes', 'no']),
    'pe': ('select', ['yes', 'no']),
    'ane': ('select', ['yes', 'no']),

    # Heart
    'sex': ('select', ['male', 'female']),
    'cp': ('slider', 0, 3, 1),
    'trestbps': ('slider', 80, 200, 1),
    'chol': ('slider', 100, 600, 1),
    'fbs': ('select', ['yes', 'no']),
    'restecg': ('slider', 0, 2, 1),
    'thalach': ('slider', 60, 220, 1),
    'exang': ('select', ['yes', 'no']),
    'oldpeak': ('slider', 0.0, 6.0, 0.1),
    'slope': ('slider', 0, 2, 1),
    'ca': ('slider', 0, 4, 1),
    'thal': ('slider', 0, 3, 1),

    # Hypertension
    'bmi': ('slider', 10.0, 50.0, 0.1),
    'smoking': ('select', ['yes', 'no']),
    'exercise': ('select', ['yes', 'no']),
    'alcohol': ('select', ['yes', 'no']),

    # Breast
    'mean_radius': ('slider', 5.0, 30.0, 0.1),
    'mean_texture': ('slider', 5.0, 40.0, 0.1),
    'mean_perimeter': ('slider', 30.0, 200.0, 0.1),
    'mean_area': ('slider', 100.0, 2500.0, 1.0),
    'mean_smoothness': ('slider', 0.05, 0.2, 0.001),
    'compactness_mean': ('slider', 0.0, 1.0, 0.01),
    'concavity_mean': ('slider', 0.0, 1.0, 0.01),
    'concave points_mean': ('slider', 0.0, 0.5, 0.01),
    'symmetry_mean': ('slider', 0.1, 0.5, 0.01),
    'fractal_dimension_mean': ('slider', 0.01, 0.2, 0.001),

    # Lung
    'GENDER': ('select', ['Male', 'Female']),
    'AGE': ('slider', 10, 100, 1),
    'SMOKING': ('select', ['Yes', 'No']),
    'YELLOW_FINGERS': ('select', ['Yes', 'No']),
    'ANXIETY': ('select', ['Yes', 'No']),
    'PEER_PRESSURE': ('select', ['Yes', 'No']),
    'CHRONIC_DISEASE': ('select', ['Yes', 'No']),
    'FATIGUE': ('select', ['Yes', 'No']),
    'ALLERGY': ('select', ['Yes', 'No']),
    'WHEEZING': ('select', ['Yes', 'No']),
    'ALCOHOL_CONSUMING': ('select', ['Yes', 'No']),
    'COUGHING': ('select', ['Yes', 'No']),
    'SHORTNESS_OF_BREATH': ('select', ['Yes', 'No']),
    'SWALLOWING_DIFFICULTY': ('select', ['Yes', 'No']),
    'CHEST_PAIN': ('select', ['Yes', 'No']),

    # Liver
    'Gender': ('select', ['Male', 'Female']),
    'Total_Bilirubin': ('slider', 0.0, 10.0, 0.1),
    'Alkaline_Phosphotase': ('slider', 50, 3000, 1),
    'Alamine_Aminotransferace': ('slider', 0, 2000, 1),
    'Aspartate_Amino': ('slider', 0, 2000, 1),
    'Protien': ('slider', 2.0, 10.0, 0.1),
    'Albumin': ('slider', 1.0, 6.0, 0.1),
    'Albumin_Globulin_ratio': ('slider', 0.0, 3.0, 0.1)
}






if page == "home":


    if radio_button == "🤖 ChatBot":

        
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = load_chat_history()

        choose_ai = st.sidebar.selectbox("Choose AI", ["Gemini"])

        # Chat Display
        with st.container():
            chat_html = "<div class='chat-box'>"
            for role, msg in st.session_state.chat_history:
                if role == "user":
                    chat_html += f"<div class='message user'>🧑‍💻 {msg}</div>"
                else:
                    chat_html += f"<div class='message bot'>🤖 {msg}</div>"
            chat_html += "</div>"
            st.markdown(chat_html, unsafe_allow_html=True)

        # Chat Input
        user_input = st.chat_input("Type your message...")

        if user_input:
            st.session_state.chat_history.append(("user", user_input))
            with st.spinner("Thinking..."):
                bot_reply = get_ai_response(choose_ai, user_input)
            st.session_state.chat_history.append(("bot", bot_reply))
            save_chat_history(st.session_state.chat_history)
            st.rerun()



    if radio_button == "🩺 Disease Diagnose":
      
        model_choice = st.selectbox("Select Option", ["Choose Disease", "Diabetes", "Kidney Disease", "Heart Disease", "Hypertension", "Breast Cancer", "Lung Cancer", "Liver Disease"])


        if model_choice == "Choose Disease":
            # st.info("Please select a disease model from the sidebar.")
            # Main title
            st.title("🩺 AI-Based Disease Diagnosis System")
            st.markdown("Welcome to the AI-powered health screening portal. Select a disease from the sidebar to get started with symptom-based diagnosis.")

            # Introduction
            st.markdown("""
            This intelligent diagnostic assistant helps assess your health by predicting potential conditions based on input symptoms and test data.
            Please note that this is not a substitute for professional medical advice.
            """)

            # Diseases section
            st.header("🔍 Diseases Covered")

            diseases = {
                "🧬 Diabetes": [
                    "Excessive thirst or hunger",
                    "Frequent urination",
                    "Unexplained weight loss",
                    "Fatigue",
                    "Blurred vision"
                ],
                "❤️ Heart Disease": [
                    "Chest pain or discomfort",
                    "Shortness of breath",
                    "Fatigue with exertion",
                    "Swelling in legs or feet",
                    "Irregular heartbeat"
                ],
                "🧠 Parkinson's Disease": [
                    "Tremors",
                    "Stiffness or muscle rigidity",
                    "Impaired posture and balance",
                    "Slurred speech",
                    "Slow movement"
                ],
                "🌬️ Lung Disease (optional)": [
                    "Chronic cough",
                    "Wheezing",
                    "Shortness of breath",
                    "Chest tightness",
                    "Frequent respiratory infections"
                ]
            }

            cols = st.columns(2)

            for i, (disease, symptoms) in enumerate(diseases.items()):
                with cols[i % 2]:
                    st.subheader(disease)
                    for symptom in symptoms:
                        st.markdown(f"- {symptom}")
                    st.markdown("---")

            # Footer
            st.markdown("""
            ---
            🔒 **Note**: Your data is not stored and is used only for model inference during your session.  
            💡 Start by selecting a disease from the sidebar to input your data and get a prediction.
            """)



        # Mapping model choices to their corresponding parameters
        MODEL_MAPPING = {
            "Diabetes": ("Diabetes", DIABETES_FEATURES, diabetes_model, diabetes_scaler),
            "Kidney Disease": ("Kidney Disease", KIDNEY_FEATURES, kidney_model, None),
            "Heart Disease": ("Heart Disease", HEART_FEATURES, heart_model, heart_scaler),
            "Hypertension": ("Hypertension", HYPERTENSION_FEATURES, hypertension_model, None),
            "Breast Cancer": ("Breast Cancer", BREAST_FEATURES, breast_model, None),
            "Lung Cancer": ("Lung Cancer", LUNG_FEATURES, lung_model, None),
            "Liver Disease": ("Liver Disease", LIVER_FEATURES, liver_model, None),
        }

        if model_choice == "Choose Disease":
            st.info("Please select a disease model from the sidebar.")

        elif model_choice in model_choice:
            disease_name, features, model, scaler = MODEL_MAPPING[model_choice]
            if scaler:
                prediction_form(disease_name, features, model, scaler)
            else:
                prediction_form(disease_name, features, model)



