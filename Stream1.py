import streamlit as st
import joblib
import google.generativeai as genai
import numpy as np
import pandas as pd
import json
import base64
from datetime import datetime
import os

# Page config
st.set_page_config(layout='wide', page_icon='🎈')

# API keys
gemini_key = st.secrets["api_keys"]["gemini"]

# Configure Gemini
genai.configure(api_key=gemini_key)
model = genai.GenerativeModel(model_name="models/gemini-2.0-flash-lite")

# Load models and scalers
diabetes_model = joblib.load('./models/diabetes_model.pkl')
diabetes_scaler = joblib.load('./models/diabetes_scaler.pkl')
kidney_model = joblib.load('./models/kidneyDisease_model.pkl')
heart_model = joblib.load('models/heartDisease_randomForest_model.pkl')
heart_scaler = joblib.load('models/heartDisease_scaler.pkl')
hypertension_model = joblib.load('models/hypertension_model.pkl')
breast_model = joblib.load('./models/breastCancer_randomForest_model.pkl')
lung_model = joblib.load('./models/lungCancer_XGBClassifier_model.pkl')
liver_model = joblib.load('./models/liverDisease_rf_model.pkl')

# Feature lists
DIABETES_FEATURES = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
KIDNEY_FEATURES = ['age', 'bp', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 'bu', 'sc', 'pot', 'wc', 'htn', 'dm', 'cad', 'pe', 'ane']
HEART_FEATURES = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach','exang','oldpeak','slope','ca','thal']
HYPERTENSION_FEATURES = ['age', 'bmi', 'smoking', 'exercise', 'alcohol']
BREAST_FEATURES = ['mean_radius', 'mean_texture', 'mean_perimeter', 'mean_area', 'mean_smoothness', 'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean']
LUNG_FEATURES = ['GENDER', 'AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE', 'CHRONIC_DISEASE', 'FATIGUE', 'ALLERGY', 'WHEEZING', 'ALCOHOL_CONSUMING', 'COUGHING', 'SHORTNESS_OF_BREATH', 'SWALLOWING_DIFFICULTY', 'CHEST_PAIN']
LIVER_FEATURES = ['Age', 'Gender', 'Total_Bilirubin', 'Alkaline_Phosphotase', 'Alamine_Aminotransferace', 'Aspartate_Amino', 'Protien', 'Albumin', 'Albumin_Globulin_ratio']


FEATURE_INPUTS = {
    # ------------------- DIABETES -------------------
    'Pregnancies': ('slider', 0, 20, 1),
    'Glucose': ('slider', 40, 300, 1),  # 40 is hypoglycemia, 300 is very high
    'BloodPressure': ('slider', 30, 180, 1),  # realistic: systolic can't be <30
    'SkinThickness': ('slider', 0, 100, 1),
    'Insulin': ('slider', 0, 900, 1),  # fasting insulin usually 16-166 µU/mL
    'BMI': ('slider', 10.0, 70.0, 0.1),  # BMI rarely below 10
    'DiabetesPedigreeFunction': ('slider', 0.01, 2.5, 0.01),  # avoid 0
    'Age': ('slider', 1, 120, 1),

    # ------------------- KIDNEY -------------------
    'age': ('slider', 1, 100, 1),
    'bp': ('slider', 40, 180, 1),  # systolic can't be <40
    'al': ('slider', 0, 5, 1),
    'su': ('slider', 0, 5, 1),
    'rbc': ('select', ['normal', 'abnormal']),
    'pc': ('select', ['normal', 'abnormal']),
    'pcc': ('select', ['present', 'notpresent']),
    'ba': ('select', ['present', 'notpresent']),
    'bgr': ('slider', 70, 500, 1),  # realistic fasting sugar min
    'bu': ('slider', 5, 200, 1),
    'sc': ('slider', 0.1, 20.0, 0.1),
    'pot': ('slider', 2.5, 10.0, 0.1),  # normal K+ ~3.5-5.5 mmol/L
    'wc': ('slider', 1000, 25000, 100),
    'htn': ('select', ['yes', 'no']),
    'dm': ('select', ['yes', 'no']),
    'cad': ('select', ['yes', 'no']),
    'pe': ('select', ['yes', 'no']),
    'ane': ('select', ['yes', 'no']),

    # ------------------- HEART -------------------
    'age': ('slider', 1, 100, 1),
    'sex': ('select', ['male', 'female']),
    'cp': ('slider', 0, 3, 1),
    'trestbps': ('slider', 80, 200, 1),  # normal range 90-120
    'chol': ('slider', 100, 600, 1),     # total cholesterol realistic range
    'fbs': ('select', ['yes', 'no']),
    'restecg': ('slider', 0, 2, 1),
    'thalach': ('slider', 60, 220, 1),   # max HR during stress
    'exang': ('select', ['yes', 'no']),
    'oldpeak': ('slider', 0.0, 6.0, 0.1),
    'slope': ('slider', 0, 2, 1),
    'ca': ('slider', 0, 4, 1),
    'thal': ('slider', 0, 3, 1),

    # ------------------- HYPERTENSION -------------------
    'bmi': ('slider', 10.0, 60.0, 0.1),
    'smoking': ('select', ['yes', 'no']),
    'exercise': ('select', ['yes', 'no']),
    'alcohol': ('select', ['yes', 'no']),
    'age': ('slider', 1, 100, 1),

    # ------------------- BREAST CANCER -------------------
    'mean_radius': ('slider', 5.0, 30.0, 0.1),
    'mean_texture': ('slider', 5.0, 40.0, 0.1),
    'mean_perimeter': ('slider', 30.0, 200.0, 0.1),
    'mean_area': ('slider', 100.0, 2500.0, 1.0),
    'mean_smoothness': ('slider', 0.05, 0.2, 0.001),
    'compactness_mean': ('slider', 0.01, 1.0, 0.01),
    'concavity_mean': ('slider', 0.01, 1.0, 0.01),
    'concave points_mean': ('slider', 0.01, 0.5, 0.01),
    'symmetry_mean': ('slider', 0.1, 0.5, 0.01),
    'fractal_dimension_mean': ('slider', 0.01, 0.2, 0.001),

    # ------------------- LUNG CANCER -------------------
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

    # ------------------- LIVER -------------------
    'Age': ('slider', 1, 100, 1),
    'Gender': ('select', ['Male', 'Female']),
    'Total_Bilirubin': ('slider', 0.1, 10.0, 0.1),
    'Alkaline_Phosphotase': ('slider', 50, 3000, 1),
    'Alamine_Aminotransferace': ('slider', 1, 2000, 1),
    'Aspartate_Amino': ('slider', 1, 2000, 1),
    'Protien': ('slider', 2.0, 10.0, 0.1),
    'Albumin': ('slider', 1.0, 6.0, 0.1),
    'Albumin_Globulin_ratio': ('slider', 0.1, 3.0, 0.1)
}




feature_fullforms = {
        'Choose Disease': {
        'Select the disease you want to diagnose': 'Please choose one disease from the list'
    },

    'Diabetes': {
        'Pregnancies': 'Number of times pregnant',
        'Glucose': 'Plasma glucose concentration',
        'BloodPressure': 'Diastolic blood pressure (mm Hg)',
        'SkinThickness': 'Triceps skin fold thickness (mm)',
        'Insulin': '2-Hour serum insulin (mu U/ml)',
        'BMI': 'Body Mass Index',
        'DiabetesPedigreeFunction': 'Diabetes pedigree function',
        'Age': 'Age in years'
    },
    'Kidney Disease': {
        'age': 'Age',
        'bp': 'Blood Pressure',
        'al': 'Albumin',
        'su': 'Sugar',
        'rbc': 'Red Blood Cells',
        'pc': 'Pus Cell',
        'pcc': 'Pus Cell Clumps',
        'ba': 'Bacteria',
        'bgr': 'Blood Glucose Random',
        'bu': 'Blood Urea',
        'sc': 'Serum Creatinine',
        'pot': 'Potassium',
        'wc': 'White Blood Cell Count',
        'htn': 'Hypertension',
        'dm': 'Diabetes Mellitus',
        'cad': 'Coronary Artery Disease',
        'pe': 'Pedal Edema',
        'ane': 'Anemia'
    },
    'Heart Disease': {
        'age': 'Age',
        'sex': 'Sex',
        'cp': 'Chest Pain Type',
        'trestbps': 'Resting Blood Pressure',
        'chol': 'Serum Cholesterol',
        'fbs': 'Fasting Blood Sugar > 120 mg/dl',
        'restecg': 'Resting ECG Results',
        'thalach': 'Maximum Heart Rate Achieved',
        'exang': 'Exercise Induced Angina',
        'oldpeak': 'ST Depression',
        'slope': 'Slope of ST Segment',
        'ca': 'Number of Major Vessels Colored',
        'thal': 'Thalassemia'
    },
    'Hypertension': {
        'bmi': 'Body Mass Index',
        'smoking': 'Smoking Habit',
        'exercise': 'Physical Activity',
        'alcohol': 'Alcohol Consumption',
        'age': 'Age'
    },
    'Breast Cancer': {
        'mean_radius': 'Mean Radius',
        'mean_texture': 'Mean Texture',
        'mean_perimeter': 'Mean Perimeter',
        'mean_area': 'Mean Area',
        'mean_smoothness': 'Mean Smoothness',
        'compactness_mean': 'Mean Compactness',
        'concavity_mean': 'Mean Concavity',
        'concave points_mean': 'Mean Concave Points',
        'symmetry_mean': 'Mean Symmetry',
        'fractal_dimension_mean': 'Mean Fractal Dimension'
    },
    'Lung Cancer': {
        'GENDER': 'Gender',
        'AGE': 'Age',
        'SMOKING': 'Smoking',
        'YELLOW_FINGERS': 'Yellow Fingers',
        'ANXIETY': 'Anxiety',
        'PEER_PRESSURE': 'Peer Pressure',
        'CHRONIC_DISEASE': 'Chronic Disease',
        'FATIGUE': 'Fatigue',
        'ALLERGY': 'Allergy',
        'WHEEZING': 'Wheezing',
        'ALCOHOL_CONSUMING': 'Alcohol Consumption',
        'COUGHING': 'Coughing',
        'SHORTNESS_OF_BREATH': 'Shortness of Breath',
        'SWALLOWING_DIFFICULTY': 'Swallowing Difficulty',
        'CHEST_PAIN': 'Chest Pain'
    },
    'Liver Disease': {
        'Age': 'Age',
        'Gender': 'Gender',
        'Total_Bilirubin': 'Total Bilirubin',
        'Alkaline_Phosphotase': 'Alkaline Phosphotase',
        'Alamine_Aminotransferace': 'Alamine Aminotransferase',
        'Aspartate_Amino': 'Aspartate Aminotransferase',
        'Protien': 'Total Protein',
        'Albumin': 'Albumin Level',
        'Albumin_Globulin_ratio': 'Albumin to Globulin Ratio'
    }
}




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

        # data = np.array([input_values], dtype=float)
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




def show_medical_report():
    import io
    from fpdf import FPDF
    import streamlit as st

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

    # Generate PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt="Medical Diagnosis Report", ln=True, align='C')
    pdf.cell(200, 10, txt="Diagnosed by IntelliMed AI System", ln=True, align='C')
    pdf.ln(10)

    pdf.cell(200, 10, txt=f"Disease: {result['title']}", ln=True)
    pdf.cell(200, 10, txt=f"Diagnosis Result: {result['diagnosis']}", ln=True)
    pdf.cell(200, 10, txt=f"Confidence Score: {result['confidence']}", ln=True)
    pdf.ln(5)

    pdf.cell(200, 10, txt="Patient Provided Details:", ln=True)
    for k, v in result['inputs'].items():
        pdf.cell(200, 10, txt=f"{k}: {v}", ln=True)

    pdf_bytes = pdf.output(dest='S').encode('latin-1')
    pdf_buffer = io.BytesIO(pdf_bytes)
    pdf_buffer.seek(0)  # Important!

    # Download button
    st.download_button(
        label="📥 Download Report",
        data=pdf_buffer,
        file_name="medical_report.pdf",
        mime="application/pdf"
    )





# ---------------------- CSS STYLES ---------------------- #
# Custom CSS
st.markdown("""
    <style>
    #MainMenu, footer, header {visibility: hidden;}
    [data-testid="stSidebar"] {
        background: linear-gradient(to bottom, #0f172a, #1e293b);
        color: white;
        padding: 1.5rem 1rem;
    }
    .sidebar-title {
        text-align: center;
        font-size: 20px;
        font-weight: 600;
        color: #10b981;
        margin-bottom: 25px;
    }
    .topnav {
        position: fixed;
        top: 10px;
        width: 50%;
        background-color: #fff;
        height: 50px;
        z-index: 1000;
        display: flex;
        align-items: center;
        
    }
    .topnav h1 {
        color: #10b981;
        font-size: 24px;
        margin: 0;
        padding-top: 5px;
    }
    .topnav a {
        float: right;
        color: white;
        background-color: #1e293b;
        padding: 6px 14px;
        margin-left: 10px;
        text-decoration: none;
        font-size: 16px;
        border-radius: 6px;
        transition: 0.3s;
    }
    .topnav a:hover {
        background-color: #10b981;
        color: black;
    }
    .topnav a.active {
        background-color: #10b981;
        color: white;
    }
            
    #     .chat-box {
    #     height: 450px;
    #     top: 0;
    #     overflow-y: auto;
    #     /* border: 2px solid #d3d3d3; */
    #     border-radius: 8px;
    #     padding: 15px;
    #     background-color: #fff;
    #     box-shadow: 0 4px 4px 4px rgba(0, 0, 0, 0.3);
    #     margin-bottom: 10px;
    #     display: flex;
    #     flex-direction: column;
    #     animation: fadeIn 0.5s ease-in-out;
    #     max-width: 100%;
    # }
            
    div[data-testid="stChatInput"] {
        padding: 12px;
        border-radius: 25px;
        background-color: #ffffff;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        /* transition: all 0.3s ease; */
        width: 100%;
        bottom: 0;

    }

    input[type="radio"]:checked + div > div {
        color: #10b981 !important;
        font-weight: bold;
        font-size: 20px !important;
    }
    input[type="radio"]:not(:checked) + div > div {
        color: white !important;
        font-weight: bold;
        font-size: 18px !important;
    }
    @media screen and (max-width: 600px) {
        .topnav {
            padding: 10px;
            width: 100%;
            visibility: visible;
        }
        .topnav a {
            padding: 8px 12px;
            font-size: 10px;
            float: none;
        }
    }
    </style>
""", unsafe_allow_html=True)




# ---------------------- CONFIG / HELPERS ---------------------- #
query_params = st.query_params
page = query_params.get("page", "home")

def get_bot_response(message):
    response = model.generate_content(message)
    return response.text




MODEL_MAPPING = {
    "Diabetes": ("Diabetes", DIABETES_FEATURES, diabetes_model, diabetes_scaler),
    "Kidney Disease": ("Kidney Disease", KIDNEY_FEATURES, kidney_model, None),
    "Heart Disease": ("Heart Disease", HEART_FEATURES, heart_model, heart_scaler),
    "Hypertension": ("Hypertension", HYPERTENSION_FEATURES, hypertension_model, None),
    "Breast Cancer": ("Breast Cancer", BREAST_FEATURES, breast_model, None),
    "Lung Cancer": ("Lung Cancer", LUNG_FEATURES, lung_model, None),
    "Liver Disease": ("Liver Disease", LIVER_FEATURES, liver_model, None),
}

# ---------------------- TOP NAVBAR ---------------------- #
st.markdown(f"""
<div class="topnav">
    <h1>🩺 IntelliMed</h1>
    <a href="?page=chatbot">Chatbot</a>
    <a href="?page=diagnose">Diagnose</a>
    <a href="?page=home">Home</a>
</div>
""", unsafe_allow_html=True)

# ---------------------- ABOUT PAGE ---------------------- #
if page == "home":
    st.title("👨‍⚕️ About This Project")
    st.markdown("""
    This is a **Smart AI Medical Diagnosis App** developed using **Streamlit** and **Machine Learning** models.
    - 🧬 Diabetes
    - 🧠 Brain & Heart Diseases
    - 🫁 Lung Cancer
    - 🏥 Kidney & Liver Disorders
    - 🧪 Breast Cancer

    **Built with 💚 by Subhash Kumar**
    ---
    ### Technologies Used:
    - Python
    - Streamlit
    - Scikit-learn
    - Joblib
    - CSS styling
    """)
    st.sidebar.image("./static/logo.png", width=200)
    st.sidebar.markdown("Email:")
    st.sidebar.markdown("Phone:")
    st.sidebar.markdown("LinkedIn:")
    st.sidebar.markdown("GitHub:")
    st.sidebar.info("This is a demo version. For full features, please contact the developer.")
    st.balloons()

# ---------------------- HOME PAGE ---------------------- #
elif page == "chatbot":
        st.sidebar.image("./static/logo.png", width=200)
        st.sidebar.title("📋 Navigation")
        

        if "chat_history" not in st.session_state:
          st.session_state.chat_history = load_chat_history()
    
        # chatmodel = st.sidebar.radio("", ["🤖 ChatBot", "🩺 Disease Diagnose", "🩺 Services"])

    # if rd == "🤖 ChatBot":
        # st.sidebar.selectbox("Choose AI", ["ChatBot (Gemini)", "Voice Assistant"])

        st.markdown("""
        <style>
        .message {
            padding: 8px 12px;
            border-radius: 8px;
            max-width: 80%;
            word-wrap: break-word;
            margin-bottom: 10px;
        }
        .user {
            # background-color: gray;
            color: black;
            align-self: flex-end;
            text-align: right;
            margin-left: auto;
        }
        .bot {
            # background-color: #e2e3e5;
            color: black;
            align-self: flex-start;
            text-align: left;
        }
        </style>
        """, unsafe_allow_html=True)

        st.markdown("<h2 style='text-align: center; color: #10b981;'>💬 IntelliMed AI Chat</h2>", unsafe_allow_html=True)

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        with st.container():
            chat_html = "<div class='chat-box'>"

            for role, msg in st.session_state.chat_history:
                tag = "user" if role == "user" else "bot"
                icon = "🧑‍💻" if role == "user" else "🤖"
                chat_html += f"<div class='message {tag}'>{icon} {msg}</div>"
            chat_html += "</div>"
            st.markdown(chat_html, unsafe_allow_html=True)

        user_input = st.chat_input("Type your message...", key="user_input")
        if user_input:
            st.session_state.chat_history.append(("user", user_input))
            with st.spinner("Thinking..."):
                bot_reply = get_bot_response(user_input)
            st.session_state.chat_history.append(("bot", bot_reply))
            st.rerun()

if page == "diagnose":


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


    # Show feature full forms in sidebar based on selected model
    st.sidebar.markdown("### 🔍 Features Info")
    for feature, fullform in feature_fullforms.get(model_choice, {}).items():
        st.sidebar.markdown(f"**{feature}**: {fullform}")




