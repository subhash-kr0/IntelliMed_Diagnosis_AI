import streamlit as st
import joblib
import google.generativeai as genai
import numpy as np
import pandas as pd
import json
import os
import io
from fpdf import FPDF # Moved FPDF import to top for clarity

# --- Global Configuration & Constants ---
NAV_HEIGHT_PX = 60 # Height of the top navigation bar in pixels

# Page config
st.set_page_config(layout='wide', page_icon='🎈', initial_sidebar_state='expanded')

# API keys
gemini_key = st.secrets["api_keys"]["gemini"]

# Configure Gemini
genai.configure(api_key=gemini_key)

# --- Model and Resource Loading with Caching ---
@st.cache_resource
def load_gemini_model():
    return genai.GenerativeModel(model_name="models/gemini-2.0-flash-lite")

model = load_gemini_model()

@st.cache_resource
def load_ml_model(path):
    return joblib.load(path)

diabetes_model = load_ml_model('./models/diabetes_model.pkl')
diabetes_scaler = load_ml_model('./models/diabetes_scaler.pkl')
kidney_model = load_ml_model('./models/kidneyDisease_model.pkl')
heart_model = load_ml_model('models/heartDisease_randomForest_model.pkl')
heart_scaler = load_ml_model('models/heartDisease_scaler.pkl')
hypertension_model = load_ml_model('models/hypertension_model.pkl')
breast_model = load_ml_model('./models/breastCancer_randomForest_model.pkl')
lung_model = load_ml_model('./models/lungCancer_XGBClassifier_model.pkl')
liver_model = load_ml_model('./models/liverDisease_rf_model.pkl')
thyroid_model = load_ml_model('./models/thyroid_cat_model.pkl')

# Feature lists (ensure these are exactly as expected by your models)
DIABETES_FEATURES = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
KIDNEY_FEATURES = ['age', 'bp', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 'bu', 'sc', 'pot', 'wc', 'htn', 'dm', 'cad', 'pe', 'ane']
HEART_FEATURES = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach','exang','oldpeak','slope','ca','thal']
HYPERTENSION_FEATURES = ['Age', 'BMI', 'Systolic_BP','Diastolic_BP', 'Total_Cholesterol']
BREAST_FEATURES = ['mean_radius', 'mean_texture', 'mean_perimeter', 'mean_area', 'mean_smoothness', 'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean']
LUNG_FEATURES = ['GENDER', 'AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE', 'CHRONIC_DISEASE', 'FATIGUE', 'ALLERGY', 'WHEEZING', 'ALCOHOL_CONSUMING', 'COUGHING', 'SHORTNESS_OF_BREATH', 'SWALLOWING_DIFFICULTY', 'CHEST_PAIN']
LIVER_FEATURES = ['Age', 'Gender', 'Total_Bilirubin', 'Alkaline_phosphate', 'Alamine_Aminotransferace', 'Aspartate_Amino', 'Protien', 'Albumin', 'Albumin_Globulin_ratio']
THYROID_FEATURES = ['Age','Sex','On Thyroxine','Query on Thyroxine','On Antithyroid Meds','Is Sick','Is Pregnant','Had Thyroid Surgery','Had I131 Treatment','Query Hypothyroid','Query Hyperthyroid','On Lithium','Has Goitre','Has Tumor','Psych Condition','TSH Level','T3 Level','TT4 Level','T4U Level','FTI Level', 'TBG Level']

FEATURE_INPUTS = {
    'Pregnancies': ('slider', 0, 20, 1), 'Glucose': ('slider', 40, 300, 1), 'BloodPressure': ('slider', 30, 180, 1),
    'SkinThickness': ('slider', 0, 100, 1), 'Insulin': ('slider', 0, 900, 1), 'BMI': ('slider', 10.0, 70.0, 0.1),
    'DiabetesPedigreeFunction': ('slider', 0.01, 2.5, 0.01), 'Age': ('slider', 1, 120, 1), # General 'Age', specific lists manage context
    'age': ('slider', 1, 100, 1), 'bp': ('slider', 40, 180, 1), 'al': ('slider', 0, 5, 1), 'su': ('slider', 0, 5, 1),
    'rbc': ('select', ['normal', 'abnormal']), 'pc': ('select', ['normal', 'abnormal']),
    'pcc': ('select', ['present', 'notpresent']), 'ba': ('select', ['present', 'notpresent']),
    'bgr': ('slider', 70, 500, 1), 'bu': ('slider', 5, 200, 1), 'sc': ('slider', 0.1, 20.0, 0.1),
    'pot': ('slider', 2.5, 10.0, 0.1), 'wc': ('slider', 1000, 25000, 100),
    'htn': ('select', ['yes', 'no']), 'dm': ('select', ['yes', 'no']), 'cad': ('select', ['yes', 'no']),
    'pe': ('select', ['yes', 'no']), 'ane': ('select', ['yes', 'no']), 'sex': ('select', ['male', 'female']),
    'cp': ('slider', 0, 3, 1), 'trestbps': ('slider', 80, 200, 1), 'chol': ('slider', 100, 600, 1),
    'fbs': ('select', ['yes', 'no']), 'restecg': ('slider', 0, 2, 1), 'thalach': ('slider', 60, 220, 1),
    'exang': ('select', ['yes', 'no']), 'oldpeak': ('slider', 0.0, 6.0, 0.1), 'slope': ('slider', 0, 2, 1),
    'ca': ('slider', 0, 4, 1), 'thal': ('slider', 0, 3, 1),
    'Systolic_BP': ('slider', 80, 200, 1), 'Diastolic_BP': ('slider', 40, 120, 1),
    'Total_Cholesterol': ('slider', 100, 400, 1), 'mean_radius': ('slider', 5.0, 30.0, 0.1),
    'mean_texture': ('slider', 5.0, 40.0, 0.1), 'mean_perimeter': ('slider', 30.0, 200.0, 0.1),
    'mean_area': ('slider', 100.0, 2500.0, 1.0), 'mean_smoothness': ('slider', 0.05, 0.2, 0.001),
    'compactness_mean': ('slider', 0.01, 1.0, 0.01), 'concavity_mean': ('slider', 0.01, 1.0, 0.01),
    'concave points_mean': ('slider', 0.01, 0.5, 0.01), 'symmetry_mean': ('slider', 0.1, 0.5, 0.01),
    'fractal_dimension_mean': ('slider', 0.01, 0.2, 0.001), 'GENDER': ('select', ['Male', 'Female']),
    'AGE': ('slider', 10, 100, 1), 'SMOKING': ('select', ['Yes', 'No']),
    'YELLOW_FINGERS': ('select', ['Yes', 'No']), 'ANXIETY': ('select', ['Yes', 'No']),
    'PEER_PRESSURE': ('select', ['Yes', 'No']), 'CHRONIC_DISEASE': ('select', ['Yes', 'No']),
    'FATIGUE': ('select', ['Yes', 'No']), 'ALLERGY': ('select', ['Yes', 'No']),
    'WHEEZING': ('select', ['Yes', 'No']), 'ALCOHOL_CONSUMING': ('select', ['Yes', 'No']),
    'COUGHING': ('select', ['Yes', 'No']), 'SHORTNESS_OF_BREATH': ('select', ['Yes', 'No']),
    'SWALLOWING_DIFFICULTY': ('select', ['Yes', 'No']), 'CHEST_PAIN': ('select', ['Yes', 'No']),
    'Gender': ('select', ['Male', 'Female']), 'Total_Bilirubin': ('slider', 0.1, 10.0, 0.1),
    'Alkaline_phosphate': ('slider', 50, 3000, 1), 'Alamine_Aminotransferace': ('slider', 1, 2000, 1),
    'Aspartate_Amino': ('slider', 1, 2000, 1), 'Protien': ('slider', 2.0, 10.0, 0.1),
    'Albumin': ('slider', 1.0, 6.0, 0.1), 'Albumin_Globulin_ratio': ('slider', 0.1, 3.0, 0.1),
    'Sex': ('select', ['Female', 'Male']), 'On Thyroxine': ('select', ['No', 'Yes']),
    'Query on Thyroxine': ('select', ['No', 'Yes']), 'On Antithyroid Meds': ('select', ['No', 'Yes']),
    'Is Sick': ('select', ['No', 'Yes']), 'Is Pregnant': ('select', ['No', 'Yes']),
    'Had Thyroid Surgery': ('select', ['No', 'Yes']), 'Had I131 Treatment': ('select', ['No', 'Yes']),
    'Query Hypothyroid': ('select', ['No', 'Yes']), 'Query Hyperthyroid': ('select', ['No', 'Yes']),
    'On Lithium': ('select', ['No', 'Yes']), 'Has Goitre': ('select', ['No', 'Yes']),
    'Has Tumor': ('select', ['No', 'Yes']), 'Psych Condition': ('select', ['No', 'Yes']),
    'TSH Level': ('slider', 0.01, 20.0, 0.1), 'T3 Level': ('slider', 0.2, 5.0, 0.1),
    'TT4 Level': ('slider', 50, 300, 1), 'T4U Level': ('slider', 0.3, 1.5, 0.01),
    'FTI Level': ('slider', 3, 50, 0.1), 'TBG Level': ('slider', 10, 50, 1),
}
# feature_fullforms can be extensive; assuming it's defined as in your previous version.
# I'll include a truncated version for brevity here, ensure you have the full one.
feature_fullforms = {
    'Diabetes': {'Pregnancies': 'Number of times pregnant', 'Glucose': 'Plasma glucose concentration', # ... and so on
                 'Age': 'Age (Diabetes)'}, # Example to show context
    'Kidney Disease': {'age': 'Age (Kidney)', 'bp': 'Blood Pressure'},
    'Heart Disease': {'age': 'Age (Heart)', 'sex': 'Sex'},
    'Hypertension': {'Age': 'Age (Hypertension)', 'BMI': 'Body Mass Index (Hypertension)'},
    # ... Add all other diseases and their features
}


@st.cache_data
def load_chat_history():
    # (Implementation from previous version)
    if os.path.exists("chat_history.json"):
        try:
            with open("chat_history.json", "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return [("bot", "Welcome! Previous history may be corrupted.")]
    return [("bot", "Welcome to IntelliMed AI Chat! How can I assist you today?")]

def save_chat_history(chat_history):
    # (Implementation from previous version)
    with open("chat_history.json", "w", encoding="utf-8") as f:
        json.dump(chat_history, f)

def prediction_form(disease_name, features, model_to_predict, scaler=None):
    # (Slightly refined implementation from previous version for clarity and robustness)
    # Using st.container for better grouping if needed, though form itself is a good group
    # st.subheader(f"🩺 {disease_name} Prediction Form") # Subheader is good
    
    with st.form(f"{disease_name.lower().replace(' ', '_')}_form"):
        st.markdown(f"### Patient Details for {disease_name} Prediction")
        
        # Dynamically create columns: 1, 2, or 3 based on number of features for better layout
        num_features = len(features)
        if num_features <= 4:
            num_cols = 1
        elif num_features <= 12:
            num_cols = 2
        else:
            num_cols = 3
        cols = st.columns(num_cols)
        
        inputs = {}
        for i, feature in enumerate(features):
            col_to_use = cols[i % num_cols]
            with col_to_use:
                f_key = feature.strip()
                display_label = feature_fullforms.get(disease_name, {}).get(f_key, f_key.replace("_", " ").capitalize())

                if f_key in FEATURE_INPUTS:
                    ftype, *params = FEATURE_INPUTS[f_key]
                    if ftype == 'slider':
                        min_val, max_val, step = params
                        default_val = float(min_val) if isinstance(min_val, (float, int)) else 0.0
                        inputs[f_key] = st.slider(display_label, float(min_val), float(max_val), default_val, step=float(step) if step else None)
                    elif ftype == 'select':
                        inputs[f_key] = st.selectbox(display_label, params[0], index=0)
                else:
                    inputs[f_key] = st.text_input(display_label) # Fallback

        submitted = st.form_submit_button("🔍 Predict Diagnosis")

    if submitted:
        # (Conversion and prediction logic from previous version)
        def convert(val):
            if isinstance(val, str):
                val_lower = val.lower()
                if val_lower in ['yes', 'present', 'male', 'normal', 'true']: return 1
                if val_lower in ['no', 'notpresent', 'female', 'abnormal', 'false']: return 0
            try: return float(val)
            except (ValueError, TypeError): return val

        input_values = [convert(inputs[f.strip()]) for f in features]
        
        try:
            data_df = pd.DataFrame([input_values], columns=[f.strip() for f in features])
            data_to_predict = data_df
            if scaler:
                data_scaled = scaler.transform(data_df)
                data_to_predict = pd.DataFrame(data_scaled, columns=data_df.columns)

            prediction = model_to_predict.predict(data_to_predict)[0]
            confidence = 0.0
            if hasattr(model_to_predict, 'predict_proba'):
                probs = model_to_predict.predict_proba(data_to_predict)[0]
                confidence = probs[np.argmax(probs)] # General confidence for the predicted class
                if prediction == 1 and len(probs) > 1 : confidence = probs[1] # If binary and positive, take positive prob

            diagnosis = "Positive" if prediction == 1 else "Negative"

            st.session_state['prediction_result'] = {
                'title': disease_name, 'inputs': inputs, 'diagnosis': diagnosis,
                'confidence': f"{confidence:.2%}" if confidence else "N/A"
            }
            
            # Result display using st.success or st.error based on diagnosis
            if diagnosis == "Positive":
                st.error(f"Diagnosis: {diagnosis} (Confidence: {st.session_state['prediction_result']['confidence']})")
            else:
                st.success(f"Diagnosis: {diagnosis} (Confidence: {st.session_state['prediction_result']['confidence']})")
            st.balloons()
            show_medical_report()

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            # st.error(f"Input data that caused error: {inputs}") # For debugging

def show_medical_report():
    # (Implementation from previous version, ensure FPDF is imported)
    result = st.session_state.get('prediction_result')
    if not result: return

    st.markdown("---")
    st.markdown("### 🏥 Medical Diagnosis Report")
    # ... (rest of PDF generation logic from previous, ensure it uses `result['inputs']` and `feature_fullforms` for display names)
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", 'B', 16)
    pdf.cell(0, 10, "Medical Diagnosis Report", ln=True, align='C')
    pdf.set_font("Helvetica", '', 10)
    pdf.cell(0, 7, "Diagnosed by IntelliMed AI System", ln=True, align='C')
    pdf.ln(10)

    pdf.set_font("Helvetica", 'B', 12)
    pdf.cell(0, 8, f"Disease Assessed: {result['title']}", ln=True)
    pdf.set_font("Helvetica", '', 11)
    pdf.cell(0, 8, f"Diagnosis Result: {result['diagnosis']}", ln=True)
    pdf.cell(0, 8, f"Confidence Score: {result['confidence']}", ln=True)
    pdf.ln(7)

    pdf.set_font("Helvetica", 'B', 12)
    pdf.cell(0, 8, "Patient Provided Details:", ln=True)
    pdf.set_font("Helvetica", '', 10)
    for k, v in result['inputs'].items():
        display_key_pdf = feature_fullforms.get(result['title'], {}).get(k, k.replace("_", " ").capitalize())
        pdf.multi_cell(0, 6, f"{display_key_pdf}: {v}")
    pdf.ln(5)

    pdf.set_font("Helvetica", 'I', 8)
    pdf.multi_cell(0, 5, "Disclaimer: This AI-generated report is for informational purposes only and is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.", align='C')
    
    pdf_bytes = pdf.output(dest='S').encode('latin-1')
    st.download_button(
        label="📥 Download Report (PDF)", data=io.BytesIO(pdf_bytes),
        file_name=f"{result['title'].replace(' ', '_')}_Report.pdf", mime="application/pdf"
    )

# ---------------------- CSS STYLES ---------------------- #
st.markdown(f"""
    <style>
        /* --- Global & Body --- */
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; /* Modern font stack */
            background-color: #f0f2f6; /* Light gray background for main area */
        }}
        /* Hide default Streamlit elements if absolutely necessary, but be cautious */
        #MainMenu, footer /*, header */ {{visibility: hidden;}} 

        /* --- Page Container for Top Nav Spacing --- */
        .page-container {{
            padding-top: {NAV_HEIGHT_PX + 10}px; /* Space for fixed topnav */
            padding-left: 1rem; /* Add some horizontal padding */
            padding-right: 1rem;
            padding-bottom: 2rem;
        }}

        /* --- Sidebar --- */
        [data-testid="stSidebar"] {{
            background: linear-gradient(to bottom, #0f172a, #1e293b); /* Dark blue/slate gradient */
            color: white;
            padding: 1.5rem 1rem;
            border-right: 1px solid #2c3e50; /* Subtle border */
        }}
        .sidebar-title {{
            text-align: center;
            font-size: 22px; /* Slightly larger */
            font-weight: 600;
            color: #10b981; /* Teal accent */
            margin-bottom: 20px;
        }}
        [data-testid="stSidebar"] a {{ /* Styling links in sidebar */
            color: #bdc3c7; /* Lighter gray for links */
            text-decoration: none;
            transition: color 0.3s ease;
        }}
        [data-testid="stSidebar"] a:hover {{
            color: #10b981; /* Teal on hover */
        }}
        [data-testid="stSidebar"] .stImage {{ /* Center logo if needed */
            margin-bottom: 1rem;
        }}


        /* --- Top Navigation Bar --- */
        .topnav {{
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: {NAV_HEIGHT_PX}px;
            background-color: #0f172a; /* Match sidebar base */
            color: white;
            display: flex;
            align-items: center;
            padding: 0 25px; /* More padding */
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
            z-index: 1001; /* Ensure it's above other elements */
        }}
        .topnav h1 {{
            color: #10b981; /* Teal accent */
            font-size: 26px; /* Larger title */
            font-weight: bold;
            margin: 0;
            margin-right: auto; /* Pushes links to the right */
        }}
        .topnav a {{
            color: #ecf0f1; /* Light text for links */
            padding: 10px 18px;
            margin-left: 8px;
            text-decoration: none;
            font-size: 16px;
            font-weight: 500;
            border-radius: 5px;
            transition: background-color 0.3s ease, color 0.3s ease;
        }}
        .topnav a:hover {{
            background-color: #10b981;
            color: white;
        }}
        .topnav a.active {{
            background-color: #10b981;
            color: white;
            font-weight: bold;
        }}

        /* --- Chat Interface --- */
        .chat-box {{
            height: 500px; /* Increased height */
            overflow-y: auto;
            border-radius: 8px;
            padding: 20px; /* More padding */
            background-color: #ffffff; /* White background for chat area */
            box-shadow: 0 2px 10px rgba(0,0,0,0.08);
            margin-bottom: 15px;
            display: flex;
            flex-direction: column;
        }}
        .message_container {{ display: flex; margin-bottom: 12px; }}
        .user_container {{ justify-content: flex-end; }}
        .bot_container {{ justify-content: flex-start; }}
        .message {{
            padding: 12px 18px;
            border-radius: 20px; /* More rounded */
            max-width: 70%;
            word-wrap: break-word;
            line-height: 1.5;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .user {{
            background: linear-gradient(135deg, #007bff, #0056b3); /* Brighter blue gradient */
            color: white;
            border-bottom-right-radius: 5px; /* Differentiated shape */
        }}
        .bot {{
            background: linear-gradient(135deg, #28a745, #1e7e34); /* Green gradient */
            color: white;
            border-bottom-left-radius: 5px; /* Differentiated shape */
        }}
        .message_container .message {{
            display: flex; /* For icon and text alignment */
            align-items: center;
        }}
        .message_container .message span {{ /* For icon */
            margin-right: 8px;
            font-size: 1.2em;
        }}

        /* --- General UI Elements --- */
        h1, h2, h3 {{ color: #2c3e50; }} /* Darker primary text color */
        .stButton>button {{ /* Style Streamlit buttons */
            background-color: #10b981;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            font-weight: bold;
            transition: background-color 0.3s ease;
        }}
        .stButton>button:hover {{
            background-color: #0d8a63; /* Darker shade on hover */
            color: white;
        }}
        .stTextInput input, .stSelectbox div[data-baseweb="select"] > div {{
             border-radius: 5px !important;
             border: 1px solid #ced4da !important;
        }}
        .stTextInput input:focus, .stSelectbox div[data-baseweb="select"] > div[aria-expanded="true"] {{
             border-color: #10b981 !important;
             box-shadow: 0 0 0 0.2rem rgba(16, 185, 129, 0.25) !important;
        }}

        /* Disease Card Styling on Home Page */
        .disease-card {{
            background-color: #ffffff;
            padding: 1.5rem;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            text-align: center;
            transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out;
            margin-bottom: 1rem; /* Ensure spacing between cards in a column */
            height: 100%; /* Make cards in a row equal height */
        }}
        .disease-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 6px 16px rgba(0,0,0,0.12);
        }}
        .disease-card-emoji {{
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
        }}
        .disease-card h5 {{
            color: #0f172a; /* Dark blue heading */
            margin-bottom: 0.3rem;
            font-weight: 600;
        }}
        .disease-card p {{
            font-size: 0.9rem;
            color: #555;
        }}

        /* Responsive Adjustments */
        @media screen and (max-width: 768px) {{
            .topnav {{ padding: 0 15px; height: {NAV_HEIGHT_PX -10}px; }}
            .topnav h1 {{ font-size: 20px; }}
            .topnav a {{ padding: 8px 12px; font-size: 14px; margin-left: 5px; }}
            .page-container {{ padding-top: {NAV_HEIGHT_PX}px; }} /* Adjust for smaller nav */
            .chat-box {{ height: 400px; padding: 15px; }}
            .message {{ max-width: 85%; }}
            .disease-card {{ padding: 1rem; }}
            .disease-card-emoji {{ font-size: 2rem; }}
        }}
    </style>
""", unsafe_allow_html=True)


# ---------------------- CONFIG / HELPERS ---------------------- #
query_params = st.query_params
page_values = query_params.get("page")
if page_values and isinstance(page_values, list) and len(page_values) > 0:
    page = page_values[0]
else:
    page = "home" # Default to home

def get_bot_response(message_text):
    try:
        response = model.generate_content(message_text)
        return response.text
    except Exception as e:
        st.error(f"AI Error: {e}")
        return "Sorry, I couldn't process that. Please try again."

MODEL_MAPPING = {
    "Diabetes": ("Diabetes", DIABETES_FEATURES, diabetes_model, diabetes_scaler),
    "Kidney Disease": ("Kidney Disease", KIDNEY_FEATURES, kidney_model, None),
    "Heart Disease": ("Heart Disease", HEART_FEATURES, heart_model, heart_scaler),
    "Hypertension": ("Hypertension", HYPERTENSION_FEATURES, hypertension_model, None),
    "Breast Cancer": ("Breast Cancer", BREAST_FEATURES, breast_model, None),
    "Lung Cancer": ("Lung Cancer", LUNG_FEATURES, lung_model, None),
    "Liver Disease": ("Liver Disease", LIVER_FEATURES, liver_model, None),
    "Thyroid Disease": ("Thyroid Disease", THYROID_FEATURES, thyroid_model, None)
}



query_params = st.query_params
current_page = query_params.get("page","home")


# ---------------------- TOP NAVBAR ---------------------- #
home_active = "active" if page == "home" else ""
diagnose_active = "active" if page == "diagnose" else ""
chatbot_active = "active" if page == "chatbot" else ""

st.markdown(f"""
<div class="topnav">
    <h1>🩺 IntelliMed</h1>
    <a class="{home_active}" href="?page=home">Home</a>
    <a class="{diagnose_active}" href="?page=diagnose">Diagnose</a>
    <a class="{chatbot_active}" href="?page=chatbot">Chatbot</a>
</div>
""", unsafe_allow_html=True)

# ---------------------- SIDEBAR (Common for all pages) ---------------------- #
with st.sidebar:
    st.image("./static/logo.png", width=120, use_column_width='auto') # Centered if use_column_width='auto' and sidebar isn't too wide
    st.markdown("<div class='sidebar-title'>IntelliMed AI</div>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("#### Navigation")
    if st.button("🏠 Home", use_container_width=True, key="sidebar_home"): st.query_params["page"] = "home"; st.rerun()
    if st.button("🔬 Diagnose Disease", use_container_width=True, key="sidebar_diagnose"): st.query_params["page"] = "diagnose"; st.rerun()
    if st.button("💬 AI Chatbot", use_container_width=True, key="sidebar_chatbot"): st.query_params["page"] = "chatbot"; st.rerun()
    st.markdown("---")
    st.markdown("#### Contact Developer")
    st.markdown("📧 [subhashkumardev@outlook.com](mailto:subhashkumardev@outlook.com)")
    st.markdown("🔗 [LinkedIn Profile](https://www.linkedin.com/in/subhashkumar-dev/)") # Example
    st.markdown("💻 [GitHub Profile](https://github.com/Subhash捃)") # Example
    st.markdown("---")
    st.info("This app is for educational purposes and not a substitute for professional medical advice.")

# --- Page Content: Each page wrapped in a container for consistent padding ---
if page == "home":
    with st.container(): # This acts as the .page-container
        st.markdown(f"<div class='page-container'>", unsafe_allow_html=True)
        st.title("👨‍⚕️ Welcome to IntelliMed AI")
        st.subheader("Your Smart AI Medical Diagnosis Assistant")
        st.markdown("""
        IntelliMed AI leverages machine learning for preliminary insights into health conditions.
        Analyze data with trained models to predict disease likelihood.
        **Navigate to Diagnose** for prediction forms, or use our **Chatbot** for health inquiries.
        **Disclaimer:** Educational use only. Always consult a healthcare professional.
        """)
        st.markdown("---")
        st.markdown("### <span style='color: #10b981;'>🧬</span> Diseases Covered for AI Assessment", unsafe_allow_html=True)
        
        disease_names_list = list(MODEL_MAPPING.keys())
        emojis = ["🩸", "🩺", "❤️", "📈", "🎗️", "🫁", "🌿", "🦋"]
        
        num_disease_cols = 3
        disease_cols = st.columns(num_disease_cols)
        
        for i, disease_name_home in enumerate(disease_names_list):
            with disease_cols[i % num_disease_cols]:
                emoji_char = emojis[i % len(emojis)]
                st.markdown(f"""
                <div class="disease-card">
                    <div class="disease-card-emoji">{emoji_char}</div>
                    <h5>{disease_name_home}</h5>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True) # Spacer
        st.markdown("---")
        # ... (rest of home page content from previous)
        st.markdown("""
        ### Technologies Behind IntelliMed AI:
        - **Backend & ML:** Python, Scikit-learn, Joblib, Pandas, NumPy
        - **Frontend:** Streamlit
        - **AI Chatbot:** Google Generative AI (Gemini)
        - **Styling:** Custom HTML/CSS

        ---
        **Developed with 💚 by Subhash Kumar**
        """)

        st.markdown("</div>", unsafe_allow_html=True)


elif page == "chatbot":
    with st.container():
        st.markdown(f"<div class='page-container'>", unsafe_allow_html=True)
        st.markdown("<h2 style='text-align: center; color: #10b981;'>💬 IntelliMed AI Chat</h2>", unsafe_allow_html=True)

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = load_chat_history()
        
        with st.container(): # Chat display area
            chat_html = "<div class='chat-box'>"
            for role, msg in st.session_state.chat_history:
                tag = "user" if role == "user" else "bot"
                icon = "<span>🧑‍💻</span>" if role == "user" else "<span>🤖</span>"
                chat_html += f"<div class='message_container {tag}_container'><div class='message {tag}'>{icon} {msg}</div></div>"
            chat_html += "</div>"
            st.markdown(chat_html, unsafe_allow_html=True)

        user_input = st.chat_input("Type your message here...", key="user_input_main") # Ensure key is unique if multiple chat_inputs exist
        if user_input:
            st.session_state.chat_history.append(("user", user_input))
            with st.spinner("🤖 Thinking..."):
                bot_reply = get_bot_response(user_input)
            st.session_state.chat_history.append(("bot", bot_reply))
            save_chat_history(st.session_state.chat_history)
            st.rerun() # Rerun to update chat display
        st.markdown("</div>", unsafe_allow_html=True)


elif page == "diagnose":
    with st.container():
        st.markdown(f"<div class='page-container'>", unsafe_allow_html=True)
        st.title("🔬 AI Disease Diagnosis")
        st.markdown("Select a method below to get started with your health assessment.")

        mode_options = ["Form Based Diagnosis", "Image Based (Under Development)"]
        if 'Mode' not in st.session_state:
            st.session_state['Mode'] = mode_options[0] # Default selection

        st.session_state['Mode'] = st.radio(
            "Select Diagnosis Type:",
            mode_options,
            index=mode_options.index(st.session_state['Mode']), # Persist selection
            horizontal=True,
            key="diagnosis_mode_radio"
        )
        st.markdown("---")

        if st.session_state.get("Mode") == "Form Based Diagnosis":
            form_page_options = ["General Symptom Checker"] + list(MODEL_MAPPING.keys())
            model_choice = st.selectbox(
                "Select a specific disease for detailed form or use the General Symptom Checker:",
                form_page_options,
                index=0,
                key="disease_model_choice_selectbox"
            )

            if model_choice == "General Symptom Checker":
                # (Symptom checker logic from previous version, ensure it's well-formatted)
                st.markdown("#### General Symptom Checker")
                st.caption("Select your symptoms. This tool provides a preliminary suggestion, not a definitive diagnosis.")
                # ... (rest of symptom checker from before)
                diseases_data = {
                    'Common Cold': ['runny nose', 'sore throat', 'cough', 'sneezing', 'mild headache', 'fatigue', 'nasal congestion'],
                    # ... (add all diseases_data from your original snippet)
                }
                all_symptoms = sorted(list(set(symptom for symptoms_list in diseases_data.values() for symptom in symptoms_list)))
                selected_symptoms = []
                num_symptom_cols = 3 # Or 2 for wider checkboxes
                symptom_cols = st.columns(num_symptom_cols)
                for i, symptom in enumerate(all_symptoms):
                    with symptom_cols[i % num_symptom_cols]:
                        if st.checkbox(symptom.capitalize(), key=f"symptom_{symptom.replace(' ', '_')}"):
                            selected_symptoms.append(symptom)
                if st.button("Suggest Possible Conditions", key="suggest_conditions_button"):
                    # ... (diagnosis logic for symptom checker)
                    if not selected_symptoms: st.warning("Please select symptoms.")
                    else: 
                        st.info("Symptom checker results would appear here.") # Placeholder for brevity

            elif model_choice in MODEL_MAPPING:
                disease_display_name, features_list, model_instance, scaler_instance = MODEL_MAPPING[model_choice]
                with st.container(border=True): # Card-like container for the form
                    st.markdown(f"### Predict <span style='color:#10b981;'>{disease_display_name}</span>", unsafe_allow_html=True)
                    prediction_form(disease_display_name, features_list, model_instance, scaler_instance)
        
        elif st.session_state.get("Mode") == "Image Based (Under Development)":
            st.info("🖼️ Image-based diagnosis is currently under development. Please check back later!")
            # Optionally show a placeholder image
            # st.image("path/to/your/coming_soon_image.png", width=300)

        st.markdown("</div>", unsafe_allow_html=True)

else: # Fallback for unknown page
    with st.container():
        st.markdown(f"<div class='page-container'>", unsafe_allow_html=True)
        st.error("🚫 Page Not Found")
        st.markdown("The page you are looking for does not exist or may have been moved.")
        st.page_link("?page=home", label="Go to Home Page", icon="🏠")
        st.markdown("</div>", unsafe_allow_html=True)