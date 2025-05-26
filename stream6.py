from fpdf import FPDF
import streamlit as st
import joblib
import google.generativeai as genai
import numpy as np
import pandas as pd
import json
import base64
from datetime import datetime
import os
from PIL import Image
# import markdown2 # Not used in the provided snippet
# from xhtml2pdf import pisa # Not used for FPDF generation
import io
import tensorflow as tf
import re

# --- Global Configuration & Constants ---
NAV_HEIGHT_PX = 60  # Height of the top navigation bar in pixels
APP_NAME = "🩺 IntelliMed AI"
APP_VERSION = "1.1" # Example version

# --- Page Configuration (should be the first Streamlit command) ---
st.set_page_config(
    layout='wide',
    page_icon='🎈', # Consider changing to a medical icon like '⚕️' or your logo
    page_title=APP_NAME,
    initial_sidebar_state='expanded'
)

# --- API Key and Gemini Configuration ---
try:
    gemini_key = st.secrets.get("api_keys", {}).get("gemini")
    if not gemini_key:
        st.error("Gemini API key not found in st.secrets. Please configure it for the chatbot to work.")
        GEMINI_CONFIGURED = False
    else:
        genai.configure(api_key=gemini_key)
        GEMINI_CONFIGURED = True
except Exception as e:
    st.error(f"Error configuring Gemini: {e}")
    GEMINI_CONFIGURED = False

# --- Model and Resource Loading with Caching ---
@st.cache_resource
def load_gemini_model():
    if GEMINI_CONFIGURED:
        return genai.GenerativeModel(model_name="gemini-1.5-flash-latest")
    return None
gemini_llm_model = load_gemini_model()

@st.cache_resource
def load_ml_model(path):
    try:
        return joblib.load(path)
    except FileNotFoundError:
        st.error(f"Model file not found: {path}. Please ensure it's in the correct location.")
        return None
    except Exception as e:
        st.error(f"Error loading model {path}: {e}")
        return None

# Load all your ML models
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

# Deep learning models
brainStrokemodel = None # Initialize
try:
    if os.path.exists("models/brainstroke_model.h5"):
        brainStrokemodel = tf.keras.models.load_model("models/brainstroke_model.h5")
    else:
        st.warning("Brain stroke model file (brainstroke_model.h5) not found in models directory.")
except Exception as e:
    st.error(f"Error loading Brain Stroke model: {e}")


# --- Feature Lists & Input Definitions (Ensure consistency with your model training) ---
DIABETES_FEATURES = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
KIDNEY_FEATURES = ['age', 'bp', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 'bu', 'sc', 'pot', 'wc', 'htn', 'dm', 'cad', 'pe', 'ane']
HEART_FEATURES = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach','exang','oldpeak','slope','ca','thal']
HYPERTENSION_FEATURES = ['Age', 'BMI', 'Systolic_BP','Diastolic_BP', 'Total_Cholesterol']
BREAST_FEATURES = ['mean_radius', 'mean_texture', 'mean_perimeter', 'mean_area', 'mean_smoothness', 'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean']
LUNG_FEATURES = ['GENDER', 'AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE', 'CHRONIC_DISEASE', 'FATIGUE', 'ALLERGY', 'WHEEZING', 'ALCOHOL_CONSUMING', 'COUGHING', 'SHORTNESS_OF_BREATH', 'SWALLOWING_DIFFICULTY', 'CHEST_PAIN']
LIVER_FEATURES = ['Age', 'Gender', 'Total_Bilirubin', 'Alkaline_phosphate', 'Alamine_Aminotransferace', 'Aspartate_Amino', 'Protien', 'Albumin', 'Albumin_Globulin_ratio']
THYROID_FEATURES = ['Age','Sex','On Thyroxine','Query on Thyroxine','On Antithyroid Meds','Is Sick','Is Pregnant','Had Thyroid Surgery','Had I131 Treatment','Query Hypothyroid','Query Hyperthyroid','On Lithium','Has Goitre','Has Tumor','Psych Condition','TSH Level','T3 Level','TT4 Level','T4U Level','FTI Level', 'TBG Level']
# TODO: Define features for Brain Stroke model
BRAIN_STROKE_FEATURES = [] # Example: ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi'] - REPLACE WITH ACTUAL

# FEATURE_INPUTS dictionary: Maps feature name to its input type and parameters.
# IMPORTANT: Standardize feature names (e.g., all 'age' to 'Age') or ensure this dict covers all variants used in _FEATURES lists.
FEATURE_INPUTS = {
    # Diabetes
    'Pregnancies': ('slider', 0, 20, 1), 'Glucose': ('slider', 40, 300, 100), 'BloodPressure': ('slider', 30, 180, 80),
    'SkinThickness': ('slider', 0, 100, 20), 'Insulin': ('slider', 0, 900, 150),
    'BMI': ('slider', 10.0, 70.0, 25.0, 0.1), 'DiabetesPedigreeFunction': ('slider', 0.01, 2.5, 0.5, 0.01),
    # Age variants - consider standardizing to one 'Age' key if possible
    'Age': ('slider', 1, 120, 30), # Used by Diabetes, Hypertension, Liver, Thyroid
    'age': ('slider', 1, 100, 30), # Used by Kidney, Heart
    'AGE': ('slider', 10, 100, 30),# Used by Lung

    # Kidney
    'bp': ('slider', 40, 180, 100), 'al': ('slider', 0, 5, 1), 'su': ('slider', 0, 5, 0),
    'rbc': ('select', ['normal', 'abnormal']), 'pc': ('select', ['normal', 'abnormal']),
    'pcc': ('select', ['present', 'notpresent']), 'ba': ('select', ['present', 'notpresent']),
    'bgr': ('slider', 70, 500, 120), 'bu': ('slider', 5, 200, 50), 'sc': ('slider', 0.1, 20.0, 1.0, 0.1),
    'pot': ('slider', 2.5, 10.0, 4.0, 0.1), 'wc': ('slider', 1000, 25000, 8000),
    'htn': ('select', ['yes', 'no']), 'dm': ('select', ['yes', 'no']), 'cad': ('select', ['yes', 'no']),
    'pe': ('select', ['yes', 'no']), 'ane': ('select', ['yes', 'no']),

    # Heart
    'sex': ('select', ['male', 'female']), # Heart (lowercase s)
    'Sex': ('select', ['Female', 'Male']), # Thyroid (uppercase S) - Note: 'Gender' might be more consistent for Liver/Lung
    'cp': ('slider', 0, 3, 0), 'trestbps': ('slider', 80, 200, 120), 'chol': ('slider', 100, 600, 200),
    'fbs': ('select', ['yes', 'no']), 'restecg': ('slider', 0, 2, 0), 'thalach': ('slider', 60, 220, 150),
    'exang': ('select', ['yes', 'no']), 'oldpeak': ('slider', 0.0, 6.0, 1.0, 0.1), 'slope': ('slider', 0, 2, 1),
    'ca': ('slider', 0, 4, 0), 'thal': ('slider', 0, 3, 2),

    # Hypertension (Age, BMI already defined)
    'Systolic_BP': ('slider', 80, 200, 120), 'Diastolic_BP': ('slider', 40, 120, 80),
    'Total_Cholesterol': ('slider', 100, 400, 200),

    # Breast Cancer
    'mean_radius': ('slider', 5.0, 30.0, 15.0, 0.1),'mean_texture': ('slider', 5.0, 40.0, 20.0, 0.1),
    'mean_perimeter': ('slider', 30.0, 200.0, 100.0, 0.1),'mean_area': ('slider', 100.0, 2500.0, 600.0, 1.0),
    'mean_smoothness': ('slider', 0.05, 0.2, 0.1, 0.001),'compactness_mean': ('slider', 0.01, 1.0, 0.1, 0.01),
    'concavity_mean': ('slider', 0.01, 1.0, 0.1, 0.01),'concave points_mean': ('slider', 0.01, 0.5, 0.05, 0.01),
    'symmetry_mean': ('slider', 0.1, 0.5, 0.2, 0.01),'fractal_dimension_mean': ('slider', 0.01, 0.2, 0.06, 0.001),

    # Lung Cancer (AGE already defined)
    'GENDER': ('select', ['Male', 'Female']), # Lung (UPPERCASE GENDER)
    'SMOKING': ('select', ['Yes', 'No']),
    'YELLOW_FINGERS': ('select', ['Yes', 'No']), 'ANXIETY': ('select', ['Yes', 'No']),
    'PEER_PRESSURE': ('select', ['Yes', 'No']), 'CHRONIC_DISEASE': ('select', ['Yes', 'No']),
    'FATIGUE': ('select', ['Yes', 'No']), 'ALLERGY': ('select', ['Yes', 'No']), # Renamed from 'ALLERGY '
    'WHEEZING': ('select', ['Yes', 'No']), 'ALCOHOL_CONSUMING': ('select', ['Yes', 'No']),
    'COUGHING': ('select', ['Yes', 'No']), 'SHORTNESS_OF_BREATH': ('select', ['Yes', 'No']),
    'SWALLOWING_DIFFICULTY': ('select', ['Yes', 'No']), 'CHEST_PAIN': ('select', ['Yes', 'No']),

    # Liver (Age already defined)
    'Gender': ('select', ['Male', 'Female']), # Liver (Capitalized Gender)
    'Total_Bilirubin': ('slider', 0.1, 10.0, 1.0, 0.1),
    'Alkaline_phosphate': ('slider', 50, 3000, 200), 'Alamine_Aminotransferace': ('slider', 1, 2000, 50),
    'Aspartate_Amino': ('slider', 1, 2000, 50), 'Protien': ('slider', 2.0, 10.0, 6.0, 0.1), # 'Protein' spelling?
    'Albumin': ('slider', 1.0, 6.0, 3.5, 0.1), 'Albumin_Globulin_ratio': ('slider', 0.1, 3.0, 1.0, 0.1),

    # Thyroid (Age, Sex already defined)
    'On Thyroxine': ('select', ['No', 'Yes']),'Query on Thyroxine': ('select', ['No', 'Yes']),
    'On Antithyroid Meds': ('select', ['No', 'Yes']),'Is Sick': ('select', ['No', 'Yes']),
    'Is Pregnant': ('select', ['No', 'Yes']),'Had Thyroid Surgery': ('select', ['No', 'Yes']),
    'Had I131 Treatment': ('select', ['No', 'Yes']),'Query Hypothyroid': ('select', ['No', 'Yes']),
    'Query Hyperthyroid': ('select', ['No', 'Yes']),'On Lithium': ('select', ['No', 'Yes']),
    'Has Goitre': ('select', ['No', 'Yes']),'Has Tumor': ('select', ['No', 'Yes']),
    'Psych Condition': ('select', ['No', 'Yes']),'TSH Level': ('slider', 0.01, 20.0, 2.0, 0.1),
    'T3 Level': ('slider', 0.2, 5.0, 1.5, 0.1),'TT4 Level': ('slider', 50, 300, 150),
    'T4U Level': ('slider', 0.3, 1.5, 0.9, 0.01),'FTI Level': ('slider', 3, 50, 25.0, 0.1),
    'TBG Level': ('slider', 10, 50, 25),

    # TODO: Add features for Brain Stroke if its model is used
    # Example (replace with actual features and types):
    # 'hypertension': ('select', ['No', 'Yes']), # Assuming 0 for No, 1 for Yes as per common practice
    # 'heart_disease': ('select', ['No', 'Yes']),
    # 'avg_glucose_level': ('slider', 50, 300, 100),
    # 'bmi': ('slider', 10.0, 70.0, 25.0, 0.1),
}

# Full forms for display names and PDF report.
# TODO: COMPLETE THIS FOR ALL FEATURES OF ALL DISEASES
feature_fullforms = {
    'Diabetes': {
        'Pregnancies': 'Pregnancies Count', 'Glucose': 'Glucose Level (mg/dL)',
        'BloodPressure': 'Blood Pressure (mm Hg)', 'SkinThickness': 'Skin Thickness (mm)',
        'Insulin': 'Insulin Level (mu U/ml)', 'BMI': 'Body Mass Index (kg/m²)',
        'DiabetesPedigreeFunction': 'Diabetes Pedigree Function', 'Age': 'Age (Years)'
    },
    'Kidney Disease': {
        'age': 'Age (Years)', 'bp': 'Blood Pressure (mm Hg)', 'al': 'Albumin (0-5)', 'su': 'Sugar (0-5)',
        'rbc': 'Red Blood Cells', 'pc': 'Pus Cells', 'pcc': 'Pus Cell Clumps', 'ba': 'Bacteria',
        'bgr': 'Blood Glucose Random (mgs/dL)', 'bu': 'Blood Urea (mgs/dL)', 'sc': 'Serum Creatinine (mgs/dL)',
        'pot': 'Potassium (mEq/L)', 'wc': 'White Blood Cell Count (cells/cmm)', 'htn': 'Hypertension',
        'dm': 'Diabetes Mellitus', 'cad': 'Coronary Artery Disease', 'pe': 'Pedal Edema', 'ane': 'Anemia'
    },
    'Heart Disease': {
        'age': 'Age (Years)', 'sex': 'Sex (0=female, 1=male)', 'cp': 'Chest Pain Type (0-3)',
        'trestbps': 'Resting Blood Pressure (mm Hg)', 'chol': 'Serum Cholesterol (mg/dl)',
        'fbs': 'Fasting Blood Sugar > 120 mg/dl', 'restecg': 'Resting Electrocardiographic Results (0-2)',
        'thalach': 'Maximum Heart Rate Achieved', 'exang': 'Exercise Induced Angina',
        'oldpeak': 'ST Depression Induced by Exercise', 'slope': 'Slope of Peak Exercise ST Segment',
        'ca': 'Number of Major Vessels Colored by Flourosopy (0-4)', 'thal': 'Thalassemia (0-3)'
    },
    'Hypertension': {
        'Age': 'Age (Years)', 'BMI': 'Body Mass Index (kg/m²)',
        'Systolic_BP': 'Systolic Blood Pressure (mm Hg)',
        'Diastolic_BP': 'Diastolic Blood Pressure (mm Hg)',
        'Total_Cholesterol': 'Total Cholesterol (mg/dL)'
    },
    'Breast Cancer': {
        'mean_radius': 'Mean Radius', 'mean_texture': 'Mean Texture', 'mean_perimeter': 'Mean Perimeter',
        'mean_area': 'Mean Area', 'mean_smoothness': 'Mean Smoothness',
        'compactness_mean': 'Mean Compactness', 'concavity_mean': 'Mean Concavity',
        'concave points_mean': 'Mean Concave Points', 'symmetry_mean': 'Mean Symmetry',
        'fractal_dimension_mean': 'Mean Fractal Dimension'
    },
    'Lung Cancer': {
        'GENDER': 'Gender (M/F encoded)', 'AGE': 'Age (Years)', 'SMOKING': 'Smoking',
        'YELLOW_FINGERS': 'Yellow Fingers', 'ANXIETY': 'Anxiety',
        'PEER_PRESSURE': 'Peer Pressure', 'CHRONIC_DISEASE': 'Chronic Disease',
        'FATIGUE': 'Fatigue', 'ALLERGY': 'Allergy', 'WHEEZING': 'Wheezing',
        'ALCOHOL_CONSUMING': 'Alcohol Consuming', 'COUGHING': 'Coughing',
        'SHORTNESS_OF_BREATH': 'Shortness of Breath',
        'SWALLOWING_DIFFICULTY': 'Swallowing Difficulty', 'CHEST_PAIN': 'Chest Pain'
    },
    'Liver Disease': {
        'Age': 'Age (Years)', 'Gender': 'Gender (M/F encoded)', 'Total_Bilirubin': 'Total Bilirubin (mg/dL)',
        'Alkaline_phosphate': 'Alkaline Phosphatase (IU/L)',
        'Alamine_Aminotransferace': 'Alamine Aminotransferase (SGPT) (IU/L)',
        'Aspartate_Amino': 'Aspartate Aminotransferase (SGOT) (IU/L)',
        'Protien': 'Total Protein (g/dL)', 'Albumin': 'Albumin (g/dL)',
        'Albumin_Globulin_ratio': 'Albumin and Globulin Ratio'
    },
    'Thyroid Disease':{
        'Age': 'Age (Years)', 'Sex': 'Sex (M/F encoded)', 'On Thyroxine': 'On Thyroxine Medication',
        'Query on Thyroxine': 'Query on Thyroxine', 'On Antithyroid Meds': 'On Antithyroid Medication',
        'Is Sick': 'Patient is Sick', 'Is Pregnant': 'Patient is Pregnant',
        'Had Thyroid Surgery': 'Had Thyroid Surgery', 'Had I131 Treatment': 'Had I131 Treatment',
        'Query Hypothyroid': 'Query Hypothyroid', 'Query Hyperthyroid': 'Query Hyperthyroid',
        'On Lithium': 'On Lithium Medication', 'Has Goitre': 'Has Goitre', 'Has Tumor': 'Has Tumor',
        'Psych Condition': 'Psychological Condition', 'TSH Level': 'TSH Level (mIU/L)',
        'T3 Level': 'T3 Level (ng/dL)', 'TT4 Level': 'Total T4 Level (µg/dL)',
        'T4U Level': 'T4U Level', 'FTI Level': 'Free Thyroxine Index (FTI)', 'TBG Level': 'TBG Level'
    },
    'Brain Stroke': {
        # TODO: Add full names for brain stroke features
        # Example: 'age': 'Age (Years)', 'avg_glucose_level': 'Average Glucose Level (mg/dL)'
    }
}


MODEL_MAPPING = {
    "Symptom Checker": {"name": "Symptom Checker", "features": [], "model": None, "scaler": None}, # Special case, handled in diagnose page
    "Diabetes": {"name": "Diabetes", "features": DIABETES_FEATURES, "model": diabetes_model, "scaler": diabetes_scaler},
    "Kidney Disease": {"name": "Kidney Disease", "features": KIDNEY_FEATURES, "model": kidney_model, "scaler": None},
    "Heart Disease": {"name": "Heart Disease", "features": HEART_FEATURES, "model": heart_model, "scaler": heart_scaler},
    "Hypertension": {"name": "Hypertension", "features": HYPERTENSION_FEATURES, "model": hypertension_model, "scaler": None},
    "Breast Cancer": {"name": "Breast Cancer", "features": BREAST_FEATURES, "model": breast_model, "scaler": None},
    "Lung Cancer": {"name": "Lung Cancer", "features": LUNG_FEATURES, "model": lung_model, "scaler": None},
    "Liver Disease": {"name": "Liver Disease", "features": LIVER_FEATURES, "model": liver_model, "scaler": None},
    "Thyroid Disease": {"name": "Thyroid Disease", "features": THYROID_FEATURES, "model": thyroid_model, "scaler": None},
    "Brain Stroke": {"name": "Brain Stroke", "features": BRAIN_STROKE_FEATURES, "model": brainStrokemodel, "scaler": None},
}

# --- Page Routing and Content ---
current_page = st.query_params.get("page", "home") # Use st.query_params

# --- Chat History Functions ---
@st.cache_data # Use st.cache_data for data that doesn't need to be re-evaluated unless input changes
def load_chat_history():
    if os.path.exists("chat_history.json"):
        try:
            with open("chat_history.json", "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return [("bot", "Chat history file might be corrupted. Starting fresh.")]
    return [("bot", f"Welcome to {APP_NAME} Chat! How can I help you today?")]

def save_chat_history(chat_history):
    try:
        with open("chat_history.json", "w", encoding="utf-8") as f:
            json.dump(chat_history, f)
    except Exception as e:
        st.error(f"Failed to save chat history: {e}")


# --- CSS Styling ---
st.markdown(f"""
    <style>
        /* --- Base & Body --- */
        body {{
            font-family: 'Roboto', 'Segoe UI', sans-serif;
            background-color: #eef2f7; /* Softer background */
            color: #333;
        }}
        /* Minimal reset/hiding */
        #MainMenu, footer {{ visibility: hidden; }}
        header {{ visibility: hidden; }} /* Hides the Streamlit hamburger menu bar */

        /* --- Page Container for Content --- */
        .block-container {{
            margin-top: -8rem !important; /* Pulls content upwards, adjust if topnav height changes */
            padding-left: 1.5rem;
            padding-right: 1.5rem;
            padding-bottom: 3rem;
            max-width: 1200px; /* Max width for content */
            margin-left: auto;
            margin-right: auto;
        }}

        /* --- Top Navigation Bar --- */
        .topnav {{
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: {NAV_HEIGHT_PX}px;
            background-color: #ffffff; /* White background */
            color: #2c3e50; /* Dark text */
            display: flex;
            align-items: center;
            padding: 0 2rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05); /* Softer shadow */
            z-index: 1000;
        }}
        .topnav .logo-text h1 {{ /* Changed class for clarity */
            color: #1a5a96; /* Primary brand color */
            font-size: 1.8rem;
            font-weight: 700;
            margin: 0;
        }}
        .topnav .nav-links {{
            margin-left: auto;
            display: flex;
        }}
        .topnav .nav-links a {{
            color: #555; /* Slightly lighter text for nav items */
            padding: 0.8rem 1rem;
            text-decoration: none;
            font-size: 0.95rem;
            font-weight: 500;
            border-radius: 4px;
            transition: background-color 0.2s ease, color 0.2s ease;
        }}
        .topnav .nav-links a:hover {{
            background-color: #eaf0f6; /* Light hover background */
            color: #1a5a96;
        }}
        .topnav .nav-links a.active {{
            background-color: #1a5a96; /* Primary color for active */
            color: white;
            font-weight: 600;
        }}

        /* --- Sidebar --- */
        [data-testid="stSidebar"] {{
            margin-top: {NAV_HEIGHT_PX}px; /* Push sidebar below topnav */
            background-color: #0E1629; /* Dark sidebar */
            padding-top: 1rem;
            border-right: none;
        }}
        .sidebar-title {{
            text-align: center;
            font-size: 1.5rem;
            font-weight: 600;
            color: #10A37F; /* Accent color */
            margin: 1rem 0 1.5rem 0;
        }}
        [data-testid="stSidebar"] .stImage img {{
            display: block;
            margin: 0 auto 1rem auto; /* Center logo */
            border-radius: 8px;
        }}
        [data-testid="stSidebar"] .stButton>button {{ /* Sidebar navigation buttons */
            background-color: transparent;
            color: #a8b3cf;
            border: 1px solid transparent;
            width: 100%;
            text-align: left;
            padding: 0.75rem 1rem;
            margin-bottom: 0.5rem;
            border-radius: 6px;
            font-weight: 500;
            transition: background-color 0.2s ease, color 0.2s ease, border-color 0.2s ease;
        }}
        [data-testid="stSidebar"] .stButton>button:hover {{
            background-color: #1C2B4A;
            color: #ffffff;
            border-left: 3px solid #10A37F;
        }}
        /* Active sidebar button - you might need JS to set this class or use query params for pages */
        [data-testid="stSidebar"] .stButton>button.active-sidebar-button {{
            background-color: #10A37F;
            color: #ffffff;
            font-weight: 600;
            border-left: 3px solid #ffffff;
        }}
        [data-testid="stSidebar"] hr {{
            border-color: #2c3e50;
            margin: 1.5rem 0;
        }}
        [data-testid="stSidebar"] h4 {{ color: #00A9E0; margin-top:1.5rem; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 0.5px; }}
        [data-testid="stSidebar"] p, [data-testid="stSidebar"] a {{ color: #a8b3cf; font-size: 0.9rem; }}
        [data-testid="stSidebar"] .stAlert {{ background-color: #1C2B4A; border-radius: 6px;}}

        /* --- Chat Interface (basic styling, enhance as needed) --- */
        .message-container {{ display: flex; max-width: 75%; margin-bottom: 0.8rem;}}
        .user-container {{ margin-left: auto; justify-content: flex-end; }}
        .bot-container {{ margin-right: auto; justify-content: flex-start; }}
        .message-bubble {{
            padding: 0.8rem 1.2rem; border-radius: 18px;
            line-height: 1.5; font-size: 0.95rem;
            display: inline-block; /* Changed from flex for natural text wrapping */
            box-shadow: 0 1px 2px rgba(0,0,0,0.08);
        }}
        .message-bubble .icon {{ margin-right: 0.7rem; font-size: 1.2em; }} /* If you add icons to messages */
        .user-bubble {{
            background: #007bff; color: white;
            border-bottom-right-radius: 2px;
            align-self: flex-end;
        }}
        .bot-bubble {{
            background: #f0f0f0; color: #333;
            border-bottom-left-radius: 2px;
            align-self: flex-start;
        }}
        [data-testid="stChatInput"] > div {{ /* Target the inner div for rounded corners */
            background-color: #ffffff;
            border-radius: 25px !important; /* Fully rounded input bar */
            padding: 0.2rem !important;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        }}
        [data-testid="stChatInput"] textarea {{ /* Target the textarea */
            border: none !important;
            box-shadow: none !important;
            padding: 0.8rem 1rem !important;
            background-color: transparent !important;
        }}

        /* General Element Styling */
        .stButton>button {{ /* Global button style, can be overridden */
            border-radius: 6px;
            padding: 0.6rem 1.2rem;
            font-weight: 500;
            transition: transform 0.1s ease-out;
        }}
        .stButton>button:active {{
            transform: scale(0.98);
        }}
        .stTextInput input, .stSelectbox div[data-baseweb="select"] > div, .stSlider div[data-testid="stTickBar"] {{
            border-radius: 6px !important;
        }}
        h1, h2, h3, h4, h5, h6 {{ color: #2c3e50; font-weight: 600;}}
        .section-header {{
            color: #1a5a96; /* Primary brand color for section headers */
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid #e0e0e0;
        }}
        .card {{ /* General purpose card */
            background-color: #ffffff;
            padding: 1.5rem;
            border-radius: 8px;
            box-shadow: 0 3px 10px rgba(0,0,0,0.07);
            margin-bottom: 1.5rem;
        }}
        .disease-card-home {{
            background-color: #ffffff; padding: 1.5rem; border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.06); text-align: center;
            transition: transform 0.2s ease, box-shadow 0.2s ease; margin-bottom: 1rem; height: 100%;
            display: flex; flex-direction: column; justify-content: center; align-items: center;
        }}
        .disease-card-home:hover {{ transform: translateY(-4px); box-shadow: 0 6px 15px rgba(0,0,0,0.1); }}
        .disease-card-home .emoji {{ font-size: 2.8rem; margin-bottom: 0.7rem; }}
        .disease-card-home h5 {{ color: #1a5a96; margin-bottom: 0.3rem; font-weight: 600; font-size: 1.1rem; }}

        /* Mobile Responsiveness */
        @media (max-width: 992px) {{ /* Tablet */
            .block-container {{
                padding-left: 1rem;    
                padding-right: 1rem;
            }}
            .topnav .logo-text h1 {{ font-size: 1.5rem; }}
            .topnav .nav-links a {{
                padding: 0.6rem 0.8rem; font-size: 0.9rem;
            }}
            .disease-card-home h5 {{ font-size: 1rem; }}
        }}
        @media (max-width: 768px) {{ /* Mobile */
            .block-container {{
                /* Adjust if topnav height changes for mobile */
                /* padding-top: {NAV_HEIGHT_PX + 10}px; */
                margin-top: -{NAV_HEIGHT_PX + 20}px !important; /* Further adjust pull for smaller topnav */
                padding-left: 0.5rem;
                padding-right: 0.5rem;
            }}
            .topnav {{
                padding: 0 1rem;
                height: {NAV_HEIGHT_PX - 10}px; /* Slightly smaller height */
            }}
            .topnav .logo-text h1 {{
                font-size: 1.3rem; /* Smaller title for mobile */
            }}
            .topnav .nav-links {{
                gap: 0.2rem; /* Reduce gap for smaller screens */
            }}
            .topnav .nav-links a {{
                padding: 0.5rem 0.6rem; font-size: 0.85rem; letter-spacing: -0.5px;
            }}
            h1 {{ font-size: 1.8rem; }} h2 {{ font-size: 1.5rem; }} h3 {{ font-size: 1.3rem; }}
            .message-container {{ max-width: 85%; }}
            .message-bubble {{ padding: 0.6rem 1rem; font-size: 0.9rem; }}
            .disease-card-home {{ padding: 1.2rem; }}
            .disease-card-home .emoji {{ font-size: 2.2rem; }}
            [data-testid="stCheckbox"] label span {{ font-size: 0.9rem; }}
        }}
    </style>
""", unsafe_allow_html=True)


# --- TOP NAVBAR HTML ---
# The empty div below acts as a spacer to push content down, adjust its height if topnav height changes.
st.markdown(f"""
<div class="topnav">
    <div class="logo-text"><h1>{APP_NAME}</h1></div>
    <div class="nav-links">
        <a href="?page=home"{' class="active"' if current_page == "home" else ''}>Home</a>
        <a href="?page=diagnose"{' class="active"' if current_page == "diagnose" else ''}>Diagnose</a>
        <a href="?page=chatbot"{' class="active"' if current_page == "chatbot" else ''}>Chatbot</a>
    </div>
</div>
<div style="height: {NAV_HEIGHT_PX + 5}px;"></div>
""", unsafe_allow_html=True)


# --- SIDEBAR ---
with st.sidebar:
    logo_path = "assets/logo.png" # Example path
    if os.path.exists(logo_path):
        st.image(logo_path, width=100) # Adjust width as needed
    else:
        # Fallback emoji or simple text if logo not found
        st.markdown(f"<h2 style='text-align:center; color:var(--accent-color);'>⚕️</h2>", unsafe_allow_html=True)

    st.markdown(f"<div class='sidebar-title'>{APP_NAME.split(' ', 1)[1] if ' ' in APP_NAME else APP_NAME}</div>", unsafe_allow_html=True)
    st.markdown("---")

    # You can add sidebar navigation using st.button or st.page_link (Streamlit 1.28+)
    # Example:
    # if st.button("🏠 Home", use_container_width=True): st.query_params["page"] = "home"; st.rerun()
    # if st.button("ախ Diagnose", use_container_width=True): st.query_params["page"] = "diagnose"; st.rerun()
    # if st.button("💬 Chatbot", use_container_width=True): st.query_params["page"] = "chatbot"; st.rerun()
    # st.markdown("---")


    st.markdown("<h4>App Information</h4>", unsafe_allow_html=True)
    st.markdown(f"<p>Version: {APP_VERSION}</p>", unsafe_allow_html=True)
    st.markdown("<h4>Contact Developer</h4>", unsafe_allow_html=True)
    st.markdown("<p><a href='mailto:subhashkumardev@outlook.com' target='_blank'>📧 Email Developer</a></p>", unsafe_allow_html=True)
    st.markdown("<p><a href='https://www.linkedin.com/in/subhashkumar-dev/' target='_blank'>🔗 LinkedIn</a></p>", unsafe_allow_html=True)
    st.markdown("<p><a href='https://github.com/Subhash捃' target='_blank'>💻 GitHub</a></p>", unsafe_allow_html=True) # Check your GitHub username
    st.markdown("---")
    st.info("This app is for educational purposes. Not a substitute for professional medical advice.")


# --- Helper Function to Sanitize String for Keys (Currently Unused) ---
# def sanitize_key(text):
# return re.sub(r'\W+', '_', text.lower())

# --- PDF Report Generation Function ---
def generate_pdf_report_bytes(result_data, current_disease_fullforms):
    pdf = FPDF()
    pdf.add_page()

    # Header
    pdf.set_font("Helvetica", 'B', 18)
    pdf.cell(0, 10, f"{APP_NAME} - Diagnosis Report", ln=True, align='C')
    pdf.set_font("Helvetica", '', 10)
    pdf.cell(0, 7, f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align='C')
    pdf.ln(10)

    # Diagnosis Summary
    pdf.set_font("Helvetica", 'B', 14)
    pdf.cell(0, 10, f"Disease Assessed: {result_data['title']}", ln=True)
    pdf.set_font("Helvetica", '', 12)
    diagnosis_color = (200, 0, 0) if result_data['diagnosis'] == "Positive" else (0, 100, 0) # Red for Positive, Green for Negative
    pdf.set_text_color(diagnosis_color[0], diagnosis_color[1], diagnosis_color[2])
    pdf.cell(0, 8, f"Diagnosis Result: {result_data['diagnosis']}", ln=True)
    pdf.set_text_color(0,0,0) # Reset to black
    pdf.cell(0, 8, f"Confidence Score: {result_data['confidence']}", ln=True)
    pdf.ln(7)

    # Patient Provided Details
    pdf.set_font("Helvetica", 'B', 12)
    pdf.cell(0, 8, "Patient Provided Details:", ln=True)
    pdf.set_font("Helvetica", '', 10)
    pdf.set_fill_color(240, 240, 240) # Light grey for alternate rows
    fill = False
    for feature_key in result_data.get('features_in_order', result_data['inputs'].keys()): # Iterate in defined order if available
        original_value = result_data['inputs'].get(feature_key, "N/A")
        display_key_pdf = current_disease_fullforms.get(feature_key, feature_key.replace("_", " ").title())

        # Handle multi-line cells better
        pdf.set_font("Helvetica", 'B', 10)
        pdf.multi_cell(50, 6, f"{display_key_pdf}: ", border=0, align='L', fill=fill) # Key part
        pdf.set_font("Helvetica", '', 10)
        pdf.multi_cell(0, 6, f"{original_value}", border=0, align='L', fill=fill) # Value part, ln=1 to move to next line
        fill = not fill
    pdf.ln(5)

    # Disclaimer
    pdf.set_font("Helvetica", 'I', 9)
    pdf.multi_cell(0, 5, "Disclaimer: This AI-generated report is for informational and educational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.", align='J')
    pdf.ln(3)
    pdf.multi_cell(0,5, f"Powered by {APP_NAME} ({APP_VERSION})", align='C')

    return pdf.output(dest='S').encode('utf-8') # Output as bytes

# --- Page Rendering Functions ---
def render_home_page():
    st.title(f"Welcome to {APP_NAME}")
    st.subheader("Your Smart AI Medical Diagnosis Assistant")
    st.markdown(f"""
    {APP_NAME.split(" ",1)[0]} leverages machine learning for preliminary insights into various health conditions.
    Analyze your medical parameters using our trained models to predict the likelihood of certain diseases.
    
    Navigate to the **Diagnose** section to use the prediction forms, or interact with our **Chatbot** for general health-related inquiries.

    **Important Disclaimer:** This platform is intended for educational and informational purposes ONLY.
    It is **NOT** a substitute for professional medical advice, diagnosis, or treatment.
    **Always consult with a qualified healthcare professional for any health concerns or before making any decisions related to your health.**
    """)
    st.markdown("---")
    st.markdown("<h3 class='section-header'>🧬 Diseases Covered</h3>", unsafe_allow_html=True)

    # Filter out "Symptom Checker" or models not meant for direct display here
    disease_display_list = {
        key: info for key, info in MODEL_MAPPING.items()
        if info.get("model") and info.get("features") # Ensure model and features are defined
    }

    emojis = ["🩸", "🩺", "❤️", "📈", "🎗️", "🫁", "🌿", "🦋", "🧠"] # Added one for Brain Stroke

    num_disease_cols = 3 if len(disease_display_list) >= 3 else (len(disease_display_list) if len(disease_display_list) > 0 else 1)
    if not disease_display_list:
        st.info("No disease models are currently configured for display on the home page.")
    else:
        disease_cols = st.columns(num_disease_cols)
        for i, (disease_key, disease_info) in enumerate(disease_display_list.items()):
            disease_name_home = disease_info["name"]
            with disease_cols[i % num_disease_cols]:
                emoji_char = emojis[i % len(emojis)]
                st.markdown(f"""
                <div class="disease-card-home">
                    <div class="emoji">{emoji_char}</div>
                    <h5>{disease_name_home}</h5>
                </div>
                """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True) # Adds some space
    st.markdown("---")
    st.markdown("<h3 class='section-header'>🚀 Technologies Used</h3>", unsafe_allow_html=True)
    st.markdown("""
    - **Backend & Machine Learning:** Python, Scikit-learn, TensorFlow/Keras, Joblib, Pandas, NumPy
    - **Frontend Web Framework:** Streamlit
    - **AI Chatbot Integration:** Google Generative AI (Gemini)
    - **Report Generation:** FPDF
    - **Styling:** Custom HTML & CSS
    """)
    st.markdown("---")
    st.markdown("<p style='text-align:center; font-size:0.9rem;'>Developed with passion by Subhash Kumar</p>", unsafe_allow_html=True)


def render_prediction_form_page(disease_key):
    disease_info = MODEL_MAPPING[disease_key]
    disease_name = disease_info["name"]
    features = disease_info["features"]
    model_instance = disease_info["model"]
    scaler_instance = disease_info.get("scaler")

    st.markdown(f"<h2 class='section-header'>Predict {disease_name}</h2>", unsafe_allow_html=True)

    form_key = f"{disease_key.lower().replace(' ', '_')}_prediction_form"
    with st.form(key=form_key):
        st.markdown("Please fill in the patient details below. Hover over input field names for more information where available.")
        inputs = {}
        
        num_features = len(features)
        if num_features == 0:
            st.warning(f"No features defined for {disease_name}. Cannot create prediction form.")
            st.form_submit_button("Predict Diagnosis", disabled=True) # Keep a button to avoid breaking form structure
            return

        num_cols = 1 if num_features <= 4 else (2 if num_features <= 12 else 3)
        form_cols = st.columns(num_cols)

        current_disease_fullforms = feature_fullforms.get(disease_name, {})

        for i, feature_name in enumerate(features):
            with form_cols[i % num_cols]:
                f_key = feature_name.strip()
                display_label = current_disease_fullforms.get(f_key, f_key.replace("_", " ").title())
                tooltip_text = f"Enter value for {display_label}" # Basic tooltip

                if f_key in FEATURE_INPUTS:
                    input_type, *params = FEATURE_INPUTS[f_key]
                    
                    if input_type == 'slider':
                        min_val, max_val, default_val = float(params[0]), float(params[1]), float(params[2])
                        step_val = float(params[3]) if len(params) > 3 else (0.01 if isinstance(default_val, float) and default_val % 1 != 0 else 1.0)
                        inputs[f_key] = st.slider(
                            display_label, min_value=min_val, max_value=max_val,
                            value=default_val, step=step_val, key=f"{f_key}_slider_{disease_key}",
                            help=tooltip_text
                        )
                    elif input_type == 'select':
                        options = params[0]
                        default_index = 0 # Or try to find a sensible default
                        inputs[f_key] = st.selectbox(
                            display_label, options, index=default_index,
                            key=f"{f_key}_select_{disease_key}", help=tooltip_text
                        )
                else: # Fallback to text input if not defined in FEATURE_INPUTS
                    inputs[f_key] = st.text_input(display_label, key=f"{f_key}_text_{disease_key}", help=tooltip_text)

        submitted = st.form_submit_button("🔍 Predict Diagnosis", use_container_width=True)

    if submitted:
        def convert_input_value(val_str): # Ensure this is robust
            val_lower = str(val_str).lower()
            if val_lower in ['yes', 'present', 'male', 'normal', 'true', 'm']: return 1
            if val_lower in ['no', 'notpresent', 'female', 'abnormal', 'false', 'f']: return 0
            try: return float(val_str)
            except (ValueError, TypeError):
                st.warning(f"Could not convert '{val_str}' to a number. Using as is or 0 if critical.")
                return 0 # Or handle error more strictly

        processed_values = []
        valid_inputs = True
        for f_name in features:
            f_key_strip = f_name.strip()
            if f_key_strip not in inputs:
                st.error(f"Missing input for feature: {f_key_strip}. Please ensure all fields are filled.")
                valid_inputs = False
                break
            processed_values.append(convert_input_value(inputs[f_key_strip]))
        
        if not valid_inputs:
            return # Stop processing if inputs are not valid

        try:
            data_df = pd.DataFrame([processed_values], columns=[f.strip() for f in features])
            data_to_predict = data_df.copy()

            if scaler_instance:
                try:
                    # Ensure columns match scaler's expectations if scaler is picky
                    scaled_data_array = scaler_instance.transform(data_df)
                    data_to_predict = pd.DataFrame(scaled_data_array, columns=data_df.columns)
                except Exception as e:
                    st.error(f"Error during data scaling: {e}. Ensure input data matches model training format.")
                    st.caption("This might happen if the number or order of features is incorrect, or if data types are mismatched.")
                    return

            prediction = model_instance.predict(data_to_predict)[0]
            confidence = "N/A"
            if hasattr(model_instance, 'predict_proba'):
                probs = model_instance.predict_proba(data_to_predict)[0]
                # For binary classification:
                # If prediction is 1, confidence is prob of class 1.
                # If prediction is 0, confidence is prob of class 0.
                confidence_score = probs[int(prediction)] if len(probs) > int(prediction) else probs.max()
                confidence = f"{confidence_score:.2%}"


            diagnosis_result = "Positive" if prediction == 1 else "Negative"

            st.session_state['last_prediction_details'] = {
                'form_key': form_key, # Store which form generated this
                'title': disease_name,
                'inputs': inputs.copy(), # Original string inputs for the report
                'processed_inputs_for_debug': processed_values, # For debugging
                'diagnosis': diagnosis_result,
                'confidence': confidence,
                'features_in_order': [f.strip() for f in features] # Store feature order for PDF
            }
            st.rerun() # Rerun to display results outside the form block

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.error("Please ensure all inputs are valid and match the expected format for the model.")
            # st.write("Data sent to model (potentially scaled):", data_to_predict) # For debugging

    # Display results and PDF download button if a prediction was made by THIS form
    if 'last_prediction_details' in st.session_state and \
       st.session_state['last_prediction_details'].get('form_key') == form_key:

        result_data = st.session_state['last_prediction_details']
        
        st.markdown("---")
        st.subheader("📋 Diagnosis Result")
        
        if result_data['diagnosis'] == "Positive":
            st.error(f"**Diagnosis:** {result_data['diagnosis']}")
        else:
            st.success(f"**Diagnosis:** {result_data['diagnosis']}")
        st.info(f"**Confidence Score:** {result_data['confidence']}")
        
        if result_data['diagnosis'] == "Positive":
            st.warning("The model predicts a positive result. Please consult a healthcare professional for confirmation and further guidance.")
        else:
            st.success("The model predicts a negative result. However, if you have concerns, please consult a healthcare professional.")
        
        st.balloons()

        try:
            current_disease_fullforms_for_pdf = feature_fullforms.get(result_data['title'], {})
            pdf_bytes = generate_pdf_report_bytes(result_data, current_disease_fullforms_for_pdf)
            current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
            pdf_file_name = f"{result_data['title'].replace(' ', '_')}_Report_{current_datetime}.pdf"

            st.download_button(
                label="📥 Download Report as PDF",
                data=pdf_bytes,
                file_name=pdf_file_name,
                mime="application/pdf",
                key=f"download_pdf_{disease_key}",
                use_container_width=True
            )
        except Exception as e:
            st.error(f"Could not generate PDF report: {e}")

        with st.expander("View Submitted Details"):
            for f_key, val in result_data['inputs'].items():
                display_key_exp = feature_fullforms.get(result_data['title'], {}).get(f_key, f_key.replace("_", " ").title())
                st.text(f"{display_key_exp}: {val}")


def render_diagnose_page():
    st.title("🩺 Disease Diagnosis Forms")
    st.markdown("""
        Select a disease from the dropdown below to access its specific prediction form.
        Fill in the required medical parameters, and our AI model will provide a preliminary diagnosis based on the trained data.
        Remember, this is not a substitute for professional medical advice.
    """)
    st.markdown("---")

    available_models_for_selection = {
        key: val["name"] for key, val in MODEL_MAPPING.items()
        # Exclude symptom checker if it's not a form, or models without actual model instances/features
        if (val.get("model") and val.get("features")) or key == "Symptom Checker"
    }
    if not available_models_for_selection:
        st.warning("No diagnosis models are currently available or configured correctly. Please check the application setup.")
        return

    # Use model names for selection, then map back to key
    model_display_names = list(available_models_for_selection.values())
    
    default_selection_index = 0
    if 'diagnose_selection_name' in st.session_state: # Store the name
        try:
            default_selection_index = model_display_names.index(st.session_state['diagnose_selection_name'])
        except ValueError:
            st.session_state.pop('diagnose_selection_name', None)

    selected_display_name = st.selectbox(
        "Choose a Disease Model:",
        options=model_display_names,
        index=default_selection_index,
        key="diagnose_disease_selector_dropdown" # Unique key for the selectbox
    )

    selected_model_key = None
    if selected_display_name:
        for key, name_val in available_models_for_selection.items():
            if name_val == selected_display_name:
                selected_model_key = key
                break
    
    if selected_model_key:
        st.session_state['diagnose_selection_name'] = selected_display_name # Store the display name

        if selected_model_key == "Symptom Checker":
            st.info("The Symptom Checker is intended for general symptom analysis. For specific disease predictions, please choose another model. You can also use the Chatbot for symptom inquiries.")
            # If Symptom Checker has its own page/logic, call it here
            # For now, if it has no model, it will be caught below
            if not MODEL_MAPPING[selected_model_key].get("model"):
                 st.warning("The Symptom Checker feature is under development or not a form-based prediction.")
                 return

        model_info = MODEL_MAPPING[selected_model_key]
        if not model_info.get("model"):
            st.error(f"The model for '{selected_display_name}' is not loaded or configured. Please contact support or select another model.")
            return
        if not model_info.get("features") and selected_model_key != "Symptom Checker":
             st.error(f"The feature list for '{selected_display_name}' is not defined. Cannot render the prediction form.")
             return

        render_prediction_form_page(selected_model_key)
    else:
        st.info("Please select a disease model from the dropdown to proceed.")


def render_chatbot_page():
    st.title("💬 IntelliMed AI Chatbot")
    st.markdown("""
    Ask general health-related questions, inquire about symptoms, or get information about medical conditions.
    **This chatbot does not provide medical diagnoses or treatment advice.** Always consult a healthcare professional for personal medical concerns.
    """)

    if not GEMINI_CONFIGURED or not gemini_llm_model:
        st.error("Chatbot is unavailable due to API key or model configuration issues.")
        return

    # Initialize chat history in session state if not present
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = load_chat_history() # Load from file or default

    # Display chat messages
    chat_container = st.container() # Use a container for chat messages
    with chat_container:
        for role, text in st.session_state.chat_messages:
            if role == "user":
                st.markdown(f"""
                <div class="message-container user-container">
                    <div class="message-bubble user-bubble">👤 You: {text}</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="message-container bot-container">
                    <div class="message-bubble bot-bubble">🤖 Bot: {text}</div>
                </div>""", unsafe_allow_html=True)

    # Chat input
    prompt = st.chat_input("Ask me anything about health...")
    if prompt:
        st.session_state.chat_messages.append(("user", prompt))
        with chat_container: # Display user message immediately
             st.markdown(f"""
                <div class="message-container user-container">
                    <div class="message-bubble user-bubble">👤 You: {prompt}</div>
                </div>""", unsafe_allow_html=True)
        
        try:
            with st.spinner("Thinking..."):
                # For Gemini, you might need to format history if using advanced chat
                # For simple Q&A:
                response = gemini_llm_model.generate_content(prompt)
                bot_response = response.text
            st.session_state.chat_messages.append(("bot", bot_response))
            save_chat_history(st.session_state.chat_messages) # Save after bot responds
            st.rerun() # Rerun to display the bot's response in the chat_container

        except Exception as e:
            st.error(f"Error communicating with the AI: {e}")
            st.session_state.chat_messages.append(("bot", "Sorry, I encountered an error. Please try again."))
            save_chat_history(st.session_state.chat_messages)


# --- Main App Logic ---
if current_page == "home":
    render_home_page()
elif current_page == "diagnose":
    render_diagnose_page()
elif current_page == "chatbot":
    render_chatbot_page()
else: # Default to home page if query param is invalid
    st.query_params["page"] = "home"
    render_home_page()
    st.rerun()