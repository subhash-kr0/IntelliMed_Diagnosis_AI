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
import markdown2 
from xhtml2pdf import pisa 
import io
import tensorflow as tf
import asyncio 
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib import colors

# Load environment variables
load_dotenv()

# --- Constants ---
PAGE_HOME = "PAGE_HOME"
PAGE_CHATBOT = "PAGE_CHATBOT"
PAGE_DIAGNOSE = "PAGE_DIAGNOSE"
DIAGNOSE_MODE_FORM = "Form Based"
DIAGNOSE_MODE_IMAGE = "Image Based"
IMAGE_AI_GEMINI = "Gemini"
IMAGE_AI_BRAIN_STROKE = "Brain Stroke"
CHATBOT_GEMINI = "ChatBot (Gemini)"
CHATBOT_MISTRAL = "Mistral"
NAV_HEIGHT_PX = 60  
APP_NAME = " 🩺 MedDio"
APP_VERSION = "1.0.0"

# --- Page Config ---
st.set_page_config(layout='wide', page_icon='🎈', initial_sidebar_state='auto')

# --- API Keys & Model Configuration ---
gemini_key = os.getenv("GEMINI_API_KEY")  #
genai.configure(api_key=gemini_key)
gemini_flash_lite_model = genai.GenerativeModel(model_name="models/gemini-2.5-flash-preview-04-17")
gemini_image_model = genai.GenerativeModel(model_name="models/gemini-2.5-flash-preview-04-17") 

# --- Caching for Model and Scaler Loading ---
@st.cache_resource
def load_diabetes_model_and_scaler():
    model = joblib.load('./models/diabetes_model.pkl')
    scaler = joblib.load('./models/diabetes_scaler.pkl')
    return model, scaler

@st.cache_resource
def load_kidney_model():
    return joblib.load('./models/kidneyDisease_model.pkl')

@st.cache_resource
def load_heart_model_and_scaler():
    model = joblib.load('models/heartDisease_randomForest_model.pkl')
    scaler = joblib.load('models/heartDisease_scaler.pkl')
    return model, scaler

@st.cache_resource
def load_hypertension_model():
    return joblib.load('models/hypertension_model.pkl')

@st.cache_resource
def load_breast_cancer_model():
    return joblib.load('./models/breastCancer_randomForest_model.pkl')

@st.cache_resource
def load_lung_cancer_model():
    return joblib.load('./models/lungCancer_XGBClassifier_model.pkl')

@st.cache_resource
def load_liver_disease_model():
    return joblib.load('./models/liverDisease_rf_model.pkl')

@st.cache_resource
def load_thyroid_model():
    return joblib.load('./models/thyroid_cat_model.pkl')

@st.cache_resource
def load_brainstroke_tf_model():
    return tf.keras.models.load_model("models/brainstroke_model.h5")

diabetes_model, diabetes_scaler = load_diabetes_model_and_scaler()
kidney_model = load_kidney_model()
heart_model, heart_scaler = load_heart_model_and_scaler()
hypertension_model = load_hypertension_model()
breast_model = load_breast_cancer_model()
lung_model = load_lung_cancer_model()
liver_model = load_liver_disease_model()
thyroid_model = load_thyroid_model()
brainStrokemodel = load_brainstroke_tf_model()

# --- Feature Lists (as provided) ---
DIABETES_FEATURES = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
KIDNEY_FEATURES = ['age', 'bp', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 'bu', 'sc', 'pot', 'wc', 'htn', 'dm', 'cad', 'pe', 'ane']
HEART_FEATURES = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach','exang','oldpeak','slope','ca','thal']
HYPERTENSION_FEATURES = ['Age', 'BMI', 'Systolic_BP','Diastolic_BP',  'Total_Cholesterol']
BREAST_FEATURES = ['mean_radius', 'mean_texture', 'mean_perimeter', 'mean_area', 'mean_smoothness', 'compactness_mean', 'concavity_mean', 
                   'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean']
LUNG_FEATURES = ['GENDER', 'AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE', 'CHRONIC_DISEASE', 'FATIGUE', 'ALLERGY', 'WHEEZING',
                  'ALCOHOL_CONSUMING', 'COUGHING', 'SHORTNESS_OF_BREATH', 'SWALLOWING_DIFFICULTY', 'CHEST_PAIN']
LIVER_FEATURES = ['Age', 'Gender', 'Total_Bilirubin', 'Alkaline_phosphate', 'Alamine_Aminotransferace', 'Aspartate_Amino', 'Protien', 'Albumin',
 'Albumin_Globulin_ratio']
THYROID_FEATURES = ['Age','Sex','On Thyroxine','Query on Thyroxine','On Antithyroid Meds','Is Sick','Is Pregnant','Had Thyroid Surgery',
                    'Had I131 Treatment','Query Hypothyroid','Query Hyperthyroid','On Lithium','Has Goitre','Has Tumor','Psych Condition',
                    'TSH Level','T3 Level','TT4 Level','T4U Level','FTI Level', 'TBG Level']


# --- Feature Input Definitions (as provided, with slight adjustments for consistency if needed) ---
FEATURE_INPUTS = {
    # DIABETES
    'Pregnancies': ('slider', 0, 20, 1), 'Glucose': ('slider', 40, 300, 1),
    'BloodPressure': ('slider', 30, 180, 1), 'SkinThickness': ('slider', 0, 100, 1),
    'Insulin': ('slider', 0, 900, 1), 'BMI': ('slider', 10.0, 70.0, 0.1),
    'DiabetesPedigreeFunction': ('slider', 0.01, 2.5, 0.01), 'Age': ('slider', 1, 120, 1),
    # KIDNEY 
    'age': ('slider', 1, 100, 1), 'bp': ('slider', 40, 180, 1), 'al': ('slider', 0, 5, 1),
    'su': ('slider', 0, 5, 1), 'rbc': ('select', ['normal', 'abnormal']),
    'pc': ('select', ['normal', 'abnormal']), 'pcc': ('select', ['present', 'notpresent']),
    'ba': ('select', ['present', 'notpresent']), 'bgr': ('slider', 70, 500, 1),
    'bu': ('slider', 5, 200, 1), 'sc': ('slider', 0.1, 20.0, 0.1),
    'pot': ('slider', 2.5, 10.0, 0.1), 'wc': ('slider', 1000, 25000, 100),
    'htn': ('select', ['yes', 'no']), 'dm': ('select', ['yes', 'no']),
    'cad': ('select', ['yes', 'no']), 'pe': ('select', ['yes', 'no']),
    'ane': ('select', ['yes', 'no']),
    # HEART
    'sex': ('select', ['male', 'female']), 'cp': ('slider', 0, 3, 1),
    'trestbps': ('slider', 80, 200, 1), 'chol': ('slider', 100, 600, 1),
    'fbs': ('select', ['yes', 'no']), 'restecg': ('slider', 0, 2, 1),
    'thalach': ('slider', 60, 220, 1), 'exang': ('select', ['yes', 'no']),
    'oldpeak': ('slider', 0.0, 6.0, 0.1), 'slope': ('slider', 0, 2, 1),
    'ca': ('slider', 0, 4, 1), 'thal': ('slider', 0, 3, 1),
    # HYPERTENSION 
    'Systolic_BP': ('slider', 80, 200, 1), 'Diastolic_BP': ('slider', 40, 120, 1),
    'Total_Cholesterol': ('slider', 100, 400, 1),
    'smoking': ('select', ['yes', 'no']), 'exercise': ('select', ['yes', 'no']),
    'alcohol': ('select', ['yes', 'no']),
    # BREAST CANCER
    'mean_radius': ('slider', 5.0, 30.0, 0.1), 'mean_texture': ('slider', 5.0, 40.0, 0.1),
    'mean_perimeter': ('slider', 30.0, 200.0, 0.1), 'mean_area': ('slider', 100.0, 2500.0, 1.0),
    'mean_smoothness': ('slider', 0.05, 0.2, 0.001), 'compactness_mean': ('slider', 0.01, 1.0, 0.01),
    'concavity_mean': ('slider', 0.01, 1.0, 0.01), 'concave points_mean': ('slider', 0.01, 0.5, 0.01),
    'symmetry_mean': ('slider', 0.1, 0.5, 0.01), 'fractal_dimension_mean': ('slider', 0.01, 0.2, 0.001),
    # LUNG CANCER
    'GENDER': ('select', ['Male', 'Female']), 'AGE': ('slider', 10, 100, 1),
    'SMOKING': ('select', ['Yes', 'No']), 'YELLOW_FINGERS': ('select', ['Yes', 'No']),
    'ANXIETY': ('select', ['Yes', 'No']), 'PEER_PRESSURE': ('select', ['Yes', 'No']),
    'CHRONIC_DISEASE': ('select', ['Yes', 'No']), 'FATIGUE': ('select', ['Yes', 'No']), 
    'ALLERGY': ('select', ['Yes', 'No']), 'WHEEZING': ('select', ['Yes', 'No']),
    'ALCOHOL_CONSUMING': ('select', ['Yes', 'No']), 'COUGHING': ('select', ['Yes', 'No']), 
    'SHORTNESS_OF_BREATH': ('select', ['Yes', 'No']), 'SWALLOWING_DIFFICULTY': ('select', ['Yes', 'No']),
    'CHEST_PAIN': ('select', ['Yes', 'No']),
    # LIVER
    'Gender': ('select', ['Male', 'Female']), 'Total_Bilirubin': ('slider', 0.1, 10.0, 0.1),
    'Alkaline_phosphate': ('slider', 50, 3000, 1), 'Alamine_Aminotransferace': ('slider', 1, 2000, 1),
    'Aspartate_Amino': ('slider', 1, 2000, 1), 'Protien': ('slider', 2.0, 10.0, 0.1), # 'Protien' vs 'Protein'
    'Albumin': ('slider', 1.0, 6.0, 0.1), 'Albumin_Globulin_ratio': ('slider', 0.1, 3.0, 0.1),
    # THYROID
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

# --- Symptom Data for Multi-Disease Diagnosis (as provided) ---
diseases_data = {
    'Common Cold': ['runny nose', 'sore throat', 'cough', 'sneezing', 'mild headache', 'fatigue', 'nasal congestion'],
    'Influenza (Flu)': ['fever', 'body aches', 'chills', 'fatigue', 'cough', 'sore throat', 'headache', 'muscle pain', 'nausea', 'vomiting'],
    'Allergies': ['sneezing', 'itchy eyes', 'runny nose', 'nasal congestion', 'itchy throat', 'watery eyes'],
    'Strep Throat': ['sore throat', 'fever', 'swollen lymph nodes', 'difficulty swallowing', 'red spots on roof of mouth', 'headache', 'nausea', 'vomiting'],
    'Bronchitis': ['cough', 'chest discomfort', 'shortness of breath', 'fatigue', 'sore throat', 'mild fever'],
    'Pneumonia': ['cough', 'fever', 'chills', 'shortness of breath', 'chest pain', 'fatigue', 'nausea', 'vomiting'],
    'Migraine': ['severe headache', 'pulsating pain', 'sensitivity to light', 'sensitivity to sound', 'nausea', 'vomiting', 'aura'],
    'Tension Headache': ['mild headache', 'pressure around head', 'neck pain', 'shoulder pain'],
    'Gastroenteritis (Stomach Flu)': ['nausea', 'vomiting', 'diarrhea', 'abdominal cramps', 'mild fever', 'fatigue'],
    'Urinary Tract Infection (UTI)': ['frequent urination', 'painful urination', 'burning sensation during urination', 'cloudy urine', 'pelvic pain', 'fever', 'chills']
}

# --- Feature Full Forms (as provided) ---
feature_fullforms = {
    # 'Choose Disease': {'Select the disease you want to diagnose': 'Please choose one disease from the list'},
    'Symptom Checker': {'1': "Instant health insights based on your symptoms.",'2':"Check what might be causing your symptoms in seconds.",
                        '3':"Smart diagnosis support from the symptoms you enter.",'4':"Input your symptoms. Get possible conditions instantly."},
    'Diabetes': {'Pregnancies': 'Number of times pregnant', 'Glucose': 'Plasma glucose concentration', 'BloodPressure': 'Diastolic blood pressure (mm Hg)',
                  'SkinThickness': 'Triceps skin fold thickness (mm)', 'Insulin': '2-Hour serum insulin (mu U/ml)', 'BMI': 'Body Mass Index', 
                  'DiabetesPedigreeFunction': 'Diabetes pedigree function', 'Age': 'Age in years'},
    'Kidney Disease': {'age': 'Age', 'bp': 'Blood Pressure', 'al': 'Albumin', 'su': 'Sugar', 'rbc': 'Red Blood Cells', 'pc': 'Pus Cell', 'pcc': 'Pus Cell Clumps',
                        'ba': 'Bacteria', 'bgr': 'Blood Glucose Random', 'bu': 'Blood Urea', 'sc': 'Serum Creatinine', 'pot': 'Potassium', 'wc': 'White Blood Cell Count',
                          'htn': 'Hypertension', 'dm': 'Diabetes Mellitus', 'cad': 'Coronary Artery Disease', 'pe': 'Pedal Edema', 'ane': 'Anemia'},
    'Heart Disease': {'age': 'Age', 'sex': 'Sex', 'cp': 'Chest Pain Type', 'trestbps': 'Resting Blood Pressure', 'chol': 'Serum Cholesterol', 
                      'fbs': 'Fasting Blood Sugar > 120 mg/dl', 'restecg': 'Resting ECG Results', 'thalach': 'Maximum Heart Rate Achieved', 
                      'exang': 'Exercise Induced Angina', 'oldpeak': 'ST Depression', 'slope': 'Slope of ST Segment', 'ca': 'Number of Major Vessels Colored',
                      'thal': 'Thalassemia'},
    'Hypertension': {'Age':'Age', 'BMI': 'Body Mass Index', 'Systolic_BP': 'Systolic Blood Pressure', 'Diastolic_BP': 'Diastolic Blood Pressure', 
                     'Total_Cholesterol':'Total Cholesterol', 'smoking': 'Smoking Habit', 'exercise': 'Physical Activity', 'alcohol': 'Alcohol Consumption'},
    'Breast Cancer': {'mean_radius': 'Mean Radius', 'mean_texture': 'Mean Texture', 'mean_perimeter': 'Mean Perimeter', 'mean_area': 'Mean Area', 
                      'mean_smoothness': 'Mean Smoothness', 'compactness_mean': 'Mean Compactness', 'concavity_mean': 'Mean Concavity', 
                      'concave points_mean': 'Mean Concave Points', 'symmetry_mean': 'Mean Symmetry', 'fractal_dimension_mean': 'Mean Fractal Dimension'},
    'Lung Cancer': {'GENDER': 'Gender', 'AGE': 'Age', 'SMOKING': 'Smoking', 'YELLOW_FINGERS': 'Yellow Fingers', 'ANXIETY': 'Anxiety', 
                    'PEER_PRESSURE': 'Peer Pressure', 'CHRONIC_DISEASE': 'Chronic Disease', 'FATIGUE': 'Fatigue', 'ALLERGY': 'Allergy', 'WHEEZING': 'Wheezing', 
                    'ALCOHOL_CONSUMING': 'Alcohol Consumption', 'COUGHING': 'Coughing', 'SHORTNESS_OF_BREATH': 'Shortness of Breath', 
                    'SWALLOWING_DIFFICULTY': 'Swallowing Difficulty', 'CHEST_PAIN': 'Chest Pain'},
    'Liver Disease': {'Age': 'Age', 'Gender': 'Gender', 'Total_Bilirubin': 'Total Bilirubin', 'Alkaline_phosphate': 'Alkaline Phosphotase', 
                      'Alamine_Aminotransferace': 'Alamine Aminotransferase', 'Aspartate_Amino': 'Aspartate Aminotransferase', 'Protien': 'Total Protein', 
                      'Albumin': 'Albumin Level', 'Albumin_Globulin_ratio': 'Albumin to Globulin Ratio'},
    'Thyroid Disease': {'Age': 'Age', 'Sex': 'Sex', 'On Thyroxine': 'On Thyroxine Medication', 'Query on Thyroxine': 'Query on Thyroxine', 
                        'On Antithyroid Meds': 'On Antithyroid Medication', 'Is Sick': 'Currently Sick', 'Is Pregnant': 'Currently Pregnant', 
                        'Had Thyroid Surgery': 'History of Thyroid Surgery', 'Had I131 Treatment': 'History of I131 Treatment', 'Query Hypothyroid': 'Query Hypothyroid',
                          'Query Hyperthyroid': 'Query Hyperthyroid', 'On Lithium': 'On Lithium Medication', 'Has Goitre': 'Presence of Goitre', 
                          'Has Tumor': 'Presence of Tumor', 'Psych Condition': 'Psychological Condition Reported', 'TSH Level': 'TSH Level', 'T3 Level': 'T3 Level',
                        'TT4 Level': 'Total T4 Level', 'T4U Level': 'T4U Level', 'FTI Level': 'Free Thyroxine Index (FTI) Level', 'TBG Level': 'TBG Level'}
}


# --- MODEL_MAPPING (Ensure HYPERTENSION_FEATURES is updated if necessary) ---
MODEL_MAPPING = {
    "Symptom Checker": {"name": "Symptom Checker", "features": [], "model": None, "scaler": None},
    "Diabetes": {"name": "Diabetes", "features": DIABETES_FEATURES, "model": diabetes_model, "scaler": diabetes_scaler},
    "Kidney Disease": {"name": "Kidney Disease", "features": KIDNEY_FEATURES, "model": kidney_model, "scaler": None},
    "Heart Disease": {"name": "Heart Disease", "features": HEART_FEATURES, "model": heart_model, "scaler": heart_scaler},
    "Hypertension": {"name": "Hypertension", "features": HYPERTENSION_FEATURES, "model": hypertension_model, "scaler": None},
    "Breast Cancer": {"name": "Breast Cancer", "features": BREAST_FEATURES, "model": breast_model, "scaler": None},
    "Lung Cancer": {"name": "Lung Cancer", "features": LUNG_FEATURES, "model": lung_model, "scaler": None},
    "Liver Disease": {"name": "Liver Disease", "features": LIVER_FEATURES, "model": liver_model, "scaler": None},
    "Thyroid Disease": {"name": "Thyroid Disease", "features": THYROID_FEATURES, "model": thyroid_model, "scaler": None},
}


# --- Consolidated CSS Styles ---
st.markdown(f"""
    <style>
        /* --- Base & Body --- */
        body {{
            font-family: 'Roboto', 'Segoe UI', sans-serif;
            background-color: #eef2f7; /* Softer background from first block */
            color: #333;
        }}
        #MainMenu, footer, header {{ visibility: hidden; }}

        /* --- Page Container for Content (from first block) --- */
        .block-container {{
            padding-top: {NAV_HEIGHT_PX + 15}px;  /* Space for fixed topnav */
            margin-top: -9rem !important; /* Pulls content upwards */
            padding-left: 1.5rem;
            padding-right: 1.5rem;
            padding-bottom: 3rem;
            max-width: 1200px;
            margin: 0 auto;
            margin-bottom: -3rem; /* Adjusted for better spacing */
        }}

        /* --- Top Navigation Bar (Styles from second block, as it matches the rendered HTML) --- */
        .topnav {{
            position: fixed;
            top: 0; /* Adjusted from 10px for consistency with NAV_HEIGHT_PX logic if page-container is used */
            right: 0; /* Added for full width potential */
            # width: 100%; /* Adjusted from 50% for better default */
            height: {NAV_HEIGHT_PX}px; /* Using NAV_HEIGHT_PX from first block's concept */
            background-color: #ffffff;
            color: #2c3e50;
            display: flex;
            align-items: center;
            padding: 0 2rem; /* From first block */
            box-shadow: 0 2px 4px rgba(0,0,0,0.05); /* From first block */
            z-index: 1000;
        }}
        .topnav h1 {{ /* For <h1>🩺 IntelliMed</h1> */
            color: #1a5a96; /* Primary brand color from first block */
            font-size: 1.8rem; /* From first block */
            font-weight: 700; /* From first block */
            margin-right: auto; /* Pushes links to the right */
        }}
        .topnav a {{ /* For nav links */
            right: 0; /* Added for full width potential */
            color: #555;
            padding: 0.8rem 1rem;
            text-decoration: none;
            font-size: 0.95rem;
            font-weight: 500;
            border-radius: 4px;
            transition: background-color 0.2s ease, color 0.2s ease;
            margin-left: 10px;
        }}


        .topnav a:hover {{
            background-color: #eaf0f6;
            color: #1a5a96;
        }}
        /* Active link styling can be tricky with st.markdown. Simpler hover is reliable.
           If query_params are used to set 'active', JS would be needed or server-side regeneration of HTML.
           The second block had .topnav a.active - this assumes class="active" is on the link.
        */
        .topnav a[href="?page={st.query_params.get("page", PAGE_HOME)}"] {{
             background-color: #1a5a96;
             color: white;
             font-weight: 600;
        }}

        /* --- Sidebar (Styles from second block, enhanced by first if compatible) --- */
        [data-testid="stSidebar"] {{
            background: linear-gradient(to bottom, #0f172a, #1e293b); /* From second block */
            color: white; /* From second block */
            padding: 1.5rem 1rem; /* From second block */
            border-right: none;
        }}
        .sidebar-title {{ /* From second block */
            text-align: center;
            font-size: 1.5rem; /* Adjusted from 20px for rem consistency */
            font-weight: 600;
            color: #10A37F; /* Accent color from first block's concept */
            margin: 1rem 0 1.5rem 0; /* Consistent margins */
        }}
        [data-testid="stSidebar"] .stImage img {{
            display: block;
            margin: 0 auto 1rem auto;
            border-radius: 8px;
        }}
        [data-testid="stSidebar"] .stButton>button {{
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
        [data-testid="stSidebar"] .stButton>button.active-sidebar-button {{
            background-color: #10A37F;
            color: #ffffff;
            font-weight: 600;
            border-left: 3px solid #ffffff;
        }}
        [data-testid="stSidebar"] hr {{ border-color: #2c3e50; margin: 1.5rem 0; }}
        [data-testid="stSidebar"] h4 {{ color: #00A9E0; margin-top:1.5rem; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 0.5px; }}
        [data-testid="stSidebar"] p, [data-testid="stSidebar"] a {{ color: #a8b3cf; font-size: 0.9rem; }}
        [data-testid="stSidebar"] .stAlert {{ background-color: #1C2B4A; border-radius: 6px;}}

        /* --- Chat Interface (Combined and refined from both blocks) --- */
        .chat-box {{ /* Using style from second block primarily, as it has animation and more specifics */
            height: 500px; /* From first block, or adjust as needed */
            overflow-y: auto;
            border-radius: 8px;
            padding: 1.5rem; /* From first block */
            background-color: #ffffff; /* From first block */
            box-shadow: 0 4px 8px rgba(0,0,0,0.1); /* Enhanced shadow */
            margin-bottom: 1rem;
            display: flex;
            flex-direction: column;
            gap: 0.8rem;
            animation: fadeIn 0.5s ease-in-out; /* From second block */
        }}
        @keyframes fadeIn {{ from {{ opacity: 0; }} to {{ opacity: 1; }} }}

        .message-container {{ display: flex; max-width: 75%; }}
        .user-container {{ margin-left: auto; justify-content: flex-end; }}
        .bot-container {{ margin-right: auto; justify-content: flex-start; }}

        .message-bubble {{ /* From first block */
            padding: 0.8rem 1.2rem; border-radius: 18px;
            line-height: 1.5; font-size: 0.95rem;
            display: flex; align-items: center;
            box-shadow: 0 1px 2px rgba(0,0,0,0.08);
        }}
        .message-bubble .icon {{ margin-right: 0.7rem; font-size: 1.2em; }}
        .user-bubble {{ background: #007bff; color: white; border-bottom-right-radius: 5px; }} /* From first block */
        .bot-bubble {{ background: #f0f0f0; color: #333; border-bottom-left-radius: 5px; }} /* From first block */

        /* Specific message styling from second block for Gemini chat page */
        .message.user {{
            background: linear-gradient(to left,  #2e3192,  #2ce8e8);
            color: white; text-align: right; margin-left: auto;
            padding: 8px 12px; border-radius: 20px; max-width: 80%; width: fit-content; margin-bottom:10px; word-wrap: break-word;
        }}
        .message.bot {{
            background: linear-gradient(to right, #9a03ff,  #117aca);
            color: white; align-self: flex-start; text-align: left;
            padding: 8px 12px; border-radius: 20px; max-width: 80%; width: fit-content; margin-bottom:10px; word-wrap: break-word;
        }}

        [data-testid="stChatInput"] {{ /* From first block, enhanced by second */
            background-color: #ffffff;
            border-radius: 25px;
            padding: 0.2rem;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1); /* Slightly stronger shadow */
            width: 100%;
            bottom: 0;
        }}
        [data-testid="stChatInput"] input {{
            border: none !important; box-shadow: none !important; padding: 0.8rem 1rem !important;
        }}

        /* General Element Styling (From first block) */
        .stButton>button {{
            border-radius: 6px; padding: 0.6rem 1.2rem;
            font-weight: 500; transition: transform 0.1s ease-out;
        }}
         .stButton>button:active {{ transform: scale(0.98); }}
        .stTextInput input, .stSelectbox div[data-baseweb="select"] > div, .stSlider div[data-testid="stTickBar"] {{
             border-radius: 6px !important;
        }}
        h1, h2, h3, h4, h5, h6 {{ color: #2c3e50; font-weight: 600;}}
        .section-header {{
            color: #1a5a96; margin-bottom: 1rem;
            padding-bottom: 0.5rem; border-bottom: 2px solid #e0e0e0;
        }}
        .card {{
            background-color: #ffffff; padding: 1.5rem; border-radius: 8px;
            box-shadow: 0 3px 10px rgba(0,0,0,0.07); margin-bottom: 1.5rem;
        }}
        .disease-card-home {{
            background-color: #ffffff; padding: 1.5rem; border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.06); text-align: center;
            transition: transform 0.2s ease, box-shadow 0.2s ease; margin-bottom: 1rem; height: 100%;
        }}
        .disease-card-home:hover {{ transform: translateY(-4px); box-shadow: 0 6px 15px rgba(0,0,0,0.1); }}
        .disease-card-home .emoji {{ font-size: 2.8rem; margin-bottom: 0.7rem; }}
        .disease-card-home h5 {{ color: #1a5a96; margin-bottom: 0.3rem; font-weight: 600; }}

        /* Radio button styling from second block */
        input[type="radio"]:checked + div > div {{
            color: #10b981 !important; font-weight: bold; font-size: 1.1em !important; /* Adjusted size */
        }}
        input[type="radio"]:not(:checked) + div > div {{
            color: #333 !important; /* Adjusted for better visibility on light bg if needed */
            font-weight: normal; font-size: 1em !important; /* Adjusted size */
        }}
        /* Responsive for topnav (from second block) */
        @media screen and (max-width: 768px) {{ /* Adjusted breakpoint */
            .topnav {{
                padding: 0 1rem; /* Reduced padding */
                /* height: auto; Allow height to adjust if items wrap, or set fixed smaller height */
            }}
            .topnav h1 {{
                font-size: 1.5rem; /* Smaller font for h1 */
            }}
            .topnav a {{
                padding: 0.6rem 0.8rem; /* Smaller padding for links */
                font-size: 0.85rem; /* Smaller font for links */
                margin-left: 5px; /* Reduced margin */
            }}
        }}
         @media screen and (max-width: 480px) {{
             .topnav h1 {{ font-size: 1.2rem; }}
         }}
    </style>
""", unsafe_allow_html=True)



# --- Helper Functions ---
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def load_chat_history_from_file():
    if os.path.exists("chat_history.json"):
        try:
            with open("chat_history.json", "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return [("bot", "Welcome to IntelliMed AI Chat! (Previous history was corrupted)")]
    return [("bot", "Welcome to IntelliMed AI Chat! How can I assist you today?")]

def save_chat_history_to_file(chat_history_data):
    with open("chat_history.json", "w", encoding="utf-8") as f:
        json.dump(chat_history_data, f)

# def get_gemini_chat_response(message_text):
#     # Using the general purpose model for chat
#     try:
#         response = gemini_flash_lite_model.generate_content(message_text)
#         return response.text
#     except Exception as e:
#         st.error(f"Error getting response from Gemini: {e}")
#         return "Sorry, I encountered an error trying to respond."



def get_gemini_chat_response(message_text, history=None):
    """
    Call Gemini with full chat context (only for medical queries).
    """
    try:
        if history is None:
            history = []

        # Build contextual prompt
        prompt = "You are a helpful and knowledgeable medical assistant. Answer only health, disease, and diagnosis-related queries. If the question is unrelated, politely refuse.\n\n"

        for role, msg in history:
            if role == "user":
                prompt += f"User: {msg}\n"
            else:
                prompt += f"Assistant: {msg}\n"

        prompt += f"User: {message_text}\nAssistant:"

        response = gemini_flash_lite_model.generate_content(prompt)
        return response.text

    except Exception as e:
        st.error(f"Error getting response from Gemini: {e}")
        return "Sorry, I encountered an error trying to respond."



# For image-based diagnosis using Gemini Pro Vision
def get_gemini_image_diagnosis_response(prompt_list_with_image):
    try:
        response = gemini_image_model.generate_content(prompt_list_with_image)
        return response.text
    except Exception as e:
        st.error(f"Error getting response from Gemini Vision: {e}")
        return "Sorry, I encountered an error processing the image."

def convert_html_to_pdf_bytes(source_html_content):
    pdf_io = io.BytesIO()
    pisa_status = pisa.CreatePDF(io.StringIO(source_html_content), dest=pdf_io)
    if pisa_status.err:
        st.error(f"PDF Conversion Error: {pisa_status.err}")
        return None
    pdf_bytes_content = pdf_io.getvalue()
    pdf_io.close()
    return pdf_bytes_content

def preprocess_brain_stroke_image(image_obj: Image.Image):
    image_obj = image_obj.resize((224, 224)).convert('RGB')
    image_array_data = np.array(image_obj) / 255.0
    image_array_data = np.expand_dims(image_array_data, axis=0)
    return image_array_data

# --- Page Rendering Functions ---

def footer():
    st.markdown("""
                <style>
            .footer {
                text-align: center;
                padding: 15px;
                margin-top: 50px;
                margin-bottom: 0px;
                font-size: 14px;
                color: #cccccc;
                background-color: #1a5a96;
                box-shadow: 0 -1px 5px rgba(0, 0, 0, 0.1);
                border-top: 1px solid #444;
                border-radius: 4px;
            }
            .footer a {
                color: #999999;
                text-decoration: none;
            }
            .footer a:hover {
                color: #ffffff;
                text-decoration: underline;
            }
        </style>
        <div class="footer">
            🤖 <strong style="color:#ffffff;">MedDio</strong> — Revolutionizing Healthcare with Artificial Intelligence 🧠<br>
            📍 Built with ❤️ using Streamlit | © 2025 MedDio<br>
            🔗 <a href="https://github.com/subhash-kr0/MedDio">GitHub</a> |
            💼 <a href="https://linkedin.com/in/subhash-kr0">LinkedIn</a>
        </div>
    """, unsafe_allow_html=True)


def render_home_page():
    st.title(APP_NAME)
    st.subheader("Your Smart AI Medical Diagnosis Assistant")
    st.markdown(f"""
    MedDio (Version {APP_VERSION}) leverages machine learning for preliminary insights into health conditions.
    Analyze data with trained models to predict disease likelihood.
    **Navigate to "Diagnose"** for prediction forms, or use our **"Chatbot"** for health inquiries.

    **Disclaimer:** This platform is for educational and informational purposes only.
    **Always consult with a qualified healthcare professional for definitive medical advice and diagnosis.**
    """)
    st.markdown("---")
    st.markdown("<h3 class='section-header'>🧬 Diseases Covered for AI Assessment</h3>", unsafe_allow_html=True)

    disease_names_list = list(MODEL_MAPPING.keys())
    # Consider adding emojis to MODEL_MAPPING for more robust association
    # emojis = ["🩸", "🩺", "❤️", "📈", "🎗️", "🫁", "🌿", "🦋", "🧠", "🔬"] 
    emojis = ["🩺", "🩸", "⚕️", "❤️", "💢", "🎀", "🫁", "🏥", "🦋"]


    num_disease_cols = 3 if len(disease_names_list) >= 3 else len(disease_names_list)
    if num_disease_cols == 0: num_disease_cols = 1 
    
    disease_cols = st.columns(num_disease_cols)
    for i, disease_key in enumerate(disease_names_list):
        disease_info = MODEL_MAPPING[disease_key]
        disease_name_home = disease_info["name"]
        with disease_cols[i % num_disease_cols]:
            emoji_char = emojis[i % len(emojis)]
            st.markdown(f"""
            <div class="disease-card-home">
                <div class="emoji">{emoji_char}</div>
                <h5>{disease_name_home}</h5>
            </div>
            """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<h3 class='section-header'>🚀 Technologies Used</h3>", unsafe_allow_html=True)
    st.markdown("""
    - **Backend & ML:** Python, Scikit-learn, TensorFlow/Keras, Joblib, Pandas, NumPy
    - **Frontend:** Streamlit
    - **AI Chatbot Integrations:** Google Generative AI (Gemini), HuggingFace (Mistral via LangChain)
    - **Styling:** Custom HTML/CSS
    """)
    st.markdown("---")
    st.markdown("<p style='text-align:center;'>Developed with 💚 by Tripti, Soumya, Subhash Kumar</p>", unsafe_allow_html=True)
    footer()  # Call footer function to render the footer


def prediction_form(disease_name_form, features_list, model_obj, scaler_obj=None):
    st.subheader(f"🩺 {disease_name_form} Prediction Form")

    with st.form(f"{disease_name_form.lower().replace(' ', '_')}_form", clear_on_submit=False):
        st.markdown("#### Fill the patient details:")
        form_cols = st.columns(3)
        inputs_dict = {}

        for i, feature_item in enumerate(features_list):
            with form_cols[i % 3]:
                f_key = feature_item.strip() # Ensure keys are clean
                # Use a more specific key for display if feature names are generic (e.g. 'Age_Diabetes')
                display_name = f"{f_key} ({disease_name_form})" if f_key in ['Age', 'BMI', 'Gender', 'Sex'] and len(MODEL_MAPPING)>1 else f_key
                
                if f_key in FEATURE_INPUTS: # Check against the master FEATURE_INPUTS
                    ftype, *params = FEATURE_INPUTS[f_key]
                    
                    # Use full name for display if available
                    label_text = feature_fullforms.get(disease_name_form, {}).get(f_key, f_key)

                    if ftype == 'slider':
                        min_val, max_val, step = params
                        default_val = float(min_val) if isinstance(step, float) or isinstance(min_val, float) else float(min_val)
                        inputs_dict[f_key] = st.slider(label_text, float(min_val), float(max_val), default_val, step=float(step) if isinstance(step,float) else float(step))
                    elif ftype == 'select':
                        inputs_dict[f_key] = st.selectbox(label_text, params[0], key=f"{f_key}_{disease_name_form}")
                else:
                    inputs_dict[f_key] = st.text_input(f_key, key=f"{f_key}_{disease_name_form}_text_fallback") # Fallback with unique key

        submitted_form = st.form_submit_button("🔍 Predict")
        


    if submitted_form:
        def convert_categorical_to_numeric(val):
            if isinstance(val, str):
                val_lower = val.lower()
                if val_lower in ['yes', 'present', 'male', 'normal', 'Male', 'Yes', 'Normal']: return 1 # Expanded for case variations
                if val_lower in ['no', 'notpresent', 'female', 'abnormal', 'Female', 'No', 'Abnormal']: return 0
            return val

        try:
            # Ensure order matches the model's expected feature order
            input_values_ordered = [convert_categorical_to_numeric(inputs_dict[feat.strip()]) for feat in features_list]
            
            data_df = pd.DataFrame([input_values_ordered], columns=[feat.strip() for feat in features_list])

            if scaler_obj:
                data_scaled = scaler_obj.transform(data_df)
                prediction_data = data_scaled
            else:
                prediction_data = data_df

            prediction_result_val = model_obj.predict(prediction_data)[0]
            
            confidence_val = 0.0
            if hasattr(model_obj, 'predict_proba'):
                proba = model_obj.predict_proba(prediction_data)[0]
                confidence_val = proba[1] if prediction_result_val == 1 else proba[0] # Confidence in the predicted class
            
            diagnosis_str = "Positive" if prediction_result_val == 1 else "Negative"

            st.session_state['prediction_result_data'] = {
                'title': disease_name_form,
                'inputs': inputs_dict, # Store user's original inputs
                'processed_inputs': dict(zip([feat.strip() for feat in features_list], input_values_ordered)), # Store processed inputs
                'diagnosis': diagnosis_str,
                'confidence': f"{confidence_val:.2%}"
            }
            st.success(f"Diagnosis for {disease_name_form}: {diagnosis_str} (Confidence: {st.session_state['prediction_result_data']['confidence']})")
            st.balloons()
            show_medical_report_pdf() # Display report section

        except KeyError as e:
            st.error(f"Input error: A feature value for '{e}' was expected but not found. Please ensure all form fields are correctly defined and filled.")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")



def show_medical_report_pdf():
    result_data = st.session_state.get('prediction_result_data', {})
    if not result_data:
        st.error("No diagnosis data found. Please submit the prediction form first.")
        return

    # Display in Streamlit
    st.subheader("🏥 Medical Diagnosis Report")
    st.write(f"Diagnosed by **{APP_NAME}**")
    st.markdown("---")
    st.markdown(f"#### 🧾 Disease: {result_data['title']}")
    st.markdown(f"#### Diagnosis Result: {' Negative' if result_data['diagnosis'] == 'Negative' else ' Positive'}")
    st.markdown(f"#### Confidence Score: {result_data['confidence']}")
    st.markdown("---")

    st.markdown("### Patient Provided Details:")
    for k, v in result_data['inputs'].items():
        label_k = feature_fullforms.get(result_data['title'], {}).get(k, k)
        st.write(f"**{label_k}**: {v}")
    st.markdown("---")

    # PDF generation
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=30)

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='CenterTitle', alignment=1, fontSize=16, spaceAfter=12))
    styles.add(ParagraphStyle(name='SectionHeader', fontSize=12, spaceAfter=6, fontName='Helvetica-Bold'))
    styles.add(ParagraphStyle(name='SmallItalic', fontSize=8, italic=True))

    story = []

    # Header
    story.append(Paragraph(f"<b>{APP_NAME}</b>", styles['Title']))
    story.append(Paragraph(f"Medical Diagnostic AI System (v{APP_VERSION})", styles['Normal']))
    story.append(Paragraph(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(HRFlowable(width="100%", color=colors.grey))
    story.append(Spacer(1, 12))

    # Report Title
    story.append(Paragraph("Medical Diagnosis Report", styles['CenterTitle']))
    story.append(Spacer(1, 6))

    # Section: Diagnosis Summary
    story.append(Paragraph("Diagnosis Summary", styles['SectionHeader']))
    summary_data = [
        ["Disease Assessed:", result_data['title']],
        ["Diagnosis Result:", result_data['diagnosis']],
        ["Confidence Score:", result_data['confidence']]
    ]
    table = Table(summary_data, colWidths=[130, 300])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.whitesmoke),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 0.25, colors.lightgrey)
    ]))
    story.append(table)
    story.append(Spacer(1, 12))

    # Section: Patient Provided Details
    story.append(Paragraph("Patient Provided Details", styles['SectionHeader']))
    input_data = []
    for k, v in result_data['inputs'].items():
        label_k_pdf = feature_fullforms.get(result_data['title'], {}).get(k, k)
        input_data.append([label_k_pdf, str(v)])
    
    input_table = Table(input_data, colWidths=[180, 250])
    input_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 0.25, colors.grey),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4)
    ]))
    story.append(input_table)
    story.append(Spacer(1, 14))

    # Disclaimer
    disclaimer = ("Disclaimer: This report is generated by an AI system and is intended for informational purposes only. "
                  "It is not a substitute for professional medical advice, diagnosis, or treatment. "
                  "Always consult a qualified medical professional for any medical concerns.")
    story.append(Paragraph(disclaimer, styles['SmallItalic']))
    story.append(Spacer(1, 10))

    # Build PDF
    try:
        doc.build(story)
        buffer.seek(0)
        st.download_button(
            label="📥 Download Report as PDF",
            data=buffer,
            file_name=f"{result_data['title'].lower().replace(' ','_')}_diagnostic_report.pdf",
            mime="application/pdf"
        )
    except Exception as e:
        st.error(f"Could not generate PDF: {e}")





def get_selected_symptoms_from_checkboxes(all_symptoms_list, column_layout):
    selected = []
    for i, symptom in enumerate(all_symptoms_list):
        current_col = column_layout[i % len(column_layout)]
        with current_col:
            if st.checkbox(symptom.capitalize(), key=f"symptom_check_{symptom.replace(' ','_')}_{i}"):
                selected.append(symptom)
    return selected

# --- Main Application Logic ---
query_params = st.query_params
current_page = query_params.get("page", "PAGE_HOME") # Get first element if list

# --- Top Navigation Bar (Simple HTML version) ---
st.markdown(f"""
<div class="topnav">
    <h1>{APP_NAME}</h1>
    <a href="?page={PAGE_HOME}" class="{'active' if current_page == PAGE_HOME else ''}">Home</a>
    <a href="?page={PAGE_DIAGNOSE}" class="{'active' if current_page == PAGE_DIAGNOSE else ''}">Diagnose</a>
    <a href="?page={PAGE_CHATBOT}" class="{'active' if current_page == PAGE_CHATBOT else ''}">Chatbot</a>
</div>
""", unsafe_allow_html=True)

# --- Page Content Area ---
# This div is for content that should be below the fixed topnav.
# Apply padding-top matching NAV_HEIGHT_PX + buffer
st.markdown(f"<div style='padding-top: {NAV_HEIGHT_PX + 20}px;'>", unsafe_allow_html=True)


if current_page == PAGE_HOME:
    render_home_page()
    # Sidebar content specific to home page
    st.sidebar.image("./static/logo.png", width=180, caption=f"{APP_NAME} {APP_VERSION}")
    st.sidebar.markdown("---")
    # greeting in sidebar
    st.sidebar.markdown("#### 👋 Welcome to MedDio")
    st.sidebar.markdown("Your AI-powered medical diagnosis assistant.")
    st.sidebar.markdown("---")
    st.sidebar.markdown("#### 📚 How to Use")
    st.sidebar.markdown("""
    1. **Diagnose**: Navigate to the "Diagnose" page to fill out forms for various diseases.
    2. **Chatbot**: Use the "Chatbot" page to interact with our AI assistant for health queries.
    3. **Medical Report**: After diagnosis, download your personalized medical report.
    4. **Feedback**: Share your experience or report issues via our GitHub repository.
    """)
    st.balloons() 

elif current_page == PAGE_CHATBOT:
    st.sidebar.image("./static/logo.png", width=150)
    st.sidebar.title("📋 Chat Options")
    
    # Initialize chat history if it doesn't exist
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = load_chat_history_from_file()

    selected_chat_model = st.sidebar.selectbox("Choose AI Chat Model", [CHATBOT_GEMINI, CHATBOT_MISTRAL], key="chatbot_model_select")

    if selected_chat_model == CHATBOT_GEMINI:
        # st.title("💬 IntelliMed AI Chat (Gemini)")

        st.subheader("💬 AI Chat Assistant (Gemini - Google Generative AI)")
        
        # Display chat messages from history
        chat_display_html = "<div class='chat-box'>" # Using the class from consolidated CSS
        for role, msg_text in st.session_state.chat_history:
            icon = "🧑‍💻" if role == "user" else "🤖"
            # Using .message .user and .message .bot classes
            chat_display_html += f"<div class='message {role}'>{icon} {msg_text}</div>"
        chat_display_html += "</div>"
        st.markdown(chat_display_html, unsafe_allow_html=True)

        user_chat_input = st.chat_input("Type your health query...", key="gemini_user_input")
        if user_chat_input:
            st.session_state.chat_history.append(("user", user_chat_input))
            with st.spinner("🤖 Gemini is thinking..."):
                bot_reply_text = get_gemini_chat_response(user_chat_input)
            st.session_state.chat_history.append(("bot", bot_reply_text))
            save_chat_history_to_file(st.session_state.chat_history)
            st.rerun()

    elif selected_chat_model == CHATBOT_MISTRAL:
        st.title("💬 AI Chat Assistant (Mistral - LangChain)")
        try:
            asyncio.get_running_loop()
        except RuntimeError: # Allow running in Streamlit's non-async thread
            asyncio.set_event_loop(asyncio.new_event_loop())

        DB_FAISS_PATH = "vectorstore/db_faiss" # Ensure this path is correct

        with st.sidebar:
            st.header("Mistral Settings ⚙️")
            HUGGINGFACE_REPO_ID = st.selectbox(
                "Select Mistral Model Variant",
                ["mistralai/Mistral-7B-Instruct-v0.3", "HuggingFaceH4/zephyr-7b-beta"],
                index=0, key="mistral_variant"
            )
            response_temp = st.slider("Response Temperature", 0.0, 1.0, 0.5, 0.1, key="mistral_temp")
            st.markdown("---")
            st.info("🤖 **Mistral Chatbot** powered by **LangChain** & **HuggingFace**.")


        @st.cache_resource # Cache the vectorstore resource
        def get_mistral_vectorstore():
            if not os.path.exists(DB_FAISS_PATH):
                st.error(f"❌ FAISS vector store not found at {DB_FAISS_PATH}. Please ensure it's created and accessible.")
                return None
            try:
                embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
                vectorstore = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
                return vectorstore
            except Exception as e:
                st.error(f"❌ Error loading vector store: {e}")
                return None

        def set_mistral_custom_prompt():
            return PromptTemplate(
                template="""
                You are an advanced AI assistant specializing in **medical diagnosis and healthcare insights**.
                Use the provided **context** to deliver accurate, relevant, and professional responses.
                Context (Medical Data / Symptoms / Reports): {context}
                User Query (Health Concern / Diagnosis Request): {question}
                AI Medical Response: (Start directly)
                """,
                input_variables=["context", "question"]
            )

        @st.cache_resource # Cache the LLM resource (depends on repo_id and temp)
        def load_mistral_llm(_repo_id, _hf_token, _temperature): 
            if not _hf_token:
                st.error("❌ HuggingFace token (HF_TOKEN) not found in environment variables.")
                return None
            try:
                return HuggingFaceEndpoint(
                    repo_id=_repo_id,
                    temperature=_temperature,
                    huggingfacehub_api_token=_hf_token,
                )
            except Exception as e:
                st.error(f"❌ Error loading LLM from HuggingFace: {e}")
                return None

        def run_mistral_chat():
            if "mistral_messages" not in st.session_state:
                st.session_state.mistral_messages = []

            for message in st.session_state.mistral_messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            prompt_input = st.chat_input("Ask Mistral about health...", key="mistral_user_input")
            if prompt_input:
                st.chat_message("user").markdown(prompt_input)
                st.session_state.mistral_messages.append({"role": "user", "content": prompt_input})

                HF_TOKEN = os.getenv("HF_TOKEN")
                vectorstore_db = get_mistral_vectorstore()
                llm_model_mistral = load_mistral_llm(HUGGINGFACE_REPO_ID, HF_TOKEN, response_temp)

                if vectorstore_db is None or llm_model_mistral is None:
                    st.warning("Required components for Mistral chat are not loaded. Cannot proceed.")
                    return

                try:
                    with st.spinner("🤖 Mistral is thinking..."):
                        qa_chain = RetrievalQA.from_chain_type(
                            llm=llm_model_mistral,
                            chain_type="stuff", # suitable for smaller contexts
                            retriever=vectorstore_db.as_retriever(search_kwargs={'k': 2}), # retrieve 2 documents
                            return_source_documents=True,
                            chain_type_kwargs={'prompt': set_mistral_custom_prompt()}
                        )
                        response_data = qa_chain.invoke({'query': prompt_input})
                        answer_text = response_data.get("result", "I'm sorry, I couldn't process that.")
                        # sources_docs = response_data.get("source_documents", [])

                    with st.chat_message("assistant"):
                        st.markdown(answer_text)

                    st.session_state.mistral_messages.append({"role": "assistant", "content": answer_text})
                    # No need to save Mistral history to file unless specified
                except Exception as e:
                    st.error(f"❌ Error during Mistral QA chain execution: {e}")

        run_mistral_chat()


elif current_page == PAGE_DIAGNOSE:
    st.title("🔬 AI-Powered Medical Diagnosis")
    st.sidebar.image("./static/logo.png", width=150)
    # st.sidebar.title("🩺 Diagnosis Options")

    diagnose_mode = st.radio(
        "Select Diagnosis Method:",
        (DIAGNOSE_MODE_FORM, DIAGNOSE_MODE_IMAGE),
        key="diagnose_method_radio",
        horizontal=True
    ) 
    # st.title("🩺 Diagnosis Options")
    # st.subheader("Choose a diagnosis method to proceed:")
    st.subheader(f"{diagnose_mode} Tool")


    if diagnose_mode == DIAGNOSE_MODE_FORM:
        available_diseases = list(MODEL_MAPPING.keys())
        model_choice_form = st.selectbox(
            "Select Disease Model (Form-Based):", 
            available_diseases, 
            index=0, 
            key="form_disease_choice"
        )

        if model_choice_form == "Symptom Checker":
            st.subheader("Symptom-Based General Diagnosis Aid")
            st.write("Select your symptoms from the list below to get a preliminary idea. This tool considers common illnesses based on typical symptoms.")
            
            all_symptoms_list = sorted(set(symptom for sublist in diseases_data.values() for symptom in sublist))
            checkbox_cols = st.columns(3)
            selected_symptoms_list = get_selected_symptoms_from_checkboxes(all_symptoms_list, checkbox_cols)

            if st.button("Diagnose Symptoms", key="diagnose_symptoms_button"):
                if not selected_symptoms_list:
                    st.warning("Please select at least one symptom.")
                else:
                    disease_scores = {}
                    for disease, symptoms_for_disease in diseases_data.items():
                        matches = [sym for sym in selected_symptoms_list if sym in symptoms_for_disease]
                        score = len(matches)
                        if score > 0:
                            disease_scores[disease] = score
                    
                    if not disease_scores:
                        st.info("No matching common diseases found for the selected symptoms. Consider consulting a doctor or trying a specific disease model if you have more information.")
                    else:
                        st.markdown("### Symptom Analysis Results:")
                        sorted_diseases_by_score = sorted(disease_scores.items(), key=lambda item: item[1], reverse=True)
                        for disease, score in sorted_diseases_by_score:
                            total_symptoms_for_disease = len(diseases_data[disease])
                            confidence_percent = int((score / total_symptoms_for_disease) * 100) if total_symptoms_for_disease > 0 else 0
                            st.write(f"**{disease}**: Potential match (Confidence: ~{confidence_percent}%) - Matched {score}/{total_symptoms_for_disease} key symptoms.")
                        st.caption("Note: This is a basic symptom matcher, not a formal diagnosis. Consult a healthcare professional.")
        
        elif model_choice_form in MODEL_MAPPING:
            selected_model_info = MODEL_MAPPING[model_choice_form]
            prediction_form(
                selected_model_info["name"],
                selected_model_info["features"],
                selected_model_info["model"],
                selected_model_info.get("scaler") # Use .get for scaler as it might be None
            )
        
        # Show feature full forms in sidebar for the selected model
        st.sidebar.markdown("---")
        st.sidebar.markdown("#### 🔍 Feature Information")
        current_features_info = feature_fullforms.get(model_choice_form, {})
        if current_features_info:
            for feature_name, full_form in current_features_info.items():
                st.sidebar.markdown(f"**{feature_name}**: {full_form}")
        elif model_choice_form != "Choose Disease":
            st.sidebar.info(f"No detailed feature information available for {model_choice_form} in the list.")

        st.markdown("---")


    elif diagnose_mode == DIAGNOSE_MODE_IMAGE:
        image_ai_model_choice = st.selectbox(
            "Select AI Model (Image-Based):", 
            [IMAGE_AI_GEMINI, IMAGE_AI_BRAIN_STROKE], 
            index=0, 
            key="image_ai_choice"
        )
        st.subheader(f"{image_ai_model_choice} - Image Analysis")


        if image_ai_model_choice == IMAGE_AI_GEMINI:
            st.markdown("AI-powered system to analyze medical images (e.g., Brain MRI) using Google Gemini Vision.")

            #feature info in sidebar
            st.sidebar.markdown("---")
            st.sidebar.markdown("#### 🔍 Gemini Vision Feature Information")
            gemini_feature_info = st.sidebar.markdown({
                "Patient Name": "Full name of the patient (optional)",
                "Age": "Age of the patient (optional, for context)",
                "Symptoms / Observations": "Reported symptoms or clinical context to assist diagnosis",
                "Medical Image": "Upload a medical image (e.g., Brain MRI) for analysis",
            })
            

            
            with st.form("gemini_image_patient_form"):
                patient_name_img = st.text_input("👤 Patient Name (Optional)")
                patient_age_img = st.number_input("🎂 Age (Optional)", min_value=0, max_value=120, step=1)
                symptoms_img = st.text_area("💬 Symptoms / Observations / Clinical Context")
                uploaded_mri_image = st.file_uploader("📤 Upload Medical Image (e.g., Brain MRI)", type=["jpg", "jpeg", "png"], key="gemini_img_upload")
                custom_prompt_img = st.text_area("🔬 Specific question for the AI regarding the image (e.g., 'Look for signs of tumor')", 
                                                 value="Analyze the provided medical image. Identify and describe any visible abnormalities or signs of disease. " \
                                                 "List possible diagnoses based on the image. Suggest further clinical steps if appropriate. Provide a concise summary.")
                submit_gemini_img = st.form_submit_button("🧪 Diagnose with Gemini Vision")

            if submit_gemini_img and uploaded_mri_image and custom_prompt_img:
                try:
                    pil_image = Image.open(uploaded_mri_image)
                    st.image(pil_image, caption="Uploaded Medical Image", use_container_width=True, width=300)

                    prompt_details = f"""
                    You are a specialized medical expert AI assisting in analyzing medical images.
                    Patient Details (if provided):
                    - Name: {patient_name_img if patient_name_img else "Not Provided"}
                    - Age: {patient_age_img if patient_age_img else "Not Provided"}
                    - Reported Symptoms/Context: {symptoms_img if symptoms_img else "Not Provided"}
                    - Image Attached.
                    - Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

                    User's Specific Question: {custom_prompt_img}

                    Instruction: Based on the image and the context, provide a detailed analysis. Structure your response clearly. If making potential diagnostic 
                    observations, state them cautiously and recommend clinical correlation and further professional review.
                    """
                    with st.spinner("🤖 Gemini Vision is analyzing the image..."):
                        gemini_diagnosis_text = get_gemini_image_diagnosis_response([prompt_details, pil_image])
                    
                    st.markdown("### 📝 Gemini Vision - Diagnostic Report")
                    st.markdown(gemini_diagnosis_text) # Gemini response is usually markdown-friendly

                    # PDF Download
                    report_html = markdown2.markdown(f"# IntelliMed Diagnostic Report (Gemini Vision)\n\n{gemini_diagnosis_text}")
                    pdf_report_bytes = convert_html_to_pdf_bytes(report_html)
                    if pdf_report_bytes:
                        st.download_button(
                            label="📥 Download Gemini Report as PDF",
                            data=pdf_report_bytes,
                            file_name="gemini_vision_report.pdf",
                            mime="application/pdf"
                        )
                except Exception as e:
                    st.error(f"❌ Error during Gemini Vision diagnosis: {e}")
            elif submit_gemini_img and (not uploaded_mri_image or not custom_prompt_img):
                st.warning("⚠️ Please upload an image and ensure the question/prompt field is filled.")


        elif image_ai_model_choice == IMAGE_AI_BRAIN_STROKE:
            st.markdown("AI-powered tool for **Brain Stroke** detection using brain scan images (trained model).")
            
            patient_name_stroke = st.text_input("👤 Patient Full Name", key="stroke_name")
            patient_age_stroke = st.number_input("🎂 Patient Age", min_value=1, max_value=120, step=1, key="stroke_age")
            uploaded_scan_file = st.file_uploader("📤 Upload Brain Scan Image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"], key="stroke_img_upload")

            if uploaded_scan_file and patient_name_stroke and patient_age_stroke:
                pil_stroke_image = Image.open(uploaded_scan_file)
                st.image(pil_stroke_image, caption='🖼️ Uploaded Brain Scan', use_container_width=False, width=300)

                if st.button("🧪 Analyze for Stroke and Generate Report", key="analyze_stroke_button"):
                    try:
                        with st.spinner("🧠 Analyzing image for stroke..."):
                            preprocessed_img_array = preprocess_brain_stroke_image(pil_stroke_image)
                            stroke_prediction = brainStrokemodel.predict(preprocessed_img_array)
                        
                        # Assuming model outputs probability for stroke class (class 1)
                        stroke_probability = stroke_prediction[0][0] # Adjust if model output is different
                        diagnosis_text_stroke = f"<span style='color:red; font-weight:bold;'>Stroke Detected (Probability: {stroke_probability:.2%})</span>" if stroke_probability > 0.5 else f"<span style='color:green; font-weight:bold;'>No Stroke Detected (Probability of No Stroke: {1-stroke_probability:.2%})</span>"
                        report_time_stroke = datetime.now().strftime("%A, %d %B %Y | %I:%M %p")

                        st.markdown("---")
                        st.markdown("<h3 style='text-align: center; color: #34495E;'>📄 Brain Stroke Diagnostic Report</h3>", unsafe_allow_html=True)
                        report_content_html = f"""
                        <div style='border: 1px solid #ddd; border-radius: 10px; padding: 20px; background-color: #FAFAFA; font-family: Arial, sans-serif;'>
                            <p><strong>🧾 Patient Name:</strong> {patient_name_stroke}</p>
                            <p><strong>🎂 Age:</strong> {patient_age_stroke} years</p>
                            <p><strong>🕒 Report Generated:</strong> {report_time_stroke}</p>
                            <hr style="margin: 10px 0;">
                            <p><strong>🔍 Diagnosis Result (AI Model):</strong> {diagnosis_text_stroke}</p>
                            <hr style="margin: 10px 0;">
                            <p style='font-size:0.8em; color: #555;'><i>This is an AI-generated analysis and should be confirmed by a qualified medical professional. This model is specific for brain stroke detection based on its training data.</i></p>
                        </div>
                        """
                        st.markdown(report_content_html, unsafe_allow_html=True)
                        st.success("✅ Stroke analysis report generated successfully!")

                        confidence_value = stroke_probability if stroke_probability > 0.5 else (1 - stroke_probability)
                        confidence_text = f"{confidence_value:.2%}"

                        # PDF Download for Stroke Report
                        pdf_stroke_html = f"""
                        <h1>IntelliMed - Brain Stroke Diagnostic Report</h1>
                        <p><strong>Patient Name:</strong> {patient_name_stroke}</p>
                        <p><strong>Age:</strong> {patient_age_stroke} years</p>
                        <p><strong>Report Generated:</strong> {report_time_stroke}</p>
                        <hr/>
                        <p><strong>Diagnosis Result (AI Model):</strong> {'Stroke Detected' if stroke_probability > 0.5 else 'No Stroke Detected'}</p>
                        <p><strong>Confidence/Probability:</strong> {confidence_text}</p>                        <hr/>
                        <p><small><i>This is an AI-generated analysis and should be confirmed by a qualified medical professional.</i></small></p>
                        """
                        pdf_stroke_bytes = convert_html_to_pdf_bytes(pdf_stroke_html)
                        if pdf_stroke_bytes:
                            st.download_button(
                                label="📥 Download Stroke Report as PDF",
                                data=pdf_stroke_bytes,
                                file_name=f"brain_stroke_report_{patient_name_stroke.replace(' ','_')}.pdf",
                                mime="application/pdf"
                            )

                    except Exception as e:
                        st.error(f"❌ Error during stroke prediction: {e}")
            elif uploaded_scan_file and (not patient_name_stroke or not patient_age_stroke):
                st.warning("⚠️ Please fill in Patient Name and Age to generate the stroke report.")

    footer()  # Call footer function to render the footer



st.sidebar.markdown("---")
# Sidebar for developer contact info
st.sidebar.markdown("#### 👨‍💻 Contact Developer")

developer = st.sidebar.radio("", ["Gihub Repo","Tripti","Subhash Kumar", "Soumya Sahoo"], key="developer_contact_select")

# Display corresponding contact info
if developer == "Subhash Kumar":
    st.sidebar.markdown("""
        **👤 Name:** Subhash Kumar</br>  
        **✉️ Email:** subhashkr855@gmail.com</br>
        **💼 LinkedIn:** [subhash-kr0](https://linkedin.com/in/subhash-kr0)</br>
        **💻 GitHub:** [subhash-kro](https://github.com/subhash-kr0)  
    """, unsafe_allow_html=True)

elif developer == "Tripti":
    st.sidebar.markdown("""
        **👤 Name:** Tripti </br>
        **✉️ Email:** vermatripti547@gmail.com  
        **💼 LinkedIn:** [triptiverma310](https://linkedin.com/in/triptiverma310)</br>
        **💻 GitHub:** [Triptiverma003](https://github.com/Triptiverma003)  
    """, unsafe_allow_html=True)

elif developer == "Soumya Sahoo":
    st.sidebar.markdown("""
        **👤 Name:** Soumya</br>
        **✉️ Email:** soumyasahoo.2907@gmail.com</br>
        **💼 LinkedIn:** [0xsoumya](https://linkedin.com/in/0xsoumya)</br>
        **💻 GitHub:** [0XSoumya](https://github.com/0XSoumya)                         
    """, unsafe_allow_html=True)

elif developer == "Gihub Repo":
    st.sidebar.markdown("""
        **👤 GitHub Repository:**  
            [MedDio][https://github.com/subhash-kr0/MedDio](https://github.com/subhash-kr0/MedDio)
        """, unsafe_allow_html=True)
    
st.sidebar.info("This application is for educational and demonstrative purposes. It is not a substitute for professional medical advice.")

st.markdown("</div>", unsafe_allow_html=True)
