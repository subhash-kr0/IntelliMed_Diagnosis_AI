import streamlit as st
import joblib
import google.generativeai as genai
import numpy as np
import pandas as pd
import json
import os
import io
from fpdf import FPDF
import re # For sanitizing keys

# --- Global Configuration & Constants ---
NAV_HEIGHT_PX = 60
APP_NAME = "🩺 IntelliMed AI"
APP_VERSION = "1.2" # Updated version

# --- Page Configuration ---
st.set_page_config(
    layout='wide',
    page_icon='assets/favicon.png', # Assuming you have a favicon in assets
    page_title=APP_NAME,
    initial_sidebar_state='expanded'
)

# --- API Key and Gemini Configuration ---
GEMINI_CONFIGURED = False
gemini_llm_model = None
try:
    gemini_key = st.secrets.get("api_keys", {}).get("gemini")
    if not gemini_key:
        # This error will show on the main page if key is missing
        st.error("Gemini API key not found. Chatbot functionality will be limited.")
        pass # Allow app to run, chatbot page will handle this
    else:
        genai.configure(api_key=gemini_key)
        GEMINI_CONFIGURED = True
except Exception as e:
    # st.error(f"Error configuring Gemini: {e}")
    pass

# --- Model and Resource Loading ---
@st.cache_resource
def load_gemini_model_cached():
    if GEMINI_CONFIGURED:
        try:
            return genai.GenerativeModel(model_name="gemini-1.5-flash-latest")
        except Exception as e:
            # st.error(f"Could not load Gemini model: {e}") # Shows in console or first page load
            return None
    return None
gemini_llm_model = load_gemini_model_cached()

@st.cache_resource
def load_ml_model(path):
    if not os.path.exists(path):
        # st.error(f"Model file not found: {path}") # Shows in console or first page load
        return None
    try:
        return joblib.load(path)
    except Exception as e:
        # st.error(f"Error loading model {path}: {e}")
        return None

# Load ML models (paths need to be correct)
MODELS_BASE_PATH = './models/' # Define base path for models
diabetes_model = load_ml_model(os.path.join(MODELS_BASE_PATH, 'diabetes_model.pkl'))
diabetes_scaler = load_ml_model(os.path.join(MODELS_BASE_PATH, 'diabetes_scaler.pkl'))
kidney_model = load_ml_model(os.path.join(MODELS_BASE_PATH, 'kidneyDisease_model.pkl'))
heart_model = load_ml_model(os.path.join(MODELS_BASE_PATH, 'heartDisease_randomForest_model.pkl'))
heart_scaler = load_ml_model(os.path.join(MODELS_BASE_PATH, 'heartDisease_scaler.pkl'))
hypertension_model = load_ml_model(os.path.join(MODELS_BASE_PATH, 'hypertension_model.pkl'))
breast_model = load_ml_model(os.path.join(MODELS_BASE_PATH, 'breastCancer_randomForest_model.pkl'))
lung_model = load_ml_model(os.path.join(MODELS_BASE_PATH, 'lungCancer_XGBClassifier_model.pkl'))
liver_model = load_ml_model(os.path.join(MODELS_BASE_PATH, 'liverDisease_rf_model.pkl'))
thyroid_model = load_ml_model(os.path.join(MODELS_BASE_PATH, 'thyroid_cat_model.pkl'))

# --- Feature Lists & Input Definitions ---
# (Keep your comprehensive DIABETES_FEATURES, KIDNEY_FEATURES, etc. lists here)
DIABETES_FEATURES = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
KIDNEY_FEATURES = ['age', 'bp', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 'bu', 'sc', 'pot', 'wc', 'htn', 'dm', 'cad', 'pe', 'ane']
HEART_FEATURES = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach','exang','oldpeak','slope','ca','thal']
HYPERTENSION_FEATURES = ['Age', 'BMI', 'Systolic_BP','Diastolic_BP', 'Total_Cholesterol']
BREAST_FEATURES = ['mean_radius', 'mean_texture', 'mean_perimeter', 'mean_area', 'mean_smoothness', 'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean']
LUNG_FEATURES = ['GENDER', 'AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE', 'CHRONIC_DISEASE', 'FATIGUE', 'ALLERGY', 'WHEEZING', 'ALCOHOL_CONSUMING', 'COUGHING', 'SHORTNESS_OF_BREATH', 'SWALLOWING_DIFFICULTY', 'CHEST_PAIN']
LIVER_FEATURES = ['Age', 'Gender', 'Total_Bilirubin', 'Alkaline_phosphate', 'Alamine_Aminotransferace', 'Aspartate_Amino', 'Protien', 'Albumin', 'Albumin_Globulin_ratio']
THYROID_FEATURES = ['Age','Sex','On Thyroxine','Query on Thyroxine','On Antithyroid Meds','Is Sick','Is Pregnant','Had Thyroid Surgery','Had I131 Treatment','Query Hypothyroid','Query Hyperthyroid','On Lithium','Has Goitre','Has Tumor','Psych Condition','TSH Level','T3 Level','TT4 Level','T4U Level','FTI Level', 'TBG Level']

# FEATURE_INPUTS: Ensure all features from above lists are keys here with correct params
# (min_val, max_val, default_val, step_val) for sliders
# (options_list) for selectbox
FEATURE_INPUTS = {
    'Pregnancies': ('slider', 0, 20, 1, 1), 'Glucose': ('slider', 40, 300, 100, 1), 'BloodPressure': ('slider', 30, 180, 80, 1),
    'SkinThickness': ('slider', 0, 100, 20, 1), 'Insulin': ('slider', 0, 900, 150, 10),
    'BMI': ('slider', 10.0, 70.0, 25.0, 0.1), 'DiabetesPedigreeFunction': ('slider', 0.01, 2.5, 0.5, 0.01),
    'Age': ('slider', 1, 120, 30, 1), 'age': ('slider', 1, 100, 30, 1), 'AGE': ('slider', 10, 100, 30, 1),
    'bp': ('slider', 40, 180, 100, 1), 'al': ('slider', 0, 5, 1, 1), 'su': ('slider', 0, 5, 0, 1),
    'rbc': ('select', ['normal', 'abnormal']), 'pc': ('select', ['normal', 'abnormal']),
    'pcc': ('select', ['present', 'notpresent']), 'ba': ('select', ['present', 'notpresent']),
    'bgr': ('slider', 70, 500, 120, 1), 'bu': ('slider', 5, 200, 50, 1), 'sc': ('slider', 0.1, 20.0, 1.0, 0.1),
    'pot': ('slider', 2.5, 10.0, 4.0, 0.1), 'wc': ('slider', 1000, 25000, 8000, 100),
    'htn': ('select', ['yes', 'no']), 'dm': ('select', ['yes', 'no']), 'cad': ('select', ['yes', 'no']),
    'pe': ('select', ['yes', 'no']), 'ane': ('select', ['yes', 'no']),
    'sex': ('select', ['male', 'female']), 'Sex': ('select', ['Female', 'Male']),
    'cp': ('slider', 0, 3, 0, 1), 'trestbps': ('slider', 80, 200, 120, 1), 'chol': ('slider', 100, 600, 200, 1),
    'fbs': ('select', ['yes', 'no']), 'restecg': ('slider', 0, 2, 0, 1), 'thalach': ('slider', 60, 220, 150, 1),
    'exang': ('select', ['yes', 'no']), 'oldpeak': ('slider', 0.0, 6.0, 1.0, 0.1), 'slope': ('slider', 0, 2, 1, 1),
    'ca': ('slider', 0, 4, 0, 1), 'thal': ('slider', 0, 3, 2, 1),
    'Systolic_BP': ('slider', 80, 200, 120, 1), 'Diastolic_BP': ('slider', 40, 120, 80, 1),
    'Total_Cholesterol': ('slider', 100, 400, 200, 1),
    'mean_radius': ('slider', 5.0,30.0,15.0,0.1),'mean_texture':('slider',5.0,40.0,20.0,0.1),
    'mean_perimeter':('slider',30.0,200.0,100.0,0.1),'mean_area':('slider',100.0,2500.0,600.0,1.0),
    'mean_smoothness':('slider',0.05,0.2,0.1,0.001),'compactness_mean':('slider',0.01,1.0,0.1,0.01),
    'concavity_mean':('slider',0.01,1.0,0.1,0.01),'concave points_mean':('slider',0.01,0.5,0.05,0.01),
    'symmetry_mean':('slider',0.1,0.5,0.2,0.01),'fractal_dimension_mean':('slider',0.01,0.2,0.06,0.001),
    'GENDER': ('select', ['Male', 'Female']), 'SMOKING': ('select', ['Yes', 'No']),
    'YELLOW_FINGERS': ('select', ['Yes', 'No']), 'ANXIETY': ('select', ['Yes', 'No']),
    'PEER_PRESSURE': ('select', ['Yes', 'No']), 'CHRONIC_DISEASE': ('select', ['Yes', 'No']),
    'FATIGUE': ('select', ['Yes', 'No']), 'ALLERGY': ('select', ['Yes', 'No']),
    'WHEEZING': ('select', ['Yes', 'No']), 'ALCOHOL_CONSUMING': ('select', ['Yes', 'No']),
    'COUGHING': ('select', ['Yes', 'No']), 'SHORTNESS_OF_BREATH': ('select', ['Yes', 'No']),
    'SWALLOWING_DIFFICULTY': ('select', ['Yes', 'No']), 'CHEST_PAIN': ('select', ['Yes', 'No']),
    'Gender': ('select', ['Male', 'Female']), 'Total_Bilirubin': ('slider',0.1,10.0,1.0,0.1),
    'Alkaline_phosphate':('slider',50,3000,200,10),'Alamine_Aminotransferace':('slider',1,2000,50,10),
    'Aspartate_Amino':('slider',1,2000,50,10),'Protien':('slider',2.0,10.0,6.0,0.1),
    'Albumin':('slider',1.0,6.0,3.5,0.1),'Albumin_Globulin_ratio':('slider',0.1,3.0,1.0,0.1),
    'On Thyroxine':('select',['No','Yes']),'Query on Thyroxine':('select',['No','Yes']),
    'On Antithyroid Meds':('select',['No','Yes']),'Is Sick':('select',['No','Yes']),
    'Is Pregnant':('select',['No','Yes']),'Had Thyroid Surgery':('select',['No','Yes']),
    'Had I131 Treatment':('select',['No','Yes']),'Query Hypothyroid':('select',['No','Yes']),
    'Query Hyperthyroid':('select',['No','Yes']),'On Lithium':('select',['No','Yes']),
    'Has Goitre':('select',['No','Yes']),'Has Tumor':('select',['No','Yes']),
    'Psych Condition':('select',['No','Yes']),'TSH Level':('slider',0.01,20.0,2.0,0.1),
    'T3 Level':('slider',0.2,5.0,1.5,0.1),'TT4 Level':('slider',50,300,150,1),
    'T4U Level':('slider',0.3,1.5,0.9,0.01),'FTI Level':('slider',3,50,25.0,0.1),
    'TBG Level':('slider',10,50,25,1),
}
# Full forms for user-friendly labels. Populate this extensively.
feature_fullforms = {
    'Diabetes': {'Pregnancies': 'Number of Pregnancies', 'Glucose': 'Fasting Glucose (mg/dL)', 'Age': 'Age (Years)'},
    'Kidney Disease': {'age': 'Age (Years)', 'bp': 'Blood Pressure (mm Hg)', 'sg': 'Specific Gravity', 'al': 'Albumin (0-5 scale)', 'su': 'Sugar (0-5 scale)', 'rbc': 'Red Blood Cells', 'pc': 'Pus Cells', 'pcc': 'Pus Cell Clumps', 'ba': 'Bacteria', 'bgr': 'Blood Glucose Random (mg/dL)', 'bu': 'Blood Urea (mg/dL)', 'sc': 'Serum Creatinine (mg/dL)', 'sod': 'Sodium (mEq/L)', 'pot': 'Potassium (mEq/L)', 'hemo': 'Hemoglobin (gms)', 'pcv': 'Packed Cell Volume (%)', 'wc': 'White Blood Cell Count (cells/cumm)', 'rc': 'Red Blood Cell Count (millions/cmm)', 'htn': 'Hypertension', 'dm': 'Diabetes Mellitus', 'cad': 'Coronary Artery Disease', 'appet': 'Appetite (good/poor)', 'pe': 'Pedal Edema', 'ane': 'Anemia'},
    'Heart Disease': {'age': 'Age (Years)', 'sex': 'Sex (Male/Female)', 'cp': 'Chest Pain Type (0-3)', 'trestbps': 'Resting Blood Pressure (mm Hg)', 'chol': 'Serum Cholesterol (mg/dL)', 'fbs': 'Fasting Blood Sugar > 120 mg/dL', 'restecg': 'Resting ECG Results (0-2)', 'thalach': 'Max Heart Rate Achieved', 'exang': 'Exercise Induced Angina', 'oldpeak': 'ST Depression (by exercise)', 'slope': 'Slope of Peak Exercise ST Segment', 'ca': 'Major Vessels Colored by Flouroscopy (0-4)', 'thal': 'Thalassemia (0-3)'},
    'Hypertension': {'Age': 'Age (Years)', 'BMI': 'Body Mass Index (kg/m²)', 'Systolic_BP': 'Systolic Blood Pressure (mm Hg)', 'Diastolic_BP': 'Diastolic Blood Pressure (mm Hg)', 'Total_Cholesterol': 'Total Cholesterol (mg/dL)'},
    'Breast Cancer': {'mean_radius':'Mean Radius','mean_texture':'Mean Texture','mean_perimeter':'Mean Perimeter','mean_area':'Mean Area','mean_smoothness':'Mean Smoothness','compactness_mean':'Mean Compactness','concavity_mean':'Mean Concavity','concave points_mean':'Mean Concave Points','symmetry_mean':'Mean Symmetry','fractal_dimension_mean':'Mean Fractal Dimension'},
    'Lung Cancer': {'GENDER':'Gender','AGE':'Age (Years)','SMOKING':'Smoking Habit','YELLOW_FINGERS':'Yellow Fingers','ANXIETY':'Anxiety Reported','PEER_PRESSURE':'Peer Pressure Influence','CHRONIC_DISEASE':'Has Chronic Disease','FATIGUE':'Fatigue Reported','ALLERGY':'Allergy Reported','WHEEZING':'Wheezing Present','ALCOHOL_CONSUMING':'Alcohol Consumption','COUGHING':'Coughing Present','SHORTNESS_OF_BREATH':'Shortness of Breath','SWALLOWING_DIFFICULTY':'Swallowing Difficulty','CHEST_PAIN':'Chest Pain Reported'},
    'Liver Disease': {'Age':'Age (Years)','Gender':'Gender','Total_Bilirubin':'Total Bilirubin (mg/dL)','Alkaline_phosphate':'Alkaline Phosphatase (IU/L)','Alamine_Aminotransferace':'Alamine Aminotransferase (SGPT) (IU/L)','Aspartate_Amino':'Aspartate Aminotransferase (SGOT) (IU/L)','Protien':'Total Protein (g/dL)','Albumin':'Albumin (g/dL)','Albumin_Globulin_ratio':'Albumin/Globulin Ratio'},
    'Thyroid Disease': {'Age':'Age (Years)','Sex':'Sex','On Thyroxine':'On Thyroxine Medication','Query on Thyroxine':'Query on Thyroxine','On Antithyroid Meds':'On Antithyroid Medication','Is Sick':'Currently Sick','Is Pregnant':'Currently Pregnant','Had Thyroid Surgery':'Had Thyroid Surgery','Had I131 Treatment':'Had I131 Radioactive Iodine Treatment','Query Hypothyroid':'Querying Hypothyroidism','Query Hyperthyroid':'Querying Hyperthyroidism','On Lithium':'On Lithium Medication','Has Goitre':'Presence of Goitre','Has Tumor':'Presence of Thyroid Tumor','Psych Condition':'Reported Psychiatric Condition','TSH Level':'TSH Level (mIU/L)','T3 Level':'T3 Level (ng/dL)','TT4 Level':'Total T4 (TT4) Level (µg/dL)','T4U Level':'T4U (Thyroxine Uptake) Level','FTI Level':'Free Thyroxine Index (FTI)','TBG Level':'Thyroid Binding Globulin (TBG) Level (µg/dL)'},
}
# MODEL_MAPPING: Ensure all models loaded are mapped here
MODEL_MAPPING = {
    "Diabetes": {"name": "Diabetes", "features": DIABETES_FEATURES, "model": diabetes_model, "scaler": diabetes_scaler, "emoji": "🩸"},
    "Kidney Disease": {"name": "Kidney Disease", "features": KIDNEY_FEATURES, "model": kidney_model, "scaler": None, "emoji": "🩺"},
    "Heart Disease": {"name": "Heart Disease", "features": HEART_FEATURES, "model": heart_model, "scaler": heart_scaler, "emoji": "❤️"},
    "Hypertension": {"name": "Hypertension", "features": HYPERTENSION_FEATURES, "model": hypertension_model, "scaler": None, "emoji": "📈"},
    "Breast Cancer": {"name": "Breast Cancer", "features": BREAST_FEATURES, "model": breast_model, "scaler": None, "emoji": "🎗️"},
    "Lung Cancer": {"name": "Lung Cancer", "features": LUNG_FEATURES, "model": lung_model, "scaler": None, "emoji": "🫁"},
    "Liver Disease": {"name": "Liver Disease", "features": LIVER_FEATURES, "model": liver_model, "scaler": None, "emoji": "🌿"},
    "Thyroid Disease": {"name": "Thyroid Disease", "features": THYROID_FEATURES, "model": thyroid_model, "scaler": None, "emoji": "🦋"},
}

# --- Chat History Functions ---
@st.cache_data
def load_chat_history():
    # ... (same as before)
    if os.path.exists("chat_history.json"):
        try:
            with open("chat_history.json", "r", encoding="utf-8") as f: return json.load(f)
        except json.JSONDecodeError: return [("bot", "Chat history might be corrupted.")]
    return [("bot", f"Welcome to {APP_NAME} Chat! How can I help?")]

def save_chat_history(chat_history):
    # ... (same as before)
    with open("chat_history.json", "w", encoding="utf-8") as f: json.dump(chat_history, f)


# --- CSS Styling ---
st.markdown(f"""
<style>
    /* --- Base & Body --- */
    :root {{
        --primary-color: #0072CE; /* A professional blue */
        --secondary-color: #00A9E0; /* A lighter, vibrant blue/teal for accents */
        --accent-color: #10A37F; /* Greenish Teal for highlights */
        --text-color: #333F48; /* Dark gray for text */
        --bg-color-light: #F8F9FA; /* Very light gray for page background */
        --bg-color-dark: #0E1629; /* Dark for sidebar */
        --card-bg-color: #FFFFFF;
        --border-radius-sm: 4px;
        --border-radius-md: 8px;
        --box-shadow-soft: 0 2px 5px rgba(0,0,0,0.06);
        --box-shadow-lifted: 0 4px 12px rgba(0,0,0,0.1);
        --font-family-sans-serif: 'Roboto', 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
    }}
    body {{
        font-family: var(--font-family-sans-serif);
        background-color: var(--bg-color-light);
        color: var(--text-color);
        line-height: 1.6;
    }}
    #MainMenu, footer, header {{ visibility: hidden; }}

    /* --- Page Container --- */
    .page-container {{
        padding-top: {NAV_HEIGHT_PX + 20}px;
        padding-left: 1rem; padding-right: 1rem; padding-bottom: 2rem;
        max-width: 1100px; margin: 0 auto;
    }}

    /* --- Top Navigation Bar --- */
    .topnav {{
        position: fixed; top: 0; left: 0; width: 100%; height: {NAV_HEIGHT_PX}px;
        background-color: var(--card-bg-color); color: var(--text-color);
        display: flex; align-items: center; padding: 0 1.5rem;
        box-shadow: var(--box-shadow-soft); z-index: 1000;
    }}
    .topnav .logo h1 {{
        color: var(--primary-color); font-size: 1.6rem; font-weight: 700; margin: 0;
    }}
    .topnav .nav-links {{ margin-left: auto; display: flex; }}
    .topnav .nav-links a {{
        color: #555; padding: 0.6rem 1rem; text-decoration: none;
        font-size: 0.9rem; font-weight: 500; border-radius: var(--border-radius-sm);
        transition: background-color 0.2s ease, color 0.2s ease; margin-left: 0.5rem;
    }}
    .topnav .nav-links a:hover {{ background-color: #e9ecef; color: var(--primary-color); }}
    .topnav .nav-links a.active {{ background-color: var(--primary-color); color: white; }}

    /* --- Sidebar --- */
    [data-testid="stSidebar"] {{
        background-color: var(--bg-color-dark); padding-top: 1rem; border-right: none;
    }}
    .sidebar-title {{
        text-align: center; font-size: 1.4rem; font-weight: 600;
        color: var(--accent-color); margin: 0.5rem 0 1.5rem 0;
    }}
    [data-testid="stSidebar"] .stImage img {{
        display: block; margin: 0 auto 1rem auto; border-radius: var(--border-radius-md);
        max-width: 100px; /* Control logo size */
    }}
    [data-testid="stSidebar"] .stButton>button {{
        background-color: transparent; color: #bac2de;
        border: none; width: 100%; text-align: left;
        padding: 0.7rem 1rem; margin-bottom: 0.3rem; border-radius: var(--border-radius-sm);
        font-weight: 500; font-size: 0.95rem;
        transition: background-color 0.2s ease, color 0.2s ease;
        display: flex; align-items: center; /* For icon alignment */
    }}
    [data-testid="stSidebar"] .stButton>button .icon {{ margin-right: 0.7rem; }} /* Class for icon span */
    [data-testid="stSidebar"] .stButton>button:hover {{
        background-color: rgba(255,255,255,0.05); color: #ffffff;
    }}
    [data-testid="stSidebar"] .stButton>button.active-sidebar-button {{
        background-color: var(--accent-color); color: #ffffff; font-weight: 600;
    }}
    [data-testid="stSidebar"] hr {{ border-color: #2c3e50; margin: 1rem 0; }}
    [data-testid="stSidebar"] h4 {{ color: var(--secondary-color); margin-top:1rem; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.5px; }}
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] a {{ color: #adb5bd; font-size: 0.85rem; }}
    [data-testid="stSidebar"] a:hover {{ color: var(--accent-color); }}
    [data-testid="stSidebar"] .stAlert {{ background-color: rgba(255,255,255,0.05); border-radius: var(--border-radius-sm); }}
    [data-testid="stSidebar"] .stAlert p {{ color: #bac2de; }}


    /* --- Chat Interface --- */
    .chat-box {{
        min-height: 400px; max-height: 60vh; overflow-y: auto; border-radius: var(--border-radius-md);
        padding: 1rem; background-color: var(--card-bg-color);
        box-shadow: var(--box-shadow-soft); margin-bottom: 1rem;
        display: flex; flex-direction: column; gap: 0.75rem;
    }}
    .message-container {{ display: flex; max-width: 78%; align-items: flex-end;}} /* Align items to bottom for icon */
    .user-container {{ margin-left: auto; justify-content: flex-end; }}
    .bot-container {{ margin-right: auto; justify-content: flex-start; }}
    .message-bubble {{
        padding: 0.7rem 1.1rem; border-radius: 16px;
        line-height: 1.5; font-size: 0.9rem;
        display: flex; /* align-items: center; remove this if icon is distinct */
        box-shadow: 0 1px 1.5px rgba(0,0,0,0.07);
        word-break: break-word; /* Ensure long words break */
    }}
    .message-bubble .icon {{ /* For icon if placed outside text block */
        margin-right: 0.5rem; font-size: 1.3em; align-self: flex-start; /* Icon at top of bubble */
        line-height: 1; /* Ensure icon doesn't affect text line height */
    }}
    .message-text {{ /* Wrap the actual text */
        flex-grow: 1;
    }}
    .user-bubble {{
        background: var(--primary-color); color: white; border-bottom-right-radius: var(--border-radius-sm);
    }}
    .bot-bubble {{
        background: #e9ecef; color: var(--text-color); border-bottom-left-radius: var(--border-radius-sm);
    }}
    [data-testid="stChatInput"] > div {{ /* Target inner div for better control */
        background-color: var(--card-bg-color); border-radius: 20px;
        padding: 0.3rem 0.5rem; box-shadow: var(--box-shadow-soft);
    }}
    [data-testid="stChatInput"] textarea {{
        border: none !important; box-shadow: none !important;
        background-color: transparent !important;
        padding: 0.6rem 0.8rem !important; font-size: 0.95rem;
    }}
    [data-testid="stChatInput"] button {{ /* Send button */
        background-color: var(--primary-color) !important;
        color: white !important;
        border-radius: 50% !important; /* Circular send button */
        padding: 0.5rem !important;
        margin-right: 0.2rem;
    }}
     [data-testid="stChatInput"] button:hover {{
        background-color: var(--secondary-color) !important;
    }}


    /* --- General UI Elements --- */
    .stButton>button {{
        background-color: var(--primary-color); color: white;
        border: none; padding: 0.6rem 1.2rem; border-radius: var(--border-radius-sm);
        font-weight: 500; font-size: 0.95rem;
        transition: background-color 0.2s ease, transform 0.1s ease;
    }}
    .stButton>button:hover {{ background-color: var(--secondary-color); }}
    .stButton>button:active {{ transform: scale(0.97); }}

    .stTextInput input, .stSelectbox div[data-baseweb="select"] > div {{
         border-radius: var(--border-radius-sm) !important;
         border: 1px solid #ced4da !important;
         font-size: 0.95rem;
         padding: 0.5rem 0.75rem !important; /* Consistent padding */
    }}
    .stTextInput input:focus, .stSelectbox div[data-baseweb="select"] > div[aria-expanded="true"] {{
         border-color: var(--primary-color) !important;
         box-shadow: 0 0 0 0.15rem rgba(0, 114, 206, 0.2) !important; /* Primary color focus shadow */
    }}
    .stSlider {{ margin-bottom: 0.5rem;}} /* Add some space below sliders */

    h1, h2, h3, h4, h5, h6 {{ color: var(--text-color); font-weight: 600; margin-top: 1.5rem; margin-bottom: 0.8rem;}}
    h1 {{ font-size: 2rem; color: var(--primary-color);}}
    h2 {{ font-size: 1.6rem; }}
    h3 {{ font-size: 1.3rem; }}
    .section-header {{
        color: var(--primary-color); margin-bottom: 1.2rem; padding-bottom: 0.6rem;
        border-bottom: 2px solid #dee2e6; font-size:1.5rem;
    }}
    .card {{
        background-color: var(--card-bg-color); padding: 1.5rem;
        border-radius: var(--border-radius-md); box-shadow: var(--box-shadow-soft);
        margin-bottom: 1.5rem;
    }}
    .disease-card-home {{
        background-color: var(--card-bg-color); padding: 1.2rem; border-radius: var(--border-radius-md);
        box-shadow: var(--box-shadow-soft); text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease; margin-bottom: 1rem; height: 100%;
        display: flex; flex-direction: column; justify-content: center; align-items: center;
    }}
    .disease-card-home:hover {{ transform: translateY(-5px); box-shadow: var(--box-shadow-lifted); }}
    .disease-card-home .emoji {{ font-size: 2.5rem; margin-bottom: 0.5rem; line-height:1; }}
    .disease-card-home h5 {{ color: var(--primary-color); margin-bottom: 0; font-weight: 600; font-size: 1.1rem; }}

    /* Result display for symptom checker */
    .symptom-result-card {{
        background-color: #f8f9fa; /* Lighter than main card */
        border: 1px solid #e9ecef;
        padding: 1rem;
        margin-top: 1rem;
        border-radius: var(--border-radius-md);
    }}
    .symptom-result-card h6 {{ color: var(--primary-color); margin-top:0; }}
    .symptom-result-card .stProgress > div > div {{ /* Progress bar color */
        background-color: var(--accent-color) !important;
    }}

    /* --- Mobile Responsiveness --- */
    @media (max-width: 992px) {{ /* Tablet */
        .page-container {{ padding-left: 1rem; padding-right: 1rem; }}
        .topnav .logo h1 {{ font-size: 1.4rem; }}
        .topnav .nav-links a {{ padding: 0.5rem 0.7rem; font-size: 0.85rem; }}
        .disease-card-home h5 {{ font-size: 1rem; }}
    }}
    @media (max-width: 768px) {{ /* Mobile */
        .page-container {{ padding-top: {NAV_HEIGHT_PX + 10}px; }}
        .topnav {{ padding: 0 1rem; height: {NAV_HEIGHT_PX - 5}px; }}
        .topnav .logo h1 {{ font-size: 1.3rem; }}
        .topnav .nav-links {{ /* Could hide some links or make them icons */
            gap: 0.1rem; /* Reduce gap for smaller screens */
        }}
        .topnav .nav-links a {{ padding: 0.4rem 0.5rem; font-size: 0.8rem; letter-spacing: -0.5px; }}
        h1 {{ font-size: 1.7rem; }} h2 {{ font-size: 1.4rem; }} h3 {{ font-size: 1.2rem; }}
        .chat-box {{ min-height: 300px; max-height: 50vh; padding: 0.8rem; }}
        .message-container {{ max-width: 85%; }}
        .message-bubble {{ padding: 0.6rem 1rem; font-size: 0.85rem; }}
        .disease-card-home {{ padding: 1rem; }}
        .disease-card-home .emoji {{ font-size: 2rem; }}
        /* Symptom checker columns */
        /* Streamlit handles column stacking, but we can adjust checkbox style */
        [data-testid="stCheckbox"] label span {{ font-size: 0.9rem; }}
    }}
</style>
""", unsafe_allow_html=True)

# --- Page Navigation Logic ---
# query_params = st.query_params
# page_values = query_params.get("page")
# current_page = "home" # Default
# if page_values and isinstance(page_values, list) and len(page_values) > 0:
#     current_page = page_values[0]

query_params = st.query_params
page_values = query_params.get("page", "home")
current_page = "home" # Default
if page_values and isinstance(page_values, list) and len(page_values) > 0:
    current_page = page_values[0]
else:
    page = "home" # Default to home

# --- Sidebar ---
with st.sidebar:
    # Ensure the logo path is correct. If you don't have an 'assets' folder, adjust or remove.
    logo_path = "assets/logo.png" # Example path
    if os.path.exists(logo_path):
        st.image(logo_path, width=100)
    else:
        st.markdown(f"<h2 style='text-align:center; color:var(--accent-color);'>⚕️</h2>", unsafe_allow_html=True) # Fallback emoji/icon

    st.markdown(f"<div class='sidebar-title'>{APP_NAME.split(' ', 1)[1]}</div>", unsafe_allow_html=True) # Show only "IntelliMed AI"
    st.markdown("---")

    def sidebar_button_nav(label, page_key, icon=""):
        # For visual active state, we'd ideally need to add a class to the button.
        # Streamlit's st.button doesn't allow adding arbitrary HTML classes directly.
        # We'll rely on the topnav for clear active state, or just highlight on hover.
        if st.button(f"{icon} {label}", key=f"sidebar_nav_{page_key}", use_container_width=True):
            st.query_params["page"] = page_key # This updates the server-side query_params
            st.query_params(page=page_key) # This updates the browser URL bar
            st.rerun()

    sidebar_button_nav("Home", "home", "🏠")
    sidebar_button_nav("Diagnose Disease", "diagnose", "🔬")
    sidebar_button_nav("AI Chatbot", "chatbot", "💬")

    st.markdown("---")
    st.markdown("<h4>App Information</h4>", unsafe_allow_html=True)
    st.markdown(f"<p>Version: {APP_VERSION}</p>", unsafe_allow_html=True)
    st.markdown("<h4>Contact Developer</h4>", unsafe_allow_html=True)
    st.markdown("<p><a href='mailto:subhashkumardev@outlook.com' target='_blank'>📧 Email Developer</a></p>", unsafe_allow_html=True)
    st.markdown("<p><a href='https://www.linkedin.com/in/subhashkumar-dev/' target='_blank'>🔗 LinkedIn</a></p>", unsafe_allow_html=True)
    st.markdown("<p><a href='https://github.com/Subhash捃' target='_blank'>💻 GitHub</a></p>", unsafe_allow_html=True)
    st.markdown("---")
    st.info("This app is for educational purposes. Not a substitute for professional medical advice.")

# --- Top Navigation Bar ---
home_active = "active" if current_page == "home" else ""
diagnose_active = "active" if current_page == "diagnose" else ""
chatbot_active = "active" if current_page == "chatbot" else ""

st.markdown(f"""
<div class="topnav">
    <div class="logo"><h1>{APP_NAME}</h1></div>
    <div class="nav-links">
        <a class="{home_active}" href="?page=home">Home</a>
        <a class="{diagnose_active}" href="?page=diagnose">Diagnose</a>
        <a class="{chatbot_active}" href="?page=chatbot">Chatbot</a>
    </div>
</div>
""", unsafe_allow_html=True)


# --- Helper Function to Sanitize String for Keys ---
def sanitize_key(text):
    return re.sub(r'\W+', '_', text.lower())

# --- Page Rendering Functions ---
def render_home_page():
    st.title("Welcome to IntelliMed AI")
    st.subheader("Your Smart AI Medical Diagnosis Assistant")
    st.markdown("""
    IntelliMed AI leverages machine learning for preliminary insights into health conditions.
    Analyze data with trained models to predict disease likelihood.
    **Navigate to Diagnose** for prediction forms, or use our **Chatbot** for health inquiries.

    **Disclaimer:** This platform is for educational and informational purposes only.
    **Always consult with a qualified healthcare professional for definitive medical advice and diagnosis.**
    """, unsafe_allow_html=True) # Allow HTML for bolding
    st.markdown("---")
    st.markdown("<h3 class='section-header'>🧬 Diseases Covered</h3>", unsafe_allow_html=True)

    model_keys = list(MODEL_MAPPING.keys())
    if not model_keys:
        st.info("No disease models are currently configured for display.")
        return

    num_disease_cols = 3 if len(model_keys) >= 3 else (2 if len(model_keys) == 2 else 1)
    disease_cols = st.columns(num_disease_cols)

    for i, disease_key in enumerate(model_keys):
        disease_info = MODEL_MAPPING[disease_key]
        disease_name_home = disease_info["name"]
        emoji_char = disease_info.get("emoji", "⚕️") # Use emoji from mapping or default
        with disease_cols[i % num_disease_cols]:
            st.markdown(f"""
            <div class="disease-card-home">
                <div class="emoji">{emoji_char}</div>
                <h5>{disease_name_home}</h5>
            </div>
            """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True) # Spacer
    st.markdown("---")
    st.markdown("<h3 class='section-header'>Technologies Used</h3>", unsafe_allow_html=True)
    # ... (rest of home page content)
    st.markdown("- **Backend & ML:** Python, Scikit-learn, Joblib, Pandas, NumPy")
    st.markdown("- **Frontend:** Streamlit")
    st.markdown("- **AI Chatbot:** Google Generative AI (Gemini)")
    st.markdown("- **Styling:** Custom HTML/CSS")
    st.markdown("---")
    st.markdown("<p style='text-align:center; font-size:0.9rem; color:#777;'>Developed with 💚 by Subhash Kumar</p>", unsafe_allow_html=True)


def render_symptom_checker():
    st.markdown("<h3 class='section-header'>General Symptom Checker</h3>", unsafe_allow_html=True)
    st.caption("Select your symptoms below. This tool offers preliminary suggestions, not a definitive diagnosis.")

    # Define diseases_data here or load from a JSON/config file for better management
    diseases_data = {
        'Common Cold': ['runny nose', 'sore throat', 'cough', 'sneezing', 'mild headache', 'fatigue', 'nasal congestion'],
        'Influenza (Flu)': ['fever', 'body aches', 'chills', 'fatigue', 'cough', 'sore throat', 'headache', 'muscle pain', 'nausea', 'vomiting'],
        'Allergies': ['sneezing', 'itchy eyes', 'runny nose', 'nasal congestion', 'itchy throat', 'watery eyes', 'skin rash'],
        'Strep Throat': ['sore throat', 'fever', 'swollen lymph nodes', 'difficulty swallowing', 'red spots on roof of mouth', 'headache', 'nausea', 'vomiting', 'body aches'],
        'Bronchitis': ['cough', 'chest discomfort', 'shortness of breath', 'fatigue', 'sore throat', 'mild fever', 'chills', 'mucus production'],
        'Pneumonia': ['cough', 'fever', 'chills', 'shortness of breath', 'chest pain', 'fatigue', 'nausea', 'vomiting', 'confusion', 'sweating'],
        'Migraine': ['severe headache', 'pulsating pain', 'sensitivity to light', 'sensitivity to sound', 'nausea', 'vomiting', 'aura', 'dizziness'],
        'Tension Headache': ['mild to moderate headache', 'pressure around head', 'neck pain', 'shoulder pain', 'scalp tenderness'],
        'Gastroenteritis (Stomach Flu)': ['nausea', 'vomiting', 'diarrhea', 'abdominal cramps', 'mild fever', 'fatigue', 'muscle pain', 'headache'],
        'Urinary Tract Infection (UTI)': ['frequent urination', 'painful urination', 'burning sensation during urination', 'cloudy urine', 'strong smelling urine', 'pelvic pain', 'fever', 'chills', 'nausea']
    }
    all_symptoms = sorted(list(set(symptom for symptoms_list in diseases_data.values() for symptom in symptoms_list)))

    if 'selected_symptoms_list' not in st.session_state:
        st.session_state.selected_symptoms_list = []

    # For responsive columns - on smaller screens, you might want fewer columns.
    # Streamlit columns stack automatically, but for initial layout:
    # This is a simplistic way; true responsive columns would need CSS Grid/Flex in HTML.
    # For Streamlit, we rely on its stacking. Let's use 2 or 3 columns.
    num_symptom_cols = st.columns(3) # Use 3 columns for symptoms
    
    # Use a temporary list to build selections for this run
    current_selection = []
    for i, symptom in enumerate(all_symptoms):
        # Robust key generation
        symptom_key = f"symptom_cb_{sanitize_key(symptom)}"
        # Check if symptom was previously selected
        is_checked = symptom in st.session_state.selected_symptoms_list
        
        if num_symptom_cols[i % len(num_symptom_cols)].checkbox(symptom.capitalize(), value=is_checked, key=symptom_key):
            if symptom not in current_selection: # Add if checked now
                current_selection.append(symptom)
        elif symptom in current_selection: # Remove if unchecked now (should not happen with this logic)
             current_selection.remove(symptom)
    
    # Update session state based on current checkbox states AFTER all checkboxes rendered
    st.session_state.selected_symptoms_list = current_selection


    if st.button("🩺 Suggest Possible Conditions", key="symptom_checker_diagnose_button", use_container_width=True):
        selected_symptoms_final = st.session_state.selected_symptoms_list
        if not selected_symptoms_final:
            st.warning("Please select at least one symptom from the list.")
        else:
            disease_scores = {}
            for disease, symptoms_list in diseases_data.items():
                matches = sum(1 for sym in selected_symptoms_final if sym in symptoms_list)
                if matches > 0:
                    total_disease_symptoms = len(symptoms_list)
                    confidence = int((matches / total_disease_symptoms) * 100) if total_disease_symptoms > 0 else 0
                    disease_scores[disease] = {"matches": matches, "total": total_disease_symptoms, "confidence": confidence}

            if not disease_scores:
                st.info("No common conditions strongly match the selected symptoms based on this basic checker. If concerned, please consult a doctor or use a specific disease prediction form.")
            else:
                st.markdown("---")
                st.subheader("Preliminary Suggestions:")
                st.caption("These are based on symptom overlap and are not medical diagnoses.")
                sorted_diseases = sorted(disease_scores.items(), key=lambda x: x[1]['confidence'], reverse=True)

                for disease, scores_data in sorted_diseases:
                    with st.container(): # class="symptom-result-card" can be applied via markdown wrapper
                        st.markdown(f"<div class='symptom-result-card'><h6>{disease}</h6>", unsafe_allow_html=True)
                        # st.metric(label=disease, value=f"{scores_data['confidence']}% Match", delta=f"{scores_data['matches']}/{scores_data['total']} symptoms")
                        st.write(f"**Symptom Match:** {scores_data['matches']} out of {scores_data['total']}")
                        st.write(f"**Match Percentage:**")
                        st.progress(scores_data['confidence'])
                        st.markdown("</div>", unsafe_allow_html=True)
                        st.markdown("<br>", unsafe_allow_html=True) # Little space between cards
    # Add a button to clear selected symptoms
    if st.button("Clear Selected Symptoms", key="clear_symptoms_button"):
        st.session_state.selected_symptoms_list = []
        st.rerun()


def render_prediction_form_page(disease_key):
    # ... (Same as your well-defined render_prediction_form_page from previous, ensure model checks)
    if disease_key not in MODEL_MAPPING:
        st.error(f"Configuration for '{disease_key}' not found.")
        return
    
    disease_info = MODEL_MAPPING[disease_key]
    if not disease_info.get("model"):
        st.error(f"Model for {disease_info['name']} is not available or failed to load. Please check the system configuration.")
        return

    disease_name = disease_info["name"]
    features = disease_info["features"]
    model_instance = disease_info["model"]
    scaler_instance = disease_info.get("scaler")

    st.markdown(f"<h3 class='section-header'>Predict {disease_name}</h3>", unsafe_allow_html=True)
    
    with st.form(key=f"{sanitize_key(disease_key)}_prediction_form"):
        st.markdown(f"Please fill in the patient details for **{disease_name}** assessment:")
        inputs = {}
        
        num_features = len(features)
        # Responsive columns: 1 on small, 2 on medium, 3 on large screens (Streamlit handles stacking)
        num_cols = 1 if num_features <= 3 else (2 if num_features <= 9 else 3)
        form_cols = st.columns(num_cols)

        for i, feature_name in enumerate(features):
            with form_cols[i % num_cols]:
                f_key = feature_name.strip()
                display_label = feature_fullforms.get(disease_name, {}).get(f_key, f_key.replace("_", " ").capitalize())

                if f_key in FEATURE_INPUTS:
                    input_type, *params = FEATURE_INPUTS[f_key]
                    # Slider: (min, max, default, step)
                    # Select: (options_list)
                    if input_type == 'slider':
                        min_val, max_val, default_val, step_val = params
                        inputs[f_key] = st.slider(
                            display_label,
                            min_value=float(min_val), max_value=float(max_val),
                            value=float(default_val), step=float(step_val),
                            key=f"{sanitize_key(disease_key)}_{f_key}_slider"
                        )
                    elif input_type == 'select':
                        options = params[0]
                        inputs[f_key] = st.selectbox(
                            display_label, options, index=0, # Default to first option
                            key=f"{sanitize_key(disease_key)}_{f_key}_select"
                        )
                else: # Fallback if feature not in FEATURE_INPUTS (should not happen with proper config)
                    inputs[f_key] = st.text_input(display_label, key=f"{sanitize_key(disease_key)}_{f_key}_text")
        
        st.markdown("<br>", unsafe_allow_html=True) # Spacer before button
        submitted = st.form_submit_button("🔍 Predict Diagnosis", use_container_width=True)

    if submitted:
        # ... (Conversion, scaling, prediction, result display, and PDF report logic from previous version)
        # Ensure convert_input_value is robust
        def convert_input_value(val_str):
            val_lower = str(val_str).lower()
            if val_lower in ['yes', 'present', 'male', 'normal', 'true']: return 1
            if val_lower in ['no', 'notpresent', 'female', 'abnormal', 'false']: return 0
            try: return float(val_str)
            except (ValueError, TypeError):
                st.warning(f"Could not convert '{val_str}' to a number. Using as is or 0 if not resolved.")
                return 0 # Or handle more gracefully, e.g., raise error or skip
        
        processed_values = []
        valid_inputs = True
        for f_name in features:
            f_key_strip = f_name.strip()
            if f_key_strip in inputs:
                processed_values.append(convert_input_value(inputs[f_key_strip]))
            else: # Should not happen if form generated from features
                st.error(f"Missing input for {f_key_strip}")
                valid_inputs = False
                break
        
        if not valid_inputs: return

        try:
            data_df = pd.DataFrame([processed_values], columns=[f.strip() for f in features])
            data_to_predict = data_df.copy()

            if scaler_instance:
                try:
                    scaled_data_array = scaler_instance.transform(data_df)
                    data_to_predict = pd.DataFrame(scaled_data_array, columns=data_df.columns)
                except Exception as e:
                    st.error(f"Error during data scaling: {e}. Please check input values' ranges and types.")
                    return

            prediction = model_instance.predict(data_to_predict)[0]
            confidence_display = "N/A"
            if hasattr(model_instance, 'predict_proba'):
                probs = model_instance.predict_proba(data_to_predict)[0]
                # For binary classification, confidence is usually prob of predicted class
                # or specifically prob of positive class if prediction is 1
                if prediction == 1 and len(probs) > 1: # Positive class
                    confidence_score = probs[1]
                elif prediction == 0 and len(probs) > 1: # Negative class
                     confidence_score = probs[0]
                else: # Single class probability or fallback
                    confidence_score = probs[np.argmax(probs)]
                confidence_display = f"{confidence_score:.0%}"


            diagnosis_result = "Positive" if prediction == 1 else "Negative"

            st.session_state['prediction_result_data'] = {
                'title': disease_name, 'inputs': inputs,
                'diagnosis': diagnosis_result, 'confidence': confidence_display
            }

            result_message_type = st.error if diagnosis_result == "Positive" else st.success
            result_message_type(f"Diagnosis: {diagnosis_result} (Confidence: {confidence_display})")
            st.balloons()

            if 'prediction_result_data' in st.session_state:
                 render_pdf_report(st.session_state['prediction_result_data'])

        except Exception as e:
            st.error(f"An error occurred during the prediction process: {e}")
            # st.dataframe(data_to_predict) # For debugging what was sent to model


def render_pdf_report(result_data):
    # ... (Your PDF rendering logic - ensure it's robust)
    # Using Helvetica, a standard PDF font.
    st.markdown("---")
    st.markdown("#### Diagnosis Report Summary")
    # ... (rest of PDF generation)
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", 'B', 16)
    pdf.cell(0, 10, "Medical Diagnosis Report", ln=True, align='C')
    pdf.set_font("Helvetica", '', 10)
    pdf.cell(0, 7, f"Diagnosed by {APP_NAME}", ln=True, align='C')
    pdf.ln(10)

    pdf.set_font("Helvetica", 'B', 12)
    pdf.cell(0, 8, f"Disease Assessed: {result_data['title']}", ln=True)
    pdf.set_font("Helvetica", '', 11)
    pdf.cell(0, 8, f"Diagnosis Result: {result_data['diagnosis']}", ln=True)
    pdf.cell(0, 8, f"Confidence Score: {result_data['confidence']}", ln=True)
    pdf.ln(7)

    pdf.set_font("Helvetica", 'B', 12)
    pdf.cell(0, 8, "Patient Provided Details:", ln=True)
    pdf.set_font("Helvetica", '', 10)
    for k, v_original in result_data['inputs'].items():
        display_key_pdf = feature_fullforms.get(result_data['title'], {}).get(k, k.replace("_", " ").capitalize())
        pdf.multi_cell(0, 6, f"{display_key_pdf}: {v_original}")
    pdf.ln(5)

    pdf.set_font("Helvetica", 'I', 8)
    pdf.multi_cell(0, 5, "Disclaimer: This AI-generated report is for informational purposes only and is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.", align='C')

    try:
        pdf_output_bytes = pdf.output(dest='S').encode('latin-1')
        pdf_buffer = io.BytesIO(pdf_output_bytes)
        st.download_button(
            label="📥 Download Full Report (PDF)", data=pdf_buffer,
            file_name=f"{sanitize_key(result_data['title'])}_report.pdf", mime="application/pdf",
            key="download_pdf_button", use_container_width=True
        )
    except Exception as e:
        st.error(f"Could not generate PDF: {e}")


def render_diagnose_page():
    st.title("🔬 Disease Diagnosis Center")
    st.markdown("Choose a diagnosis method below to get started with your health assessment.")
    st.markdown("---")

    diag_mode_options = ["Specific Disease Prediction", "General Symptom Checker", "Image Based (Coming Soon)"]
    
    # Use st.tabs for a more modern selection method
    tab1, tab2, tab3 = st.tabs(diag_mode_options)

    with tab1:
        st.markdown("#### Predict Based on Specific Disease Model")
        st.caption("Select a disease model for a detailed diagnosis using specific patient data points.")
        model_options_keys = [k for k,v in MODEL_MAPPING.items() if v.get("model")] # Only show if model loaded
        if not model_options_keys:
            st.warning("No specific disease prediction models are currently available or loaded correctly.")
        else:
            selected_disease_key = st.selectbox(
                "Choose a Disease Model:", model_options_keys, index=0,
                format_func=lambda k: MODEL_MAPPING[k]["name"], # Show user-friendly name
                key="diagnose_disease_select_tab"
            )
            if selected_disease_key:
                with st.container(border=True): # class="card"
                    render_prediction_form_page(selected_disease_key)
    
    with tab2:
        render_symptom_checker()

    with tab3:
        st.info("🖼️ Image-based diagnosis features are under development and will be available soon. Thank you for your patience!")
        # Placeholder for future image-based diagnosis
        # st.image("path/to/your/coming_soon_image.png", caption="Feature Coming Soon", width=300)


def render_chatbot_page():
    st.title("💬 IntelliMed AI Chatbot")

    if not GEMINI_CONFIGURED or not gemini_llm_model:
        st.warning("🤖 AI Chatbot is currently unavailable. Please ensure the Gemini API key is configured correctly in your application secrets.", icon="🛠️")
        return

    st.markdown("Ask general health-related questions or get information. This chatbot is for informational purposes and not a substitute for professional medical advice.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = load_chat_history()

    # Display chat messages
    chat_display_container = st.container() # No explicit height, let chat-box CSS handle it
    with chat_display_container:
        chat_html = "<div class='chat-box'>"
        for role, msg_text in st.session_state.chat_history:
            bubble_class = "user-bubble" if role == "user" else "bot-bubble"
            container_class = "user-container" if role == "user" else "bot-container"
            icon_html = "<span class='icon'>🧑‍💻</span>" if role == "user" else "<span class='icon'>🤖</span>"
            # Sanitize msg_text to prevent HTML injection if it ever comes from untrusted source
            # For now, assuming Gemini output is safe or will be markdown-rendered by Streamlit later if needed
            escaped_msg_text = msg_text.replace("<", "&lt;").replace(">", "&gt;")

            chat_html += f"<div class='message-container {container_class}'>"
            chat_html += f"<div class='message-bubble {bubble_class}'>{icon_html}<div class='message-text'>{escaped_msg_text}</div></div>"
            chat_html += "</div>"
        chat_html += "</div>"
        st.markdown(chat_html, unsafe_allow_html=True)

    user_input = st.chat_input("Ask me anything about health...", key="chatbot_user_main_input")

    if user_input:
        st.session_state.chat_history.append(("user", user_input))
        with st.spinner("🤖 Thinking..."):
            try:
                # Ensure you handle potential errors from the API call
                response = gemini_llm_model.generate_content(user_input)
                bot_reply = response.text
            except Exception as e:
                bot_reply = f"Sorry, I encountered an error communicating with the AI: {e}"
                st.error(bot_reply) # Show error clearly
        st.session_state.chat_history.append(("bot", bot_reply))
        save_chat_history(st.session_state.chat_history)
        st.rerun()

# --- Main App Logic: Route to the correct page ---
st.markdown("<div class='page-container'>", unsafe_allow_html=True)

if current_page == "home":
    render_home_page()
elif current_page == "diagnose":
    render_diagnose_page()
elif current_page == "chatbot":
    render_chatbot_page()
else: # Fallback for unknown page
    st.error("🚫 Page Not Found")
    st.markdown(f"The page `?page={current_page}` was not found.")
    if st.button("Go to Home Page", use_container_width=True):
        st.query_params.page = "home" # Corrected to use st.query_params.page
        st.query_params(page="home")
        st.rerun()

st.markdown("</div>", unsafe_allow_html=True) # End of page-container