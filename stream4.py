import streamlit as st
import joblib
import google.generativeai as genai
import numpy as np
import pandas as pd
import json
import os
import io
from fpdf import FPDF

# --- Global Configuration & Constants ---
NAV_HEIGHT_PX = 60  # Height of the top navigation bar in pixels
APP_NAME = "🩺 IntelliMed AI"
APP_VERSION = "1.1" # Example version

# --- Page Configuration (should be the first Streamlit command) ---
st.set_page_config(
    layout='wide',
    page_icon='🎈',
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
        return genai.GenerativeModel(model_name="gemini-1.5-flash-latest") # Using a common updated model
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


# --- Feature Lists & Input Definitions (Ensure consistency with your model training) ---
# (Using truncated versions for brevity in this example. Use your full lists.)
DIABETES_FEATURES = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
KIDNEY_FEATURES = ['age', 'bp', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 'bu', 'sc', 'pot', 'wc', 'htn', 'dm', 'cad', 'pe', 'ane']
HEART_FEATURES = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach','exang','oldpeak','slope','ca','thal']
HYPERTENSION_FEATURES = ['Age', 'BMI', 'Systolic_BP','Diastolic_BP', 'Total_Cholesterol']
BREAST_FEATURES = ['mean_radius', 'mean_texture', 'mean_perimeter', 'mean_area', 'mean_smoothness', 'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean']
LUNG_FEATURES = ['GENDER', 'AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE', 'CHRONIC_DISEASE', 'FATIGUE', 'ALLERGY', 'WHEEZING', 'ALCOHOL_CONSUMING', 'COUGHING', 'SHORTNESS_OF_BREATH', 'SWALLOWING_DIFFICULTY', 'CHEST_PAIN']
LIVER_FEATURES = ['Age', 'Gender', 'Total_Bilirubin', 'Alkaline_phosphate', 'Alamine_Aminotransferace', 'Aspartate_Amino', 'Protien', 'Albumin', 'Albumin_Globulin_ratio']
THYROID_FEATURES = ['Age','Sex','On Thyroxine','Query on Thyroxine','On Antithyroid Meds','Is Sick','Is Pregnant','Had Thyroid Surgery','Had I131 Treatment','Query Hypothyroid','Query Hyperthyroid','On Lithium','Has Goitre','Has Tumor','Psych Condition','TSH Level','T3 Level','TT4 Level','T4U Level','FTI Level', 'TBG Level']


# FEATURE_INPUTS dictionary should contain all unique feature names from ALL _FEATURES lists above.
# This dictionary maps a feature name (string) to its input type and parameters.
# Example:
FEATURE_INPUTS = {
    # Diabetes
    'Pregnancies': ('slider', 0, 20, 1), 'Glucose': ('slider', 40, 300, 100), 'BloodPressure': ('slider', 30, 180, 80),
    'SkinThickness': ('slider', 0, 100, 20), 'Insulin': ('slider', 0, 900, 150),
    'BMI': ('slider', 10.0, 70.0, 25.0, 0.1), 'DiabetesPedigreeFunction': ('slider', 0.01, 2.5, 0.5, 0.01),
    # Age is used by many, ensure parameters are sensible as a general default or handle contextually
    'Age': ('slider', 1, 120, 30), # Hypertension, Liver, Thyroid (uppercase A)
    'age': ('slider', 1, 100, 30), # Kidney, Heart (lowercase a)
    'AGE': ('slider', 10, 100, 30),# Lung (UPPERCASE A)

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
    'Sex': ('select', ['Female', 'Male']), # Thyroid (uppercase S)
    'cp': ('slider', 0, 3, 0), 'trestbps': ('slider', 80, 200, 120), 'chol': ('slider', 100, 600, 200),
    'fbs': ('select', ['yes', 'no']), 'restecg': ('slider', 0, 2, 0), 'thalach': ('slider', 60, 220, 150),
    'exang': ('select', ['yes', 'no']), 'oldpeak': ('slider', 0.0, 6.0, 1.0, 0.1), 'slope': ('slider', 0, 2, 1),
    'ca': ('slider', 0, 4, 0), 'thal': ('slider', 0, 3, 2), # 0,1,2,3 are typical values

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
    'GENDER': ('select', ['Male', 'Female']), 'SMOKING': ('select', ['Yes', 'No']),
    'YELLOW_FINGERS': ('select', ['Yes', 'No']), 'ANXIETY': ('select', ['Yes', 'No']),
    'PEER_PRESSURE': ('select', ['Yes', 'No']), 'CHRONIC_DISEASE': ('select', ['Yes', 'No']),
    'FATIGUE': ('select', ['Yes', 'No']), 'ALLERGY': ('select', ['Yes', 'No']),
    'WHEEZING': ('select', ['Yes', 'No']), 'ALCOHOL_CONSUMING': ('select', ['Yes', 'No']),
    'COUGHING': ('select', ['Yes', 'No']), 'SHORTNESS_OF_BREATH': ('select', ['Yes', 'No']),
    'SWALLOWING_DIFFICULTY': ('select', ['Yes', 'No']), 'CHEST_PAIN': ('select', ['Yes', 'No']),

    # Liver (Age already defined)
    'Gender': ('select', ['Male', 'Female']), 'Total_Bilirubin': ('slider', 0.1, 10.0, 1.0, 0.1),
    'Alkaline_phosphate': ('slider', 50, 3000, 200), 'Alamine_Aminotransferace': ('slider', 1, 2000, 50),
    'Aspartate_Amino': ('slider', 1, 2000, 50), 'Protien': ('slider', 2.0, 10.0, 6.0, 0.1),
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
}
# Full forms for display names (truncated for brevity)
feature_fullforms = {
    'Diabetes': {'Pregnancies': 'Pregnancies Count', 'Glucose': 'Glucose Level', 'Age': 'Age (Years)'},
    # ... Add ALL your features and diseases here for user-friendly labels
}

MODEL_MAPPING = {
    "Diabetes": {"name": "Diabetes", "features": DIABETES_FEATURES, "model": diabetes_model, "scaler": diabetes_scaler},
    "Kidney Disease": {"name": "Kidney Disease", "features": KIDNEY_FEATURES, "model": kidney_model, "scaler": None},
    "Heart Disease": {"name": "Heart Disease", "features": HEART_FEATURES, "model": heart_model, "scaler": heart_scaler},
    "Hypertension": {"name": "Hypertension", "features": HYPERTENSION_FEATURES, "model": hypertension_model, "scaler": None},
    "Breast Cancer": {"name": "Breast Cancer", "features": BREAST_FEATURES, "model": breast_model, "scaler": None},
    "Lung Cancer": {"name": "Lung Cancer", "features": LUNG_FEATURES, "model": lung_model, "scaler": None},
    "Liver Disease": {"name": "Liver Disease", "features": LIVER_FEATURES, "model": liver_model, "scaler": None},
    "Thyroid Disease": {"name": "Thyroid Disease", "features": THYROID_FEATURES, "model": thyroid_model, "scaler": None},
}


# --- Chat History Functions ---
@st.cache_data
def load_chat_history():
    if os.path.exists("chat_history.json"):
        try:
            with open("chat_history.json", "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return [("bot", "Welcome! Chat history might be corrupted.")]
    return [("bot", f"Welcome to {APP_NAME} Chat! How can I assist?")]

def save_chat_history(chat_history):
    with open("chat_history.json", "w", encoding="utf-8") as f:
        json.dump(chat_history, f)

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
        header {{ visibility: hidden; }} /* Hides the Streamlit hamburger menu bar if you want full control with custom nav */

        /* --- Page Container for Content --- */
        .page-container {{
            padding-top: {NAV_HEIGHT_PX + 15}px; /* Space for fixed topnav */
            padding-left: 1.5rem;
            padding-right: 1.5rem;
            padding-bottom: 3rem;
            max-width: 1200px; /* Max width for content for better readability */
            margin: 0 auto; /* Center content */
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
        .topnav .logo h1 {{
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
            background-color: #0E1629; /* Dark sidebar */
            padding-top: 1rem;
            border-right: none; /* Remove border if using dark theme that contrasts well */
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
            border-radius: 8px; /* Rounded logo */
        }}
        [data-testid="stSidebar"] .stButton>button {{
            background-color: transparent;
            color: #a8b3cf; /* Lighter text for sidebar buttons */
            border: 1px solid transparent; /* Initially no border */
            width: 100%;
            text-align: left;
            padding: 0.75rem 1rem;
            margin-bottom: 0.5rem;
            border-radius: 6px;
            font-weight: 500;
            transition: background-color 0.2s ease, color 0.2s ease, border-color 0.2s ease;
        }}
        [data-testid="stSidebar"] .stButton>button:hover {{
            background-color: #1C2B4A; /* Slightly lighter dark on hover */
            color: #ffffff;
            border-left: 3px solid #10A37F;
        }}
        [data-testid="stSidebar"] .stButton>button.active-sidebar-button {{ /* Custom class for active sidebar button */
            background-color: #10A37F; /* Accent color for active */
            color: #ffffff;
            font-weight: 600;
            border-left: 3px solid #ffffff;
        }}
        [data-testid="stSidebar"] hr {{
            border-color: #2c3e50; /* Darker separator */
            margin: 1.5rem 0;
        }}
        [data-testid="stSidebar"] h4 {{ color: #00A9E0; margin-top:1.5rem; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 0.5px; }}
        [data-testid="stSidebar"] p, [data-testid="stSidebar"] a {{ color: #a8b3cf; font-size: 0.9rem; }}
        [data-testid="stSidebar"] .stAlert {{ background-color: #1C2B4A; border-radius: 6px;}}

        /* --- Chat Interface --- */
        .chat-box {{
            height: 500px; overflow-y: auto; border-radius: 8px;
            padding: 1.5rem; background-color: #ffffff;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05); margin-bottom: 1rem;
            display: flex; flex-direction: column; gap: 0.8rem; /* Spacing between messages */
        }}
        .message-container {{ display: flex; max-width: 75%; }}
        .user-container {{ margin-left: auto; justify-content: flex-end; }}
        .bot-container {{ margin-right: auto; justify-content: flex-start; }}
        .message-bubble {{
            padding: 0.8rem 1.2rem; border-radius: 18px;
            line-height: 1.5; font-size: 0.95rem;
            display: flex; align-items: center;
            box-shadow: 0 1px 2px rgba(0,0,0,0.08);
        }}
        .message-bubble .icon {{ margin-right: 0.7rem; font-size: 1.2em; }}
        .user-bubble {{
            background: #007bff; color: white;
            border-bottom-right-radius: 5px;
        }}
        .bot-bubble {{
            background: #f0f0f0; color: #333;
            border-bottom-left-radius: 5px;
        }}
        [data-testid="stChatInput"] {{
            background-color: #ffffff;
            border-radius: 25px; /* Fully rounded input bar */
            padding: 0.2rem; /* Adjust padding around the actual input */
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        }}
        [data-testid="stChatInput"] input {{
            border: none !important;
            box-shadow: none !important;
            padding: 0.8rem 1rem !important;
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
        .card {{
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
        }}
        .disease-card-home:hover {{ transform: translateY(-4px); box-shadow: 0 6px 15px rgba(0,0,0,0.1); }}
        .disease-card-home .emoji {{ font-size: 2.8rem; margin-bottom: 0.7rem; }}
        .disease-card-home h5 {{ color: #1a5a96; margin-bottom: 0.3rem; font-weight: 600; }}

    </style>
""", unsafe_allow_html=True)


# --- Page Navigation Logic ---
query_params = st.query_params
page_values = query_params.get("page")
current_page = "home" # Default
if page_values and isinstance(page_values, list) and len(page_values) > 0:
    current_page = page_values[0]


# --- Sidebar ---
with st.sidebar:
    st.image("./static/logo.png", width=100) # Ensure path is correct
    st.markdown(f"<div class='sidebar-title'>{APP_NAME}</div>", unsafe_allow_html=True)
    st.markdown("---")

    # Active button styling logic
    def sidebar_button(label, page_key, icon=""):
        active_class = "active-sidebar-button" if current_page == page_key else ""
        # Using markdown for button to apply custom class for active state easily
        button_html = f"""
        <button class='stButton {active_class}' style='width:100%; text-align:left; background:transparent; border:none; padding:10px 15px; color: #a8b3cf;'>
            {icon} {label}
        </button>
        """
        # This HTML button won't directly work with Streamlit's callback system like st.button.
        # We'll use st.button and then rely on CSS to style based on page, or use JS (not ideal in pure Streamlit).
        # For simplicity, let's use st.button and not try to dynamically add CSS class this way.
        # The active state in sidebar is visual flair, main functionality is navigation.

        if st.button(f"{icon} {label}", key=f"sidebar_nav_{page_key}", use_container_width=True):
            st.query_params["page"] = page_key
            st.rerun()

    sidebar_button("Home", "home", "🏠")
    sidebar_button("Diagnose Disease", "diagnose", "🔬")
    sidebar_button("AI Chatbot", "chatbot", "💬")

    st.markdown("---")
    st.markdown("<h4>App Information</h4>", unsafe_allow_html=True)
    st.markdown(f"<p>Version: {APP_VERSION}</p>", unsafe_allow_html=True)
    st.markdown("<h4>Contact Developer</h4>", unsafe_allow_html=True)
    st.markdown("<p><a href='mailto:subhashkumardev@outlook.com'>📧 Email Developer</a></p>", unsafe_allow_html=True)
    st.markdown("<p><a href='https://www.linkedin.com/in/subhashkumar-dev/' target='_blank'>🔗 LinkedIn</a></p>", unsafe_allow_html=True)
    st.markdown("<p><a href='https://github.com/Subhash捃' target='_blank'>💻 GitHub</a></p>", unsafe_allow_html=True)
    st.markdown("---")
    st.info("This app is for educational purposes only. Not a substitute for professional medical advice.")


query_params = st.query_params
current_page = query_params.get("page","home")


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
    """)
    st.markdown("---")
    st.markdown("<h3 class='section-header'>🧬 Diseases Covered</h3>", unsafe_allow_html=True)

    disease_names_list = list(MODEL_MAPPING.keys())
    emojis = ["🩸", "🩺", "❤️", "📈", "🎗️", "🫁", "🌿", "🦋"] # Example Emojis

    num_disease_cols = 3 if len(disease_names_list) > 2 else len(disease_names_list)
    if num_disease_cols == 0: num_disease_cols = 1 # Avoid st.columns(0)
    
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
    st.markdown("<h3 class='section-header'>Technologies Used</h3>", unsafe_allow_html=True)
    st.markdown("""
    - **Backend & ML:** Python, Scikit-learn, Joblib, Pandas, NumPy
    - **Frontend:** Streamlit
    - **AI Chatbot:** Google Generative AI (Gemini)
    - **Styling:** Custom HTML/CSS
    """)
    st.markdown("---")
    st.markdown("<p style='text-align:center;'>Developed with 💚 by Subhash Kumar</p>", unsafe_allow_html=True)

def render_prediction_form_page(disease_key):
    if disease_key not in MODEL_MAPPING or not MODEL_MAPPING[disease_key].get("model"):
        st.error(f"Model for {disease_key} is not available or failed to load. Please check the configuration.")
        return

    disease_info = MODEL_MAPPING[disease_key]
    disease_name = disease_info["name"]
    features = disease_info["features"]
    model_instance = disease_info["model"]
    scaler_instance = disease_info.get("scaler") # Use .get for optional scaler

    st.markdown(f"<h3 class='section-header'>Predict {disease_name}</h3>", unsafe_allow_html=True)
    with st.form(key=f"{disease_key.lower().replace(' ', '_')}_prediction_form"):
        st.markdown("Please fill in the patient details below:")
        inputs = {}
        
        num_features = len(features)
        num_cols = 1 if num_features <= 4 else (2 if num_features <= 12 else 3)
        form_cols = st.columns(num_cols)

        for i, feature_name in enumerate(features):
            with form_cols[i % num_cols]:
                f_key = feature_name.strip()
                # Try to get a more descriptive label
                display_label = feature_fullforms.get(disease_name, {}).get(f_key, f_key.replace("_", " ").capitalize())

                if f_key in FEATURE_INPUTS:
                    input_type, *params = FEATURE_INPUTS[f_key]
                    default_val = params[2] if len(params) > 2 else (params[0] if input_type=='slider' else None) # Default for slider
                    step_val = params[3] if len(params) > 3 and input_type=='slider' else (0.1 if isinstance(default_val, float) else 1)


                    if input_type == 'slider':
                        inputs[f_key] = st.slider(
                            display_label,
                            min_value=float(params[0]),
                            max_value=float(params[1]),
                            value=float(default_val), # Ensure default is within min/max
                            step=float(step_val)
                        )
                    elif input_type == 'select':
                        inputs[f_key] = st.selectbox(display_label, params[0], index=0, key=f"{f_key}_select_{disease_key}")
                else:
                    inputs[f_key] = st.text_input(display_label, key=f"{f_key}_text_{disease_key}") # Fallback

        submitted = st.form_submit_button("🔍 Predict Diagnosis")

    if submitted:
        def convert_input_value(val_str):
            val_lower = str(val_str).lower()
            if val_lower in ['yes', 'present', 'male', 'normal', 'true']: return 1
            if val_lower in ['no', 'notpresent', 'female', 'abnormal', 'false']: return 0
            try: return float(val_str)
            except (ValueError, TypeError): return val_str # Or handle as error

        processed_values = [convert_input_value(inputs[f.strip()]) for f in features]

        try:
            data_df = pd.DataFrame([processed_values], columns=[f.strip() for f in features])
            data_to_predict = data_df.copy()

            if scaler_instance:
                try:
                    scaled_data_array = scaler_instance.transform(data_df)
                    data_to_predict = pd.DataFrame(scaled_data_array, columns=data_df.columns)
                except Exception as e:
                    st.error(f"Error during data scaling: {e}")
                    return

            prediction = model_instance.predict(data_to_predict)[0]
            confidence = "N/A" # Default if no probability
            if hasattr(model_instance, 'predict_proba'):
                probs = model_instance.predict_proba(data_to_predict)[0]
                # Assuming binary classification, positive class is typically index 1
                confidence_score = probs[1] if prediction == 1 and len(probs) > 1 else probs[0]
                confidence = f"{confidence_score:.2%}"

            diagnosis_result = "Positive" if prediction == 1 else "Negative"

            st.session_state['prediction_result'] = {
                'title': disease_name,
                'inputs': inputs, # Original string inputs for the report
                'diagnosis': diagnosis_result,
                'confidence': confidence
            }

            if diagnosis_result == "Positive":
                st.error(f"Diagnosis: {diagnosis_result} (Confidence: {confidence})")
            else:
                st.success(f"Diagnosis: {diagnosis_result} (Confidence: {confidence})")
            st.balloons()
            # Call PDF report generation
            if 'prediction_result' in st.session_state:
                render_pdf_report(st.session_state['prediction_result'])


        except Exception as e:
            st.error(f"Prediction Error: {e}")
            # st.write("Data sent to model:", data_to_predict) # For debugging

def render_pdf_report(result_data):
    st.markdown("---")
    st.markdown("#### Diagnosis Report")
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
    for k, v_original in result_data['inputs'].items(): # Use original inputs
        display_key_pdf = feature_fullforms.get(result_data['title'], {}).get(k, k.replace("_", " ").capitalize())
        pdf.multi_cell(0, 6, f"{display_key_pdf}: {v_original}") # Display original value
    pdf.ln(5)

    pdf.set_font("Helvetica", 'I', 8)
    pdf.multi_cell(0, 5, "Disclaimer: This AI-generated report is for informational purposes only and is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.", align='C')

    try:
        # Output PDF to bytes, then use BytesIO for download button
        pdf_output_bytes = pdf.output(dest='S').encode('latin-1') # 'S' returns as string, encode to bytes
        pdf_buffer = io.BytesIO(pdf_output_bytes)

        st.download_button(
            label="📥 Download Report (PDF)",
            data=pdf_buffer,
            file_name=f"{result_data['title'].replace(' ', '_')}_Diagnosis_Report.pdf",
            mime="application/pdf",
            key="download_pdf_button"
        )
    except Exception as e:
        st.error(f"Could not generate PDF for download: {e}")


def render_diagnose_page():
    st.title("🔬 Disease Diagnosis Center")
    st.markdown("Select a specific disease model for a detailed diagnosis using patient data.")
    st.markdown("---")

    model_options = list(MODEL_MAPPING.keys())
    if not model_options:
        st.warning("No disease models are currently configured.")
        return

    # Use columns for better layout of selectbox and description
    col1, col2 = st.columns([1, 2])
    with col1:
        selected_disease_key = st.selectbox(
            "Choose a Disease Model:",
            model_options,
            index=0,
            key="diagnose_disease_select"
        )
    with col2:
        st.caption(f"You are about to use the AI model for **{MODEL_MAPPING[selected_disease_key]['name']}**.")

    if selected_disease_key:
        with st.container(): # Class="card" can be applied via markdown if needed
            render_prediction_form_page(selected_disease_key)


def render_chatbot_page():
    st.title("💬 IntelliMed AI Chatbot")
    if not GEMINI_CONFIGURED or not gemini_llm_model:
        st.warning("AI Chatbot is currently unavailable due to configuration issues. Please contact support.")
        return

    st.markdown("Ask general health-related questions or get information. I am not a substitute for a doctor.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = load_chat_history()

    # Display chat messages
    chat_display_container = st.container()
    with chat_display_container:
        chat_html = "<div class='chat-box'>"
        for role, msg_text in st.session_state.chat_history:
            bubble_class = "user-bubble" if role == "user" else "bot-bubble"
            container_class = "user-container" if role == "user" else "bot-container"
            icon_html = "<span class='icon'>🧑‍💻</span>" if role == "user" else "<span class='icon'>🤖</span>"
            chat_html += f"<div class='message-container {container_class}'><div class='message-bubble {bubble_class}'>{icon_html}<div>{msg_text}</div></div></div>"
        chat_html += "</div>"
        st.markdown(chat_html, unsafe_allow_html=True)


    user_input = st.chat_input("Ask me anything...", key="chatbot_user_input")

    if user_input:
        st.session_state.chat_history.append(("user", user_input))
        with st.spinner("🤖 Thinking..."):
            try:
                response = gemini_llm_model.generate_content(user_input)
                bot_reply = response.text
            except Exception as e:
                bot_reply = f"Sorry, I encountered an error: {e}"
        st.session_state.chat_history.append(("bot", bot_reply))
        save_chat_history(st.session_state.chat_history)
        st.rerun()


# --- Main App Logic: Route to the correct page ---
# Each page rendering function will be wrapped by the page_container div
st.markdown("<div class='page-container'>", unsafe_allow_html=True) # Start of main content area



if current_page == "home":
    render_home_page()
elif current_page == "diagnose":
    render_diagnose_page()
elif current_page == "chatbot":
    render_chatbot_page()
else:
    st.error("Page not found. Please use the navigation.")
    if st.button("Go to Home"):
        st.query_params["page"] = "home"
        st.rerun()

st.markdown("</div>", unsafe_allow_html=True) # End of main content area