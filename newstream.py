import streamlit as st
import numpy as np
import joblib

# Load models & scalers (same as before)
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

# CSS styles for navbar and sidebar
st.markdown("""
    <style>
    #MainMenu, footer, header {visibility: hidden;}
    [data-testid="stSidebar"] {
        background: linear-gradient(to bottom, #0f172a, #1e293b);
        color: white;
        padding-top: 1.5rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    .sidebar-title {
        text-align: center;
        font-size: 20px;
        font-weight: 600;
        margin-top: 5px;
        margin-bottom: 25px;
        color: #10b981;
    }
    .topnav {
        background-color: #0f172a;
        overflow: hidden;
        padding: 10px 20px;
        margin-bottom: 20px;
        border-radius: 10px 0 0 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        top: 3%;
        position: fixed;
        right: 0;
        width: 62%;
        z-index: 1000;
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
        text-align: center;
        padding: 6px 14px;
        text-decoration: none;
        font-size: 16px;
        border-radius: 6px;
        margin-left: 10px;
        background-color: #1e293b;
        transition: 0.3s;
    }
    .topnav a:hover {
        background-color: #10b981;
        color: black;
    }

    input[type="radio"]:checked + div > div {
        color: #10b981 !important;
        font-weight: bold;
        font-size: 18px !important;
    }

    input[type="radio"]:not(:checked) + div > div {
        color: white !important;
        font-weight: bold;
        font-size: 18px !important;
    }

    </style>
""", unsafe_allow_html=True)

# ---- NAVIGATION ----
query_params = st.query_params
page = query_params.get("page", "home")

# ---- Navbar with routing ----
st.markdown(f"""
<div class="topnav">
    <h1>🩺 IntelliMed</h1>
    <a href="?page=about">About</a>
    <a href="?page=home">Home</a>
</div>
""", unsafe_allow_html=True)

# ---- Sidebar (show only on Home page) ----
if page == "home":
    st.sidebar.image("./static/logo.png", width=200)
    st.sidebar.title("📋 Navigation")

    rd = st.sidebar.radio("", ["🩺 Disease Diagnose", "🤖 ChatBot", "🩺 Services", "📬 Contact", "⚙️ Settings"])

    if rd == "🤖 ChatBot":
        choose_ai = st.sidebar.selectbox("Choose AI", ["ChatBot", "Voice Assistant"])
        st.sidebar.info("Chatbot is under development. Stay tuned!")
    elif rd == "🩺 Disease Diagnose":
        model_choice = st.sidebar.selectbox("Select Option", [
            "Choose Disease", "Diabetes", "Kidney Disease", "Heart Disease", "Hypertension",
            "Breast Cancer", "Lung Cancer", "Liver Disease"
        ])
        st.sidebar.info("Choose a disease model to predict.")
    elif rd == "🩺 Services":
        st.sidebar.info("Services are under development. Stay tuned!")
    elif rd == "📬 Contact":
        st.sidebar.info("Contact us at:")
        st.sidebar.markdown("Email:")
        st.sidebar.markdown("Phone:")
        st.sidebar.markdown("LinkedIn:")
    elif rd == "⚙️ Settings":
        st.sidebar.info("Settings are under development. Stay tuned!")

    st.button("Refresh", key="refresh_button")

# ---- Page: About ----
if page == "about":
    st.title("👨‍⚕️ About This Project")
    st.markdown("""
    This is a **Smart AI Medical Diagnosis App** developed using **Streamlit** and **Machine Learning** models.
    It helps predict the risk of diseases like:
    - 🧬 Diabetes
    - 🧠 Brain & Heart Diseases
    - 🫁 Lung Cancer
    - 🏥 Kidney & Liver Disorders
    - 🧪 Breast Cancer
    """)
    st.info("🔁 Click **Home** on the top navbar to return.")
    st.balloons()

# ---- Page: Home ----
if page == "home":
    st.title("Medical Diagnosis with Machine Learning")

    # def prediction_form(title, features, model, scaler=None):
    #     st.subheader(title)
        
    #     # Using columns to arrange form fields horizontally
    #     with st.form(f"{title.lower()}_form"):
    #         cols = st.columns(3)  # Create two columns for better alignment
            
    #         inputs = []
    #         for i, feature in enumerate(features):
    #             # Place input fields in columns
    #             if feature in ['age', 'bmi', 'Glucose']:  # numeric features
    #                 inputs.append(cols[i % 3].slider(f"{feature}", 0, 200, 50))
    #             else:  # categorical features
    #                 inputs.append(cols[i % 3].radio(f"{feature}", ['Yes', 'No'], index=0))

    #         submit = st.form_submit_button(f"🔍 Predict")
    #         reset = st.form_submit_button(f"🔄 Reset")
    #         help = st.form_submit_button(f"❓ Help")

    #     if reset:
    #         st.refresh()
    #         st.success("Form reset!")
    #     if help:
    #         st.info(f"Help for {title}:")
    #         st.markdown(f"Please provide the following details:")
    #         for feature in features:
    #             st.markdown(f"- **{feature}**: Description of the feature.")


    def prediction_form(title, features, model, scaler=None):
        st.subheader(title)

        with st.form(f"{title.lower()}_form"):
            cols = st.columns(3)  # Split into 3 columns for cleaner layout
            inputs = []

            for i, feature in enumerate(features):
                if feature.lower() in ['age', 'bmi', 'glucose']:  # Adjust for consistent casing
                    val = cols[i % 3].slider(f"{feature}", 0, 200, 50)
                    inputs.append(val)
                else:
                    val = cols[i % 3].radio(f"{feature}", ['Yes', 'No'], index=0)
                    inputs.append(1 if val == 'Yes' else 0)  # Convert to numeric

            col_submit, col_reset, col_help = st.columns([1, 1, 1])
            submit = col_submit.form_submit_button("🔍 Predict")
            reset = col_reset.form_submit_button("🔄 Reset")
            help = col_help.form_submit_button("❓ Help")

        if submit:
            data = np.array([inputs], dtype=float)
            if scaler:
                data = scaler.transform(data)
            prediction = model.predict(data)[0]
            confidence = model.predict_proba(data)[0][1] if hasattr(model, 'predict_proba') else 0.0
            st.success(f"🩺 Diagnosis: {'Positive' if prediction == 1 else 'Negative'} (Confidence: {confidence:.2%})")
            st.balloons()

        elif reset:
            st.warning("Form has been reset. Please re-enter the values.")

        elif help:
            st.info("👉 Fill out the form with patient details. Select appropriate values. Click Predict to get a diagnosis.")


            if submit:
                data = np.array([inputs])
                if scaler:
                    scaled = scaler.transform(data)
                else:
                    scaled = data
                pred = model.predict(scaled)[0]
                prob = model.predict_proba(scaled)[0][1] if hasattr(model, 'predict_proba') else 0.0
                st.success(f"🩺 Diagnosis: {'Positive' if pred == 1 else 'Negative'} (Confidence: {prob:.2%})")
                st.balloons()

    if model_choice == "Choose Disease":
        st.info("Please select a disease model from the sidebar.")

    elif model_choice == "Diabetes":
        prediction_form("Diabetes", DIABETES_FEATURES, diabetes_model, diabetes_scaler)
    elif model_choice == "Kidney Disease":
        prediction_form("Kidney Disease", KIDNEY_FEATURES, kidney_model)
    elif model_choice == "Heart Disease":
        prediction_form("Heart Disease", HEART_FEATURES, heart_model, heart_scaler)
    elif model_choice == "Hypertension":
        prediction_form("Hypertension", HYPERTENSION_FEATURES, hypertension_model)
    elif model_choice == "Breast Cancer":
        prediction_form("Breast Cancer", BREAST_FEATURES, breast_model)
    elif model_choice == "Lung Cancer":
        prediction_form("Lung Cancer", LUNG_FEATURES, lung_model)
    elif model_choice == "Liver Disease":
        prediction_form("Liver Disease", LIVER_FEATURES, liver_model)
    elif model_choice == "ChatBot":
        st.subheader("🗣️ Chat with AI")
        user_input = st.text_input("Ask me anything!")
        if user_input:
            st.write(f"Chatbot says: {user_input[::-1]} (This is a placeholder response, implement a real chatbot here!)")
