# import streamlit as st
# import joblib
# import numpy as np

# # Load pre-trained models and scaler
# model = joblib.load('./models/diabetes_model.pkl')
# scaler = joblib.load('./models/diabetes_scaler.pkl')
# kidneyModel = joblib.load('./models/kidneyDisease_model.pkl')

# # Diabetes feature names
# DIABETES_FEATURES = [
#     'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
#     'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
# ]

# # ---- CSS Styling ----
# st.markdown("""
#     <style>
#     /* Hide default Streamlit header/footer */
#     #MainMenu, footer, header {visibility: hidden;}

#     /* Custom Sidebar */
#     [data-testid="stSidebar"] {
#         background: linear-gradient(to bottom, #0f172a, #1e293b);
#         color: white;
#         padding-top: 1.5rem;
#         padding-left: 1rem;
#         padding-right: 1rem;
#     }

#     /* Profile Image Styling */
#     .profile-pic {
#         display: block;
#         margin: 0 auto;
#         margin-bottom: 10px;
#         width: 75px;
#         height: 75px;
#         border-radius: 50%;
#         object-fit: cover;
#         border: 2px solid #10b981;
#         box-shadow: 0 0 8px rgba(16,185,129,0.4);
#     }

#     /* Title below logo */
#     .sidebar-title {
#         text-align: center;
#         font-size: 20px;
#         font-weight: 600;
#         margin-top: 5px;
#         margin-bottom: 25px;
#         color: #10b981;
#     }

#     /* Vertical radio buttons */
#     .stRadio > div {
#         flex-direction: column;
#     }

#     /* Radio button labels */
#     label[data-baseweb="radio"] {
#         background-color: #1f2937;
#         border-radius: 8px;
#         padding: 10px 14px;
#         margin-bottom: 10px;
#         font-weight: 500;
#         border: 1px solid transparent;
#         transition: all 0.3s ease;
#         color: black;
#         text-align: center;
#         font-size: 15px;
#     }

#     /* Hover effect */
#     label[data-baseweb="radio"]:hover {
#         border: 1px solid #10b981;
#         background-color: #374151;
#         color: #10b981;
#     }

#     /* Selected tab */
#     input[type="radio"]:checked + div > div {
#         color: #10b981 !important;
#         font-weight: bold;
#     }
            
#     input[type="radio"]:not(:checked) + div > div {
#         color: white !important;
#         font-weight: bold;
#     }
#     </style>
# """, unsafe_allow_html=True)

# # ---- Sidebar ----
# with st.sidebar:
#     st.markdown('<img src="https://upload.wikimedia.org/wikipedia/commons/a/ab/Logo_TV_2015.png" class="profile-pic">', unsafe_allow_html=True)
#     st.markdown('<div class="sidebar-title">IntekkiMed</div>', unsafe_allow_html=True)
#     st.sidebar.title("📋 Navigation")

#     # Tabs/Menu
#     page = st.radio(" ", ["🤖 ChatBot", "📖 Diabetes", "🛠️ Heart Disease", "Breast Cancer", "Hypertension", "Kidney Disease", "Liver Disease", "Lung Disease"])

# # ---- Main Area ----
# # Ensure the page title can handle cases without spaces
# page_name = page.split(" ", 1)[1] if " " in page else page
# st.title(page_name + " Page")

# if "Diabetes" in page:
#     # st.subheader("🩺 Compact Diabetes Prediction Form")

#     with st.form("diabetes_form"):
#         # Use columns for compact layout
#         col1, col2 = st.columns(2)

#         with col1:
#             pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, step=1, format="%d")
#             blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=180, step=1)
#             insulin = st.number_input("Insulin", min_value=0, max_value=900, step=1)
#             diabetes_pedigree = st.number_input("Pedigree Function", min_value=0.0, max_value=2.5, step=0.01)

#         with col2:
#             glucose = st.number_input("Glucose", min_value=0, max_value=300, step=1)
#             skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, step=1)
#             bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, step=0.1)
#             age = st.number_input("Age", min_value=1, max_value=120, step=1)

#         # Action Buttons
#         col3, col4, col5 = st.columns([1, 1, 2])
#         with col3:
#             submit = st.form_submit_button("🔍 Predict")
#         with col4:
#             reset = st.form_submit_button("🔄 Reset")
#         with col5:
#             help_btn = st.form_submit_button("❓ Help")

#     if submit:
#         input_features = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]])
#         scaled_features = scaler.transform(input_features)
#         prediction = model.predict(scaled_features)
#         result = "Positive" if prediction[0] == 1 else "Negative"
#         st.success(f"🧾 Prediction: You are likely **{result}** for diabetes.")

#     elif help_btn:
#         st.info("👉 Fill in all details accurately. Glucose level above 120 is considered risky in this example.")

#     elif reset:
#         st.warning("🔁 Please refresh the page to reset all values.")











# import joblib
# import numpy as np
# import streamlit as st

# # Load models and scalers
# diabetes_model = joblib.load('./models/diabetes_model.pkl')
# diabetes_scaler = joblib.load('./models/diabetes_scaler.pkl')

# kidney_model = joblib.load('./models/kidneyDisease_model.pkl')

# heart_model = joblib.load('models/heartDisease_randomForest_model.pkl')
# heart_scaler = joblib.load('models/heartDisease_scaler.pkl')

# hypertension_model = joblib.load('models/hypertension_model.pkl')

# breast_model = joblib.load('./models/breastCancer_randomForest_model.pkl')

# lung_model = joblib.load('./models/lungCancer_XGBClassifier_model.pkl')

# liver_model = joblib.load('./models/liverDisease_rf_model.pkl')

# # Feature lists
# DIABETES_FEATURES = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
#                      'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

# KIDNEY_FEATURES = ['age', 'bp', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr',
#                    'bu', 'sc', 'pot', 'wc', 'htn', 'dm', 'cad', 'pe', 'ane']

# BREAST_FEATURES = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
#                    'smoothness_mean', 'compactness_mean', 'concavity_mean',
#                    'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean']

# LUNG_FEATURE_ORDER = ['GENDER', 'AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY',
#                       'PEER_PRESSURE', 'CHRONIC_DISEASE', 'FATIGUE', 'ALLERGY',
#                       'WHEEZING', 'ALCOHOL_CONSUMING', 'COUGHING',
#                       'SHORTNESS_OF_BREATH', 'SWALLOWING_DIFFICULTY', 'CHEST_PAIN']

# # Streamlit App UI
# st.title("Health Prediction System")

# st.sidebar.header("Select Prediction Model")
# model_choice = st.sidebar.selectbox(
#     "Choose a model", 
#     ["Diabetes", "Kidney Disease", "Heart Disease", "Hypertension", "Breast Cancer", 
#      "Lung Cancer", "Liver Disease"]
# )

# # Diabetes Prediction
# if model_choice == "Diabetes":
#     st.subheader("Diabetes Prediction")
#     features = {}
#     for feature in DIABETES_FEATURES:
#         features[feature] = st.number_input(f"Enter {feature}", min_value=0.0, max_value=200.0, step=0.1)

#     if st.button('Predict Diabetes'):
#         data = np.array([list(features.values())])
#         scaled_data = diabetes_scaler.transform(data)
#         prediction = diabetes_model.predict(scaled_data)
#         probability = diabetes_model.predict_proba(scaled_data)[0][1]
#         diagnosis = "Diabetes Detected" if prediction[0] == 1 else "No Diabetes"
#         st.write(f"Diagnosis: {diagnosis} (Confidence: {probability:.2%})")

# # Kidney Disease Prediction
# elif model_choice == "Kidney Disease":
#     st.subheader("Kidney Disease Prediction")
#     features = {}
#     for feature in KIDNEY_FEATURES:
#         features[feature] = st.number_input(f"Enter {feature}", min_value=0.0, max_value=200.0, step=0.1)

#     if st.button('Predict Kidney Disease'):
#         data = np.array([list(features.values())])
#         prediction = kidney_model.predict(data)
#         probability = kidney_model.predict_proba(data)[0][1]
#         diagnosis = "Kidney Disease Present" if prediction[0] == 1 else "Healthy"
#         st.write(f"Diagnosis: {diagnosis} (Confidence: {probability:.2%})")

# # Heart Disease Prediction
# elif model_choice == "Heart Disease":
#     st.subheader("Heart Disease Prediction")
#     features = [st.number_input(f"Enter {feature}", min_value=0.0, max_value=200.0, step=0.1) for feature in DIABETES_FEATURES]
    
#     if st.button('Predict Heart Disease'):
#         final_features = np.array(features).reshape(1, -1)
#         scaled_features = heart_scaler.transform(final_features)
#         prediction = heart_model.predict(scaled_features)
#         result = "Heart Disease Detected!" if prediction[0] == 1 else "No Heart Disease Detected"
#         st.write(result)

# # Hypertension Prediction
# elif model_choice == "Hypertension":
#     st.subheader("Hypertension Prediction")
#     features = {
#         "age": st.number_input("Enter Age", min_value=0, max_value=150),
#         "bmi": st.number_input("Enter BMI", min_value=10.0, max_value=100.0),
#         "systolic_bp": st.number_input("Enter Systolic BP", min_value=60, max_value=200),
#         "diastolic_bp": st.number_input("Enter Diastolic BP", min_value=40, max_value=120),
#         "cholesterol": st.number_input("Enter Cholesterol", min_value=100, max_value=400)
#     }
    
#     if st.button('Predict Hypertension'):
#         data = np.array([list(features.values())])
#         prediction = hypertension_model.predict(data)
#         result = "Hypertensive" if prediction[0] == 1 else "Non-Hypertensive"
#         st.write(result)

# # Breast Cancer Prediction
# elif model_choice == "Breast Cancer":
#     st.subheader("Breast Cancer Prediction")
#     features = {}
#     for feature in BREAST_FEATURES:
#         features[feature] = st.number_input(f"Enter {feature}", min_value=0.0, max_value=1000.0, step=0.1)

#     if st.button('Predict Breast Cancer'):
#         data = np.array([list(features.values())])
#         prediction = breast_model.predict(data)
#         probability = breast_model.predict_proba(data)[0][1]
#         diagnosis = "Malignant" if prediction[0] == 1 else "Benign"
#         st.write(f"Diagnosis: {diagnosis} (Confidence: {probability:.2%})")

# # Lung Cancer Prediction
# elif model_choice == "Lung Cancer":
#     st.subheader("Lung Cancer Prediction")
#     features = [st.selectbox(f"Enter {feature}", ['Male', 'Female']) if 'GENDER' in feature else 
#                 st.number_input(f"Enter {feature}", min_value=0.0, max_value=200.0) for feature in LUNG_FEATURE_ORDER]
    
#     if st.button('Predict Lung Cancer'):
#         prediction = lung_model.predict([features])[0]
#         result = "Positive (लंग कैंसर का खतरा है)" if prediction == 1 else "Negative (लंग कैंसर का खतरा नहीं है)"
#         st.write(result)

# # Liver Disease Prediction
# elif model_choice == "Liver Disease":
#     st.subheader("Liver Disease Prediction")
#     features = {
#         "Age": st.number_input("Enter Age", min_value=0, max_value=100),
#         "Gender": st.selectbox("Enter Gender", ["Male", "Female"]),
#         "Total_Bilirubin": st.number_input("Enter Total Bilirubin", min_value=0.0, max_value=100.0),
#         "Alkaline_phosphate": st.number_input("Enter Alkaline Phosphate", min_value=0.0, max_value=1000.0),
#         "Alamine_Aminotransferace": st.number_input("Enter Alamine Aminotransferace", min_value=0.0, max_value=1000.0),
#         "Aspartate_Amino": st.number_input("Enter Aspartate Aminotransferace", min_value=0.0, max_value=1000.0),
#         "Protien": st.number_input("Enter Protein", min_value=0.0, max_value=100.0),
#         "Albumin": st.number_input("Enter Albumin", min_value=0.0, max_value=100.0),
#         "Albumin_Globulin_ratio": st.number_input("Enter Albumin/Globulin Ratio", min_value=0.0, max_value=5.0)
#     }

#     if st.button('Predict Liver Disease'):
#         data = np.array([list(features.values())])
#         prediction = liver_model.predict(data)[0]
#         result = "Liver Disease Detected" if prediction == 1 else "No Liver Disease"
#         st.write(result)
















# import joblib
# import numpy as np
# import streamlit as st

# # Load models and scalers
# diabetes_model = joblib.load('./models/diabetes_model.pkl')
# diabetes_scaler = joblib.load('./models/diabetes_scaler.pkl')

# kidney_model = joblib.load('./models/kidneyDisease_model.pkl')

# heart_model = joblib.load('models/heartDisease_randomForest_model.pkl')
# heart_scaler = joblib.load('models/heartDisease_scaler.pkl')

# hypertension_model = joblib.load('models/hypertension_model.pkl')

# breast_model = joblib.load('./models/breastCancer_randomForest_model.pkl')

# lung_model = joblib.load('./models/lungCancer_XGBClassifier_model.pkl')

# liver_model = joblib.load('./models/liverDisease_rf_model.pkl')

# # Feature lists
# DIABETES_FEATURES = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
#                      'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

# KIDNEY_FEATURES = ['age', 'bp', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr',
#                    'bu', 'sc', 'pot', 'wc', 'htn', 'dm', 'cad', 'pe', 'ane']

# BREAST_FEATURES = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
#                    'smoothness_mean', 'compactness_mean', 'concavity_mean',
#                    'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean']

# LUNG_FEATURE_ORDER = ['GENDER', 'AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY',
#                       'PEER_PRESSURE', 'CHRONIC_DISEASE', 'FATIGUE', 'ALLERGY',
#                       'WHEEZING', 'ALCOHOL_CONSUMING', 'COUGHING',
#                       'SHORTNESS_OF_BREATH', 'SWALLOWING_DIFFICULTY', 'CHEST_PAIN']

# # App UI
# st.title("Health Prediction System")
# st.sidebar.header("Select Prediction Model")

# model_choice = st.sidebar.selectbox(
#     "Choose a model", 
#     ["Diabetes", "Kidney Disease", "Heart Disease", "Hypertension", "Breast Cancer", 
#      "Lung Cancer", "Liver Disease"]
# )

# # Form wrappers with styled buttons
# def styled_button_form(title, input_fields, predict_callback):
#     with st.form(key=title):
#         st.subheader(title)
#         features = {}
#         for field in input_fields:
#             if isinstance(field, tuple):  # For selectboxes
#                 features[field[0]] = st.selectbox(f"{field[0]}", field[1])
#             else:
#                 features[field] = st.number_input(f"{field}", min_value=0.0, step=0.1)
#         submit = st.form_submit_button(f"🚀 Predict {title}")
#         if submit:
#             predict_callback(features)

# # Models
# if model_choice == "Diabetes":
#     styled_button_form("Diabetes Prediction", DIABETES_FEATURES, lambda f: (
#         st.write(f"Diagnosis: {'Diabetes Detected' if (p := diabetes_model.predict(diabetes_scaler.transform([list(f.values())])))[0] == 1 else 'No Diabetes'} "
#                  f"(Confidence: {diabetes_model.predict_proba(diabetes_scaler.transform([list(f.values())]))[0][1]:.2%})")
#     ))

# elif model_choice == "Kidney Disease":
#     styled_button_form("Kidney Disease Prediction", KIDNEY_FEATURES, lambda f: (
#         st.write(f"Diagnosis: {'Kidney Disease Present' if (p := kidney_model.predict([list(f.values())]))[0] == 1 else 'Healthy'} "
#                  f"(Confidence: {kidney_model.predict_proba([list(f.values())])[0][1]:.2%})")
#     ))

# elif model_choice == "Heart Disease":
#     styled_button_form("Heart Disease Prediction", DIABETES_FEATURES, lambda f: (
#         st.write("Heart Disease Detected!" if heart_model.predict(heart_scaler.transform([list(f.values())]))[0] == 1 else "No Heart Disease Detected")
#     ))

# elif model_choice == "Hypertension":
#     styled_button_form("Hypertension Prediction", ['age', 'bmi', 'systolic_bp', 'diastolic_bp', 'cholesterol'], lambda f: (
#         st.write("Hypertensive" if hypertension_model.predict([list(f.values())])[0] == 1 else "Non-Hypertensive")
#     ))

# elif model_choice == "Breast Cancer":
#     styled_button_form("Breast Cancer Prediction", BREAST_FEATURES, lambda f: (
#         st.write(f"Diagnosis: {'Malignant' if (p := breast_model.predict([list(f.values())]))[0] == 1 else 'Benign'} "
#                  f"(Confidence: {breast_model.predict_proba([list(f.values())])[0][1]:.2%})")
#     ))

# elif model_choice == "Lung Cancer":
#     styled_button_form("Lung Cancer Prediction", [
#         ('GENDER', ['Male', 'Female'])
#     ] + LUNG_FEATURE_ORDER[1:], lambda f: (
#         st.write("Positive (लंग कैंसर का खतरा है)" if lung_model.predict([[1 if f['GENDER'] == 'Male' else 0] + [f[k] for k in LUNG_FEATURE_ORDER[1:]]])[0] == 1 
#                  else "Negative (लंग कैंसर का खतरा नहीं है)")
#     ))

# elif model_choice == "Liver Disease":
#     styled_button_form("Liver Disease Prediction", [
#         ('Gender', ['Male', 'Female']),
#         'Age', 'Total_Bilirubin', 'Alkaline_phosphate', 'Alamine_Aminotransferace',
#         'Aspartate_Amino', 'Protien', 'Albumin', 'Albumin_Globulin_ratio'
#     ], lambda f: (
#         st.write("Liver Disease Detected" if liver_model.predict([[1 if f['Gender'] == 'Male' else 0] + [f[k] for k in list(f) if k != 'Gender']])[0] == 1
#                  else "No Liver Disease")
#     ))












import streamlit as st
import numpy as np
import joblib

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




DIABETES_FEATURES = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
KIDNEY_FEATURES = ['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc']
HEART_FEATURES = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach']
HYPERTENSION_FEATURES = ['age', 'bmi', 'smoking', 'exercise', 'alcohol']
BREAST_FEATURES = ['mean_radius', 'mean_texture', 'mean_perimeter', 'mean_area', 'mean_smoothness']
LUNG_FEATURES = ['age', 'smoking', 'yellow_fingers', 'anxiety', 'chronic_disease']
LIVER_FEATURES = ['Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin', 'Alkaline_Phosphotase']

import streamlit as st
import numpy as np
import joblib

# Load models & scalers (same as before)
# ... (keep your model loading code unchanged)

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

    .profile-pic {
        display: block;
        margin: 0 auto;
        margin-bottom: 10px;
        width: 75px;
        height: 75px;
        border-radius: 50%;
        object-fit: cover;
        border: 2px solid #10b981;
        box-shadow: 0 0 8px rgba(16,185,129,0.4);
    }

    .topnav {
        background-color: #0f172a;
        overflow: hidden;
        padding: 10px 20px;
        margin-bottom: 20px;
        border-radius: 10px 0 0 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        top: 5%;
        position: fixed;
        righ: 0;    
        width: 62%;
        z-index: 1000;
    }

    .topnav h1 {
        color: #10b981;
        float: left;
        font-size: 24px;
        margin: 0;
        padding-top: 5px;
        top: 0;
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
    </style>
""", unsafe_allow_html=True)

# ---- NAVIGATION ----
query_params = st.query_params
page = query_params.get("page", "home")

# ---- Navbar with routing ----
st.markdown(f"""
<div class="topnav">
    <h1>🩺 Health AI</h1>
    <a href="?page=about">About</a>
    <a href="?page=home">Home</a>
</div>
""", unsafe_allow_html=True)

# ---- Sidebar (show only on Home page) ----
if page == "home":
    st.sidebar.image("https://i.ibb.co/7QpKsCX/user.png", width=75)
    st.sidebar.markdown('<div class="sidebar-title">Health Predictor</div>', unsafe_allow_html=True)
    st.sidebar.header("Choose Disease Model")
    model_choice = st.sidebar.selectbox("Select Model", [
        "Diabetes", "Kidney Disease", "Heart Disease", "Hypertension",
        "Breast Cancer", "Lung Cancer", "Liver Disease"])
    
      # Tabs/Menu
    st.sidebar.radio(" ", ["🤖 ChatBot", "🩺 Services", "📬 Contact", "⚙️ Settings"])


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
    
    **Built with 💚 by Aashu Karn**

    ---
    ### Technologies Used:
    - Python
    - Streamlit
    - Scikit-learn
    - Joblib
    - CSS styling
    """)
    st.info("🔁 Click **Home** on the top navbar to return.")

# ---- Page: Home ----
if page == "home":
    st.title("Medical Diagnosis with Machine Learning")

    def prediction_form(title, features, model, scaler):
        st.subheader(title)
        with st.form(f"{title.lower()}_form"):
            inputs = [st.number_input(f"{feature}", key=f"{title}_{feature}") for feature in features]
            submit = st.form_submit_button(f"🔍 Predict")

        if submit:
            data = np.array([inputs])
            scaled = scaler.transform(data)
            pred = model.predict(scaled)[0]
            prob = model.predict_proba(scaled)[0][1] if hasattr(model, 'predict_proba') else 0.0
            st.success(f"🩺 Diagnosis: {'Positive' if pred == 1 else 'Negative'} (Confidence: {prob:.2%})")

    # Model Switcher
    if model_choice == "Diabetes":
        prediction_form("Diabetes", DIABETES_FEATURES, diabetes_model, diabetes_scaler)
    elif model_choice == "Kidney Disease":
        prediction_form("Kidney Disease", KIDNEY_FEATURES, kidney_model)
    elif model_choice == "Heart Disease":
        prediction_form("Heart Disease", HEART_FEATURES, heart_model, heart_scaler)
    elif model_choice == "Hypertension":
        prediction_form("Hypertension", HYPERTENSION_FEATURES, hypertension_model, hypertension_scaler)
    elif model_choice == "Breast Cancer":
        prediction_form("Breast Cancer", BREAST_FEATURES, breast_model, breast_scaler)
    elif model_choice == "Lung Cancer":
        prediction_form("Lung Cancer", LUNG_FEATURES, lung_model, lung_scaler)
    elif model_choice == "Liver Disease":
        prediction_form("Liver Disease", LIVER_FEATURES, liver_model, liver_scaler)



