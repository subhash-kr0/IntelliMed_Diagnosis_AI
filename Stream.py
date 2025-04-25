import streamlit as st
import numpy as np
import joblib

import google.generativeai as genai
import os

st.set_page_config(page_title="ChatBot", layout="centered")

# Fetch the Gemini API key securely
gemini_key = st.secrets["api_keys"]["gemini"]
openai_key = st.secrets["api_keys"]["openai"]

# Set up the API client for Gemini using the key
genai.configure(api_key=gemini_key)
model = genai.GenerativeModel(model_name="models/gemini-2.5-flash-preview-04-17") 



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
            

        .topnav1 {
        background-color: #0f172a;
        overflow: hidden;
        padding: 10px 20px;
        margin-bottom: 20px;
        border-radius: 10px 10px 10px 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        # top: 3%;
        position: fixed;
        # width: 40%;
        # height: 100px;
        z-index: 1000;
        bottom: 0;
    }
            
        .topnav1 a {
        float: right;
        color: white;
        text-align: center;
        padding: 6px 14px;
        text-decoration: none;
        font-size: 12px;
        border-radius: 6px;
        margin-left: 10px;
        background-color: #1e293b;
        transition: 0.3s;
        
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

            
    
       input[type="radio"]:checked + div > div {
        # background-color: #FFFFFF !important;
        color: #10b981 !important;
        font-weight: bold;
        font-size: 20px !important;
    }


    input[type="radio"]:not(:checked) + div > div {
        # background-color: #FFFFFF !important;
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
    # st.sidebar.image("https://i.ibb.co/7QpKsCX/user.png", width=75)
    st.sidebar.image("./static/logo.png", width=200)
    # st.sidebar.markdown('<div class="sidebar-title">IntelliMed</div>', unsafe_allow_html=True)
    st.sidebar.title("📋 Navigation")
    # st.sidebar.header("Choose Disease Model or Chatbot")

    
    # Add model choice dropdown and chatbot option
    # model_choice = st.sidebar.selectbox("Select Option", [
    #     "Choose Model", "Diabetes", "Kidney Disease", "Heart Disease", "Hypertension",
    #     "Breast Cancer", "Lung Cancer", "Liver Disease", "ChatBot"
    # ])

    model_choice = []
    # Tabs/Menu
    rd = st.sidebar.radio("", ["🤖 ChatBot","🩺 Disease Diagnose", "🩺 Services", "📬 Contact", "⚙️ Settings"])
        

    # if rd == "🤖 ChatBot":
    #     choose_ai = st.sidebar.selectbox("Choose AI", ["ChatBot", "Voice Assistant"])
        # st.sidebar.info("Chatbot is under development. Stay tuned!")
    def get_bot_response(message):
        response = model.generate_content(message)
        return response.text

    if rd == "🤖 ChatBot":
        choose_ai = st.sidebar.selectbox("Choose AI", ["ChatBot (Gemini)", "Voice Assistant"])
        
        # st.subheader("💬 IntelliMed ChatBot")

        # if "chat_history" not in st.session_state:
        #     st.session_state.chat_history = []

        # user_input = st.text_input("👨‍⚕️ Ask something medical...")

        # if user_input:
        #     st.session_state.chat_history.append(("You", user_input))
        #     bot_reply = get_bot_response(user_input)
        #     st.session_state.chat_history.append(("Bot", bot_reply))

        # for sender, msg in st.session_state.chat_history:
        #     if sender == "You":
        #         st.markdown(f"**🧑 You:** {msg}")
        #     else:
        #         st.markdown(f"**🤖 Bot:** {msg}")




        
        st.markdown("""
            <style>
            # .chat-box {
            #     height: 450px;
            #     overflow-y: auto;
            #     border: 2px solid #d3d3d3;
            #     border-radius: 12px;
            #     padding: 10px;
            #     background-color: #f0f2f6;
            #     margin-bottom: 10px;
            # }
            .message {
                padding: 8px 12px;
                border-radius: 8px;
                margin: 6px 0;
                max-width: 80%;
                word-wrap: break-word;
            }
            .user {
                background-color: #dcf8c6;
                align-self: flex-end;
                text-align: right;
                margin-left: auto;
            }
            .bot {
                background-color: #e2e3e5;
                align-self: flex-start;
                text-align: left;
                margin-right: auto;
            }
            .chat-container {
                display: flex;
                flex-direction: column;
            }
            </style>
        """, unsafe_allow_html=True)

        # --- Title ---
        st.markdown("<h2 style='text-align: center; color: #10b981;'>💬 IntelliMed AI Chat</h2>", unsafe_allow_html=True)

        # --- Session State ---
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # --- Chat Display Container ---
        with st.container():
            st.markdown("<div class='chat-box chat-container'>", unsafe_allow_html=True)
            for role, msg in st.session_state.chat_history:
                if role == "user":
                    st.markdown(f"<div class='message user'>🧑‍💻 {msg}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='message bot'>🤖 {msg}</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # --- Input Below Window ---
        user_input = st.chat_input("Type your message...")

        if user_input:
            st.session_state.chat_history.append(("user", user_input))
            with st.spinner("Thinking..."):
                bot_reply = model.generate_content(user_input).text
            st.session_state.chat_history.append(("bot", bot_reply))
            st.rerun()  


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
    

    # st.balloons()
    # st.button("Refresh", key="refresh_button")

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
    
    **Built with 💚 by Subhash Kumar**
    --- 
    ### Technologies Used:
    - Python
    - Streamlit
    - Scikit-learn
    - Joblib
    - CSS styling
    """)
    st.info("🔁 Click **Home** on the top navbar to return.")
    st.sidebar.image("./static/logo.png", width=200)
    st.sidebar.markdown('<div class="sidebar-title">IntelliMed</div>', unsafe_allow_html=True)
    st.sidebar.markdown("Email:")
    st.sidebar.markdown("Phone:")
    st.sidebar.markdown("LinkedIn:")
    st.sidebar.markdown("GitHub:")

    st.balloons()
    st.sidebar.info("This is a demo version. For full features, please contact the developer.")
    # st.sidebar.markdown('<div class="sidebar-title">IntelliMed</div>', unsafe_allow_html=True)

# ---- Page: Home ----
if page == "home":
    # st.title("Medical Diagnosis with Machine Learning")

    def prediction_form(title, features, model, scaler=None):
        st.subheader(title)
        with st.form(f"{title.lower()}_form"):
            inputs = [st.number_input(f"{feature}", key=f"{title}_{feature}") for feature in features]
            submit = st.form_submit_button(f"🔍 Predict")

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











# ---- NAVIGATION ----
query_params1 = st.query_params
page1 = query_params.get("page1", "home1")

# ---- Navbar with routing ----
st.markdown(f"""
<div class="topnav1">
    <a href="?page=about">About</a>
    <a href="?page=home">Home</a>
    <a href="?page=home">Project</a>
</div>
""", unsafe_allow_html=True)

# st.download_button(
#     label="Download Report",
#     data="This is a sample report.",
#     file_name="report.txt",
#     mime="text/plain",
# )











# st.markdown("""
#     <style>
#     .chat-window {
#         display: flex;
#         flex-direction: column-reverse;
#         height: 500px;
#         border: 1px solid #ccc;
#         border-radius: 12px;
#         padding: 10px;
#         overflow-y: auto;
#         background-color: #f8f9fa;
#         margin-bottom: 10px;
#     }
#     .user-message {
#         background-color: #DCF8C6;
#         border-radius: 10px;
#         padding: 8px 12px;
#         margin: 8px 0;
#         max-width: 75%;
#         align-self: flex-start;
#     }
#     .bot-message {
#         background-color: #E4E6EB;
#         border-radius: 10px;
#         padding: 8px 12px;
#         margin: 8px 0;
#         max-width: 75%;
#         align-self: flex-end;
#     }
#     </style>
# """, unsafe_allow_html=True)

# # --- Header / Title ---
# st.markdown("<h2 style='text-align:center; color:#10b981;'>🤖 IntelliMed - ChatBot</h2>", unsafe_allow_html=True)

# # --- Initialize Session State for Chat History ---
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []

# # --- Handle User Input ---
# user_input = st.chat_input("Type your message here...")

# if user_input:
#     st.session_state.chat_history.append(("user", user_input))
#     bot_response = model.generate_content(user_input).text
#     st.session_state.chat_history.append(("bot", bot_response))

# # --- Display Chat History in Reversed Order (Bottom-up) ---
# st.markdown("<div class='chat-window'>", unsafe_allow_html=True)
# for sender, message in reversed(st.session_state.chat_history):
#     if sender == "user":
#         st.markdown(f"<div class='user-message'><strong>🧑 You:</strong><br>{message}</div>", unsafe_allow_html=True)
#     else:
#         st.markdown(f"<div class='bot-message'><strong>🤖 Bot:</strong><br>{message}</div>", unsafe_allow_html=True)
# st.markdown("</div>", unsafe_allow_html=True)








# st.markdown("""
#     <style>
#     .chat-box {
#         height: 450px;
#         overflow-y: auto;
#         border: 2px solid #d3d3d3;
#         border-radius: 12px;
#         padding: 10px;
#         background-color: #f0f2f6;
#         margin-bottom: 10px;
#     }
#     .message {
#         padding: 8px 12px;
#         border-radius: 8px;
#         margin: 6px 0;
#         max-width: 80%;
#         word-wrap: break-word;
#     }
#     .user {
#         background-color: #dcf8c6;
#         align-self: flex-end;
#         text-align: right;
#         margin-left: auto;
#     }
#     .bot {
#         background-color: #e2e3e5;
#         align-self: flex-start;
#         text-align: left;
#         margin-right: auto;
#     }
#     .chat-container {
#         display: flex;
#         flex-direction: column;
#     }
#     </style>
# """, unsafe_allow_html=True)

# # --- Title ---
# st.markdown("<h2 style='text-align: center; color: #10b981;'>💬 IntelliMed AI Chat</h2>", unsafe_allow_html=True)

# # --- Session State ---
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []

# # --- Chat Display Container ---
# with st.container():
#     st.markdown("<div class='chat-box chat-container'>", unsafe_allow_html=True)
#     for role, msg in st.session_state.chat_history:
#         if role == "user":
#             st.markdown(f"<div class='message user'>🧑‍💻 {msg}</div>", unsafe_allow_html=True)
#         else:
#             st.markdown(f"<div class='message bot'>🤖 {msg}</div>", unsafe_allow_html=True)
#     st.markdown("</div>", unsafe_allow_html=True)

# # --- Input Below Window ---
# user_input = st.chat_input("Type your message...")

# if user_input:
#     st.session_state.chat_history.append(("user", user_input))
#     with st.spinner("Thinking..."):
#         bot_reply = model.generate_content(user_input).text
#     st.session_state.chat_history.append(("bot", bot_reply))
#     st.rerun()  