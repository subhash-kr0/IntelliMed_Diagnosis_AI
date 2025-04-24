import streamlit as st

# Page config
st.set_page_config(page_title="IntekkiMed", layout="wide")

# ---- Logo URL ----
logo_url = "https://upload.wikimedia.org/wikipedia/commons/a/ab/Logo_TV_2015.png"

# ---- Global CSS Styling ----
st.markdown("""
    <style>
        /* Hide default elements */
        #MainMenu, footer, header {visibility: hidden;}

        .block-container {
            padding-top: 0rem !important;
        }

        /* --- Top Header Navbar --- */
        .header-container {
            display: flex;
            justify-content: space-between;
            align-items: center;
            background-color: rgba(0, 0, 0, 0.09);
            padding: 10px 20px;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
            margin-bottom: 20px;
            flex-wrap: wrap;
        }

        .logo-container img {
            height: 50px;
        }

        .navbar {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
        }

        .navbar a {
            color: #fff;
            background-color: #FF00FF;
            border: 2px solid #FF00FF;
            padding: 8px 16px;
            border-radius: 8px;
            text-decoration: none;
            font-size: 16px;
            font-weight: 500;
            transition: 0.3s;
        }

        .navbar a:hover {
            background-color: #8B008B;
        }

        .navbar a.active {
            background-color: #1f77b4;
        }

        @media screen and (max-width: 600px) {
            .header-container {flex-direction: column; align-items: flex-start;}
            .navbar {justify-content: center; width: 100%;}
        }

        /* --- Sidebar Styling --- */
        [data-testid="stSidebar"] {
            background: linear-gradient(to bottom, #0f172a, #1e293b);
            color: white;
            padding-top: 1.5rem;
        }

        .profile-pic {
            display: block;
            margin: 0 auto 10px;
            width: 75px;
            height: 75px;
            border-radius: 50%;
            object-fit: cover;
            border: 2px solid #10b981;
            box-shadow: 0 0 8px rgba(16,185,129,0.4);
        }

        .sidebar-title {
            text-align: center;
            font-size: 20px;
            font-weight: 600;
            margin-bottom: 25px;
            color: #10b981;
        }

        .stRadio > div {flex-direction: column;}

        label[data-baseweb="radio"] {
            background-color: #1f2937;
            border-radius: 8px;
            padding: 10px 14px;
            margin-bottom: 10px;
            font-weight: 500;
            color: black;
            text-align: center;
            font-size: 15px;
            transition: all 0.3s ease;
        }

        label[data-baseweb="radio"]:hover {
            border: 1px solid #10b981;
            background-color: #374151;
            color: #10b981;
        }

        input[type="radio"]:checked + div > div {
            color: #10b981 !important;
            font-weight: bold;
        }
            
        input[type="radio"]:not(:checked) + div > div {
        # background-color: #FFFFFF !important;
        color: white !important;
        font-weight: bold;
       }

        /* --- Bottom Nav --- */
        .bottom_navbar {
            position: fixed;
            bottom: 7%;
            left: 80%;
            transform: translateX(-50%);
            background-color: rgba(0, 0, 0, 0.1);
            display: flex;
            justify-content: center;
            flex-wrap: wrap;
            padding: 6px 10px;
            border-radius: 12px;
            z-index: 999;
            width: 20%;
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
            border: 2px solid #A9A9A9;
        }

        .bottom_navbar a {
            color: white;
            padding: 8px 16px;
            margin: 4px;
            font-size: 16px;
            font-weight: 500;
            border-radius: 8px;
            text-decoration: none;
            background-color: #FF00FF;
            border: 1px solid #A9A9A9;
        }

        .bottom_navbar a:hover {
            background-color: #8B008B;
        }

        @media screen and (max-width: 600px) {
            .bottom_navbar {
                width: 80%;
                left: auto;
                right: 10%;
                text-align: center;
                border-radius: 25px;
            }

            .bottom_navbar a {
                font-size: 12px;
            }
        }
    </style>
""", unsafe_allow_html=True)

# ---- Define Tabs ----
tabs = ["Home", "About", "Projects", "Contact", "Next1", "Next2"]
query_params = st.query_params
selected_tab = query_params.get("selected_tab", ["Home"])[0]
st.session_state["selected_tab"] = selected_tab

# ---- Top Header Navbar ----
nav_html = f"""
<div class="header-container">
    <div class="logo-container">
        <img src="{logo_url}" alt="Logo">
    </div>
    <div class="navbar">
"""
for tab in tabs:
    active_class = "active" if tab == selected_tab else ""
    nav_html += f'<a href="/?selected_tab={tab}" class="{active_class}">{tab}</a>'
nav_html += "</div></div>"
st.markdown(nav_html, unsafe_allow_html=True)

# ---- Top Navbar Content ----
st.title(f"{selected_tab} Page")
if selected_tab == "Home":
    st.write("🏠 Welcome to the Home tab.")
elif selected_tab == "About":
    st.write("📖 This is the About section.")
elif selected_tab == "Projects":
    st.write("🛠️ Here are some cool projects.")
elif selected_tab == "Contact":
    st.write("📩 Get in touch with us!")

# ---- Sidebar Navigation ----
with st.sidebar:
    st.markdown(f'<img src="{logo_url}" class="profile-pic">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">IntekkiMed</div>', unsafe_allow_html=True)
    st.sidebar.title("📋 Navigation")
    page = st.radio(" ", ["🤖 ChatBot", "📖 Diabetes", "🛠️ Heart Disease", "🩺 Services", "📬 Contact", "⚙️ Settings"])

# ---- Main Content ----
st.title(page.split(" ", 1)[1] + " Page")

if "Diabetes" in page:
    with st.form("diabetes_form"):
        col1, col2 = st.columns(2)
        with col1:
            pregnancies = st.number_input("Pregnancies", 0, 20)
            blood_pressure = st.number_input("Blood Pressure", 0, 180)
            insulin = st.number_input("Insulin", 0, 900)
            diabetes_pedigree = st.number_input("Pedigree Function", 0.0, 2.5, step=0.01)
        with col2:
            glucose = st.number_input("Glucose", 0, 300)
            skin_thickness = st.number_input("Skin Thickness", 0, 100)
            bmi = st.number_input("BMI", 0.0, 70.0, step=0.1)
            age = st.number_input("Age", 1, 120)

        col3, col4, col5 = st.columns([1, 1, 2])
        submit = col3.form_submit_button("🔍 Predict")
        reset = col4.form_submit_button("🔄 Reset")
        help_btn = col5.form_submit_button("❓ Help")

    if submit:
        prediction = "Positive" if glucose > 120 else "Negative"
        st.success(f"🧾 Prediction: You are likely **{prediction}** for diabetes.")
    elif help_btn:
        st.info("👉 Glucose level above 120 is considered risky in this example.")
    elif reset:
        st.warning("🔁 Please refresh the page to reset all values.")

# ---- Bottom Nav ----
bottom_nav_html = """
<div class="bottom_navbar">
    <a href="#home">Home</a>
    <a href="#about">About</a>
    <a href="#services">Services</a>
</div>
"""
st.markdown(bottom_nav_html, unsafe_allow_html=True)
