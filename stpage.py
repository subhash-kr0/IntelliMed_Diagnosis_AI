import streamlit as st







# # ---- Logo URL ----
# # logo_url = "https://upload.wikimedia.org/wikipedia/commons/a/a7/React-icon.svg"  # example logo

# # ---- Tabs ----
# tabs = ["Home", "About", "Projects", "Contact"]

# # ---- Session Management ----
# if "active_tab" not in st.session_state:
#     st.session_state.active_tab = None
# if "tab_index" not in st.session_state:
#     st.session_state.tab_index = None

# # ---- CSS for Ultra-Compact & Responsive Navbar ----
# st.markdown("""
# <style>
# /* Header */
# .header-container {
#     display: flex;
#     align-items: center;
#     justify-content: center;
#     margin-bottom: 10px;
# }
# .logo-container img {
#     height: 60px;
#     max-width: 100%;
# }

# /* Make navbar compact */
# .block-container > div > div > div > div {
#     display: flex;
#     justify-content: center;
#     flex-wrap: wrap;
#     gap: 10px;  /* 🔻 very minimal spacing */
#     margin-top: 10px;
#     margin-bottom: 10px;
#     right: 10%;
#     # position: fixed;
# }

# /* Nav button styling */
# button[kind="secondary"] {
#     background-color: #f2f2f2;
#     color: #333;
#     padding: 6px 10px;
#     border-radius: 5px;
#     font-size: 13px;
#     border: 1px solid #ccc;
#     font-weight: 500;
#     min-width: 80px;
#     transition: all 0.3s ease-in-out;
# }

# /* Active tab */
# button.nav-active {
#     background-color: #1a73e8 !important;
#     color: white !important;
#     border: 1px solid #1a73e8;
# }

# /* Mobile nav bar style */
# @media (max-width: 768px) {
#     .block-container > div > div > div > div {
#         flex-direction: row;
#         flex-wrap: wrap;
#         justify-content: space-around;
#         gap: 4px;
#     }
#     button[kind="secondary"] {
#         width: 45vw;
#         font-size: 12px;
#         padding: 6px 6px;
#         border-radius: 4px;
#     }
#     .logo-container img {
#         height: 45px;
#     }
# }
# </style>
# """, unsafe_allow_html=True)

# # ---- Logo/Header ----
# # st.markdown(f"""
# # <div class="header-container">
# #     <div class="logo-container">
# #         # <img src="{logo_url}" alt="Logo">
# #     </div>
# # </div>
# # """, unsafe_allow_html=True)

# # ---- Navbar Buttons ----
# cols = st.columns(len(tabs))
# for i, tab in enumerate(tabs):
#     is_active = st.session_state.active_tab == tab and st.session_state.tab_index == i
#     if cols[i].button(tab, key=f"btn_{tab}_{i}"):
#         if is_active:
#             st.session_state.active_tab = None
#             st.session_state.tab_index = None
#         else:
#             st.session_state.active_tab = tab
#             st.session_state.tab_index = i

# # ---- Content Display ----
# if st.session_state.active_tab:
#     st.title(f"{st.session_state.active_tab} Page")

#     if st.session_state.active_tab == "Home":
#         st.write("🏠 Welcome to the Home tab.")
#     elif st.session_state.active_tab == "About":
#         st.write("📖 This is the About section.")
#     elif st.session_state.active_tab == "Projects":
#         st.write("🛠️ Here are some cool projects.")
#     elif st.session_state.active_tab == "Contact":
#         st.write("📩 Get in touch with us!")









# # ---- Logo URL ----
# # logo_url = "https://upload.wikimedia.org/wikipedia/commons/a/a7/React-icon.svg"  # example logo

# # ---- Tabs ----
# tabs = ["Home", "About", "Projects", "Contact"]

# # ---- Session Management ----
# if "active_tab" not in st.session_state:
#     st.session_state.active_tab = None
# if "tab_index" not in st.session_state:
#     st.session_state.tab_index = None

# # ---- CSS for Ultra-Compact & Responsive Navbar ----
# st.markdown("""
# <style>
# /* Header */
# .header-container {
#     display: flex;
#     align-items: center;
#     justify-content: center;
#     margin-bottom: 10px;
# }
# .logo-container img {
#     height: 60px;
#     max-width: 100%;
# }

# /* Make navbar compact */
# .block-container > div > div > div > div {
#     display: flex;
#     justify-content: center;
#     flex-wrap: wrap;
#     gap: 10px;  /* 🔻 very minimal spacing */
#     margin-top: 10px;
#     margin-bottom: 10px;
#     right: 10%;
#     position: fixed;
            
# }

# /* Nav button styling */
# button[kind="secondary"] {
#     background-color: #f2f2f2;
#     color: #333;
#     padding: 6px 10px;
#     border-radius: 5px;
#     font-size: 13px;
#     border: 1px solid #ccc;
#     font-weight: 500;
#     min-width: 80px;
#     transition: all 0.3s ease-in-out;
# }

# /* Active tab */
# button.nav-active {
#     background-color: #1a73e8 !important;
#     color: white !important;
#     border: 1px solid #1a73e8;
# }

# /* Mobile nav bar style */
# @media (max-width: 768px) {
#     .block-container > div > div > div > div {
#         flex-direction: row;
#         flex-wrap: wrap;
#         justify-content: space-around;
#         gap: 4px;
#     }
#     button[kind="secondary"] {
#         width: 45vw;
#         font-size: 12px;
#         padding: 6px 6px;
#         border-radius: 4px;
#     }
#     .logo-container img {
#         height: 45px;
#     }
# }
# </style>
# """, unsafe_allow_html=True)

# # ---- Logo/Header ----
# # st.markdown(f"""
# # <div class="header-container">
# #     <div class="logo-container">
# #         <img src="{logo_url}" alt="Logo">
# #     </div>
# # </div>
# # """, unsafe_allow_html=True)

# # ---- Navbar Buttons ----
# cols = st.columns(len(tabs))
# for i, tab in enumerate(tabs):
#     is_active = st.session_state.active_tab == tab and st.session_state.tab_index == i
#     if cols[i].button(tab, key=f"btn_{tab}_{i}"):
#         if is_active:
#             st.session_state.active_tab = None
#             st.session_state.tab_index = None
#         else:
#             st.session_state.active_tab = tab
#             st.session_state.tab_index = i

# # ---- Content Display ----
# if st.session_state.active_tab:
#     st.title(f"{st.session_state.active_tab} Page")

#     if st.session_state.active_tab == "Home":
#         st.write("🏠 Welcome to the Home tab.")
#     elif st.session_state.active_tab == "About":
#         st.write("📖 This is the About section.")
#     elif st.session_state.active_tab == "Projects":
#         st.write("🛠️ Here are some cool projects.")
#     elif st.session_state.active_tab == "Contact":
#         st.write("📩 Get in touch with us!")
#     elif st.session_state.active_tab == "next":
#         st.write("⏭️ This is the next page content.")












# st.set_page_config(page_title="IntekkiMed", layout="wide")

# logo_url = "https://upload.wikimedia.org/wikipedia/commons/a/ab/Logo_TV_2015.png"

# # ---- CSS Styling ----
# st.markdown("""
#     <style>
#         #MainMenu {visibility: hidden;}
#         footer {visibility: hidden;}
#         header {visibility: hidden;}

#         .block-container {
#             padding-top: 0rem !important;
#             position: relative;
#         }

#         .header-container {
#             display: flex;
#             justify-content: space-between;
#             align-items: center;
#             background-color: rgba(0, 0, 0, 0.09);
#             padding: 10px 20px;
#             border-radius: 12px;
#             # border: 2px solid #A9A9A9;
#             box-shadow: 0 4px 12px rgba(0,0,0,0.2);
#             margin-bottom: 20px;
#             flex-wrap: wrap;
#         }

#         .logo-container img {
#             height: 40px;
#         }

#         .navbar {
#             display: flex;
#             justify-content: flex-end;
#             flex-wrap: wrap;
#             gap: 10px;
#         }

#         .navbar a {
#             color: #fff;
#             padding: 8px 16px;
#             text-decoration: none;
#             font-size: 16px;
#             font-weight: 500;
#             border-radius: 8px;
#             transition: 0.3s;
#             background-color: #FF00FF;
#             border: 2px solid #FF00FF;
#             display: flex;
#             justify-content: flex-end;
#             flex-wrap: wrap;
#             gap: 10px;
#         }

#         .navbar a:hover {
#             background-color: #FF00FF;
#         }

#         .navbar a.active {
#             background-color: #1f77b4;
#         }

#         @media screen and (max-width: 600px) {
#             .header-container {
#                 flex-direction: column;
#                 align-items: flex-start;
#             }

#             .navbar {
#                 justify-content: center;
#                 width: 100%;
#             }
#         }
#     </style>
# """, unsafe_allow_html=True)

# # ---- Tabs ----
# tabs = ["Home", "About", "Projects", "Contact", "next", "next"]

# query_params = st.query_params
# selected_tab = query_params.get("selected_tab", ["Home"])[0]
# st.session_state["selected_tab"] = selected_tab

# # ---- Custom HTML: Logo + Navbar ----
# nav_html = f"""
# <div class="header-container">
#     <div class="logo-container">
#         <img src="{logo_url}" alt="Logo">
#     </div>
#     <div class="navbar">
# """

# for tab in tabs:
#     active_class = "active" if tab == selected_tab else ""
#     nav_html += f'<a href="/?selected_tab={tab}" class="{active_class}">{tab}</a>'

# nav_html += "</div></div>"


# st.markdown(nav_html, unsafe_allow_html=True)

# ---- Tab Content ----
# st.title(f"{selected_tab} Page")
# if selected_tab == "Home":
#     st.write("🏠 Welcome to the Home tab.")
# elif selected_tab == "About":
#     st.write("📖 This is the About section.")
# elif selected_tab == "Projects":
#     st.write("🛠️ Here are some cool projects.")
# elif selected_tab == "Contact":
#     st.write("📩 Get in touch with us!")







# # ---- Tabs ----
# tabs = ["Home", "About", "Projects", "Contact", "next", "next"]

# # ---- Session Management for Toggle ----
# if "active_tab" not in st.session_state:
#     st.session_state.active_tab = None

# # ---- Header Section with Logo ----
# st.markdown(f"""
# <div class="header-container">
#     <div class="logo-container">
#         <img src="{logo_url}" alt="Logo">
#     </div>
# </div>
# """, unsafe_allow_html=True)

# # Navbar Buttons
# cols = st.columns(len(tabs))
# for i, tab in enumerate(tabs):
#     btn_style = "nav-button"
#     if st.session_state.active_tab == tab and st.session_state.tab_index == i:
#         btn_style += " nav-active"

#     if cols[i].button(tab, key=f"btn_{tab}_{i}"):
#         if st.session_state.active_tab == tab and st.session_state.tab_index == i:
#             st.session_state.active_tab = None
#             st.session_state.tab_index = None
#         else:
#             st.session_state.active_tab = tab
#             st.session_state.tab_index = i


# # ---- Display Content ----
# if st.session_state.active_tab:
#     st.title(f"{st.session_state.active_tab} Page")

#     if st.session_state.active_tab == "Home":
#         st.write("🏠 Welcome to the Home tab.")
#     elif st.session_state.active_tab == "About":
#         st.write("📖 This is the About section.")
#     elif st.session_state.active_tab == "Projects":
#         st.write("🛠️ Here are some cool projects.")
#     elif st.session_state.active_tab == "Contact":
#         st.write("📩 Get in touch with us!")
#     elif st.session_state.active_tab == "next":
#         st.write("⏭️ This is the next page content.")















# ---- CSS Styling ----
st.markdown("""
    <style>
    /* Hide default Streamlit header/footer */
    #MainMenu, footer, header {visibility: hidden;}

    /* Custom Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(to bottom, #0f172a, #1e293b);
        color: white;
        padding-top: 1.5rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }

    /* Profile Image Styling */
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

    /* Title below logo */
    .sidebar-title {
        text-align: center;
        font-size: 20px;
        font-weight: 600;
        margin-top: 5px;
        margin-bottom: 25px;
        color: #10b981;
    }

    /* Vertical radio buttons */
    .stRadio > div {
        flex-direction: column;
    }

                    
    /* Radio button labels */
    label[data-baseweb="radio"] {
        background-color: #1f2937;
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 10px;
        font-weight: 500;
        border: 1px solid transparent;
        transition: all 0.3s ease;
        color: black; /* Change tab text color to white */
        text-align: center;
        font-size: 15px;
    }

    /* Hover effect */
    label[data-baseweb="radio"]:hover {
        border: 1px solid #10b981;
        background-color: #374151;
        color: #10b981;
    }

    /* Selected tab */
    input[type="radio"]:checked + div > div {
        # background-color: #FFFFFF !important;
        color: #10b981 !important;
        font-weight: bold;
    }
            
             input[type="radio"]:not(:checked) + div > div {
        # background-color: #FFFFFF !important;
        color: white !important;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)




    

# ---- Sidebar ----
with st.sidebar:
    st.markdown('<img src="https://upload.wikimedia.org/wikipedia/commons/a/ab/Logo_TV_2015.png" class="profile-pic">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">IntekkiMed</div>', unsafe_allow_html=True)
    st.sidebar.title("📋 Navigation")

    # Tabs/Menu
    page = st.radio(" ", ["🤖 ChatBot", "📖 Diabetes", "🛠️ Heart Disease", "🩺 Services", "📬 Contact", "⚙️ Settings"])

# ---- Main Area ----
st.title(page.split(" ", 1)[1] + " Page")

if "Diabetes" in page:
    # st.subheader("🩺 Compact Diabetes Prediction Form")

    with st.form("diabetes_form"):
        # Use columns for compact layout
        col1, col2 = st.columns(2)

        with col1:
            pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, step=1, format="%d")
            blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=180, step=1)
            insulin = st.number_input("Insulin", min_value=0, max_value=900, step=1)
            diabetes_pedigree = st.number_input("Pedigree Function", min_value=0.0, max_value=2.5, step=0.01)

        with col2:
            glucose = st.number_input("Glucose", min_value=0, max_value=300, step=1)
            skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, step=1)
            bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, step=0.1)
            age = st.number_input("Age", min_value=1, max_value=120, step=1)

        # Action Buttons
        col3, col4, col5 = st.columns([1, 1, 2])
        with col3:
            submit = st.form_submit_button("🔍 Predict")
        with col4:
            reset = st.form_submit_button("🔄 Reset")
        with col5:
            help_btn = st.form_submit_button("❓ Help")

    if submit:
        prediction = "Positive" if glucose > 120 else "Negative"
        st.success(f"🧾 Prediction: You are likely **{prediction}** for diabetes.")

    elif help_btn:
        st.info("👉 Fill in all details accurately. Glucose level above 120 is considered risky in this example.")

    elif reset:
        st.warning("🔁 Please refresh the page to reset all values.")


# st.title(f"{selected_tab} Page")
# if selected_tab == "Home":
#     if "Diabetes" in page:
#         st.write("🏠 Welcome to the Home tab.")
# elif selected_tab == "About":
#     st.write("📖 This is the About section.")
# elif selected_tab == "Projects":
#     st.write("🛠️ Here are some cool projects.")
# elif selected_tab == "Contact":
#     st.write("📩 Get in touch with us!")



st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}

        .block-container {
            padding-top: 0rem !important;
            position: relative;
        }

        .bottom_navbar {
            position: fixed;
            bottom: 7%;
            left: 80%;
            transform: translateX(-50%);
            background-color: rgba(0, 0, 0, 0.1);  /* Transparent background */
            display: flex;
            justify-content: center;
            flex-wrap: wrap;
            padding: 6px 10px;
            border-radius: 12px;
            z-index: 999;
            width: 20%;
            margin-top: 10px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
            border: 2px solid #A9A9A9;  
        }

        .bottom_navbar a {
            color: white;
            text-align: center;
            padding: 8px 16px;
            margin: 4px;
            text-decoration: none;
            font-size: 16px;
            font-weight: 500;
            border-radius: 8px;
            transition: 0.3s;
            background-color: #FF00FF;  /* Dark gray background */
            border: 2px solid #FF00FF;  /* Magenta border */
            border: 1px solid #A9A9A9;  
        }

        .bottom_navbar a:hover {
            background-color: #8B008B;  /* Magenta hover effect */
        }

        .bottom_navbar a.active {
            background-color: #1f77b4;
        }
            
            @media screen and (max-width: 600px) {
            .bottom_navbar {
                width: 80%;
                align-items: center;
                text-align: center;
                transform: none;
                bottom: 7%;
                border-radius: 25px;
                left: auto;
                right: 10%;
            }

            .bottom_navbar a {
                # width: 100%;
                margin: 1px 0;
                text-align: center;
                align-items: center;
                font-size: 11px;
            }
        }

        .spacer {
            margin-top: 80px;
        }
    </style>
""", unsafe_allow_html=True)


# ---- Define Tabs ----
tabs = ["Home", "About", "Projects"]

# ---- Handle Query Params Properly ----
query_params = st.query_params
selected_tab = query_params.get("selected_tab", ["Home"])[0]  # default is "Home"
st.session_state["selected_tab"] = selected_tab

# ---- Create Navbar HTML ----
nav_html = '<div class="bottom_navbar">'
for tab in tabs:
    active_class = "active" if tab == selected_tab else ""
    nav_html += f'<a href="/?selected_tab={tab}" class="{active_class}">{tab}</a>'
nav_html += '</div>'

st.markdown(nav_html, unsafe_allow_html=True)
st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)  # Push content below fixed navbar







