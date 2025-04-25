# import streamlit as st

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
#             box-shadow: 0 4px 12px rgba(0,0,0,0.2);
#             margin-bottom: 20px;
#             flex-wrap: wrap;
#         }

#         .logo-container img {
#             height: 50px;
#         }

#         .navbar {
#             display: flex;
#             justify-content: flex-end;
#             flex-wrap: wrap;
#             gap: 10px;
#         }

#         .navbar button {
#             color: #fff;
#             padding: 8px 16px;
#             font-size: 16px;
#             font-weight: 500;
#             border-radius: 8px;
#             border: none;
#             background-color: #FF00FF;
#             cursor: pointer;
#             transition: 0.3s;
#         }

#         .navbar button:hover {
#             background-color: #d100d1;
#         }

#         .navbar button.active {
#             background-color: #1f77b4 !important;
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

# # Set default tab in session state
# if "selected_tab" not in st.session_state:
#     st.session_state.selected_tab = "Home"

# # ---- Render Logo + Navbar Buttons ----
# nav_html = f"""
# <div class="header-container">
#     <div class="logo-container">
#         <img src="{logo_url}" alt="Logo">
#     </div>
#     <div class="navbar">
# """

# for tab in tabs:
#     active_class = "active" if st.session_state.selected_tab == tab else ""
#     nav_html += f'<button class="{active_class}" onclick="window.dispatchEvent(new CustomEvent(\'{tab}-clicked\'))">{tab}</button>'

# nav_html += "</div></div>"

# st.markdown(nav_html, unsafe_allow_html=True)

# # ---- JavaScript for Button Handling ----
# script_parts = []
# for tab in tabs:
#     script_parts.append(f"""
#         window.addEventListener("{tab}-clicked", function() {{
#             const form = document.createElement('form');
#             form.method = 'POST';
#             form.style.display = 'none';
#             const input = document.createElement('input');
#             input.name = 'selected_tab';
#             input.value = '{tab}';
#             form.appendChild(input);
#             document.body.appendChild(form);
#             form.submit();
#         }});
#     """)

# script_full = "<script>" + "\n".join(script_parts) + "</script>"
# st.markdown(script_full, unsafe_allow_html=True)

# # ---- Hidden Form Handler ----
# if 'selected_tab' in st.query_params:
#     st.session_state.selected_tab = st.query_params.get['selected_tab'][0]

# if st.session_state.get("selected_tab") is None:
#     st.session_state.selected_tab = "Home"

# # ---- Tab Content ----
# selected_tab = st.session_state.selected_tab
# st.title(f"{selected_tab} Page")

# if selected_tab == "Home":
#     st.write("🏠 Welcome to the Home tab.")
# elif selected_tab == "About":
#     st.write("📖 This is the About section.")
# elif selected_tab == "Projects":
#     st.write("🛠️ Here are some cool projects.")
# elif selected_tab == "Contact":
#     st.write("📩 Get in touch with us!")
# elif selected_tab == "next":
#     st.write("➡️ This is the 'next' tab content.")








# import streamlit as st

# st.set_page_config(page_title="IntekkiMed", layout="wide")

# logo_url = "https://upload.wikimedia.org/wikipedia/commons/a/ab/Logo_TV_2015.png"

# # ---- CSS Styling ----
# st.markdown("""
#     <style>
#         #MainMenu {visibility: hidden;}
#         footer {visibility: hidden;}
#         header {visibility: hidden;}
#         .block-container {padding-top: 0rem !important;}
#         .header-container {
#             display: flex;
#             justify-content: space-between;
#             align-items: center;
#             background-color: rgba(0, 0, 0, 0.09);
#             padding: 10px 20px;
#             border-radius: 12px;
#             box-shadow: 0 4px 12px rgba(0,0,0,0.2);
#             margin-bottom: 20px;
#             flex-wrap: wrap;
#         }
#         .logo-container img { height: 50px; }
#     </style>
# """, unsafe_allow_html=True)

# # ---- Tabs ----
# tabs = ["Home", "About", "Projects", "Contact", "next", "next"]

# # ---- Custom HTML Header (logo only) ----
# st.markdown(f"""
# <div class="header-container">
#     <div class="logo-container">
#         <img src="{logo_url}" alt="Logo">
#     </div>
# </div>
# """, unsafe_allow_html=True)

# # ---- Streamlit-native Tab Selector ----
# selected_tab = st.radio("Navigation", tabs, horizontal=True)

# # ---- Tab Content ----
# st.title(f"{selected_tab} Page")

# if selected_tab == "Home":
#     st.write("🏠 Welcome to the Home tab.")
# elif selected_tab == "About":
#     st.write("📖 This is the About section.")
# elif selected_tab == "Projects":
#     st.write("🛠️ Here are some cool projects.")
# elif selected_tab == "Contact":
#     st.write("📩 Get in touch with us!")
# elif selected_tab == "next":
#     st.write("⏭️ This is the next page content.")











# import streamlit as st

# st.set_page_config(page_title="IntekkiMed", layout="wide")

# # ---- Logo URL ----
# logo_url = "https://upload.wikimedia.org/wikipedia/commons/a/ab/Logo_TV_2015.png"

# # ---- CSS Styling ----
# st.markdown("""
#     <style>
#         #MainMenu, footer, header {visibility: hidden;}
#         .block-container {padding-top: 0rem !important;}
#         .header-container {
#             display: flex;
#             justify-content: space-between;
#             align-items: center;
#             background-color: rgba(0, 0, 0, 0.09);
#             padding: 10px 20px;
#             border-radius: 12px;
#             box-shadow: 0 4px 12px rgba(0,0,0,0.2);
#             margin-bottom: 20px;
#             flex-wrap: wrap;
#         }
#         .logo-container img { height: 50px; }

#         .nav-button {
#             # background-color: #FF00FF;
#             # color: white;
#             # padding: 8px 16px;
#             # margin: 5px;
#             # border: none;
#             # border-radius: 8px;
#             # font-size: 16px;
#             # font-weight: 500;
#             # cursor: pointer;

#             display: flex;
#             justify-content: flex-end;
#             flex-wrap: wrap;
#             gap: 10px;
#         }
            
#             .navbar a {
#             color: #fff;
#             padding: 8px 16px;
#             text-decoration: none;
#             font-size: 16px;
#             font-weight: 500;
#             border-radius: 8px;
#             transition: 0.3s;
#             background-color: #FF00FF;
#             border: 2px solid #FF00FF;
#         }

#         .nav-button:hover {
#             background-color: #cc00cc;
#         }

#         .nav-active {
#             background-color: #1f77b4 !important;
#         }
#     </style>
# """, unsafe_allow_html=True)

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










# import streamlit as st

# st.set_page_config(page_title="IntekkiMed", layout="wide")

# # ---- Logo URL ----
# logo_url = "https://upload.wikimedia.org/wikipedia/commons/a/ab/Logo_TV_2015.png"

# # ---- CSS Styling ----
# st.markdown("""
#     <style>
#         #MainMenu, footer, header {visibility: hidden;}
#         .block-container {padding-top: 0rem !important;}
#         .header-container {
#             display: flex;
#             justify-content: space-between;
#             align-items: center;
#             background-color: rgba(0, 0, 0, 0.09);
#             padding: 10px 20px;
#             border-radius: 12px;
#             box-shadow: 0 4px 12px rgba(0,0,0,0.2);
#             margin-bottom: 20px;
#             flex-wrap: wrap;
#         }
#         .logo-container img { height: 50px; }

#         .navbar-container {
#             display: flex;
#             flex-wrap: wrap;
#             gap: 10px;
#             margin-top: 10px;
#         }

#         .stButton>button {
#             background-color: #FF00FF;
#             color: white;
#             padding: 8px 16px;
#             border: none;
#             border-radius: 8px;
#             font-size: 16px;
#             font-weight: 500;
#             cursor: pointer;
#         }

#         .stButton>button:hover {
#             background-color: #cc00cc;
#         }

#         .active-button {
#             background-color: #1f77b4 !important;
#         }
#     </style>
# """, unsafe_allow_html=True)

# # ---- Tabs ----
# tabs = ["Home", "About", "Projects", "Contact", "next", "next"]

# # ---- Session State for Toggle ----
# if "active_tab" not in st.session_state:
#     st.session_state.active_tab = None
#     st.session_state.tab_index = None

# # ---- Header Section (Logo) ----
# st.markdown(f"""
# <div class="header-container">
#     <div class="logo-container">
#         <img src="{logo_url}" alt="Logo">
#     </div>
# </div>
# """, unsafe_allow_html=True)

# # ---- Navbar Buttons (Visible!) ----
# st.markdown('<div class="navbar-container">', unsafe_allow_html=True)
# cols = st.columns(len(tabs))
# for i, tab in enumerate(tabs):
#     key = f"btn_{i}"
#     is_active = (st.session_state.active_tab == tab and st.session_state.tab_index == i)

#     button_label = f"**{tab}**" if is_active else tab
#     if cols[i].button(button_label, key=key):
#         if is_active:
#             st.session_state.active_tab = None
#             st.session_state.tab_index = None
#         else:
#             st.session_state.active_tab = tab
#             st.session_state.tab_index = i
# st.markdown("</div>", unsafe_allow_html=True)

# # ---- Tab Content ----
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
#         st.write("⏭️ This is a next page content.")









import streamlit as st

# ---- Logo URL (Replace with your actual image URL or local path if needed) ----
logo_url = "https://upload.wikimedia.org/wikipedia/commons/a/a7/React-icon.svg"  # example logo

# ---- Tabs ----
tabs = ["Home", "About", "Projects", "Contact", "next", "next"]

# ---- Session Management for Toggle ----
if "active_tab" not in st.session_state:
    st.session_state.active_tab = None
if "tab_index" not in st.session_state:
    st.session_state.tab_index = None

# ---- Inject CSS for Styling ----
st.markdown("""
<style>
/* Header section */
.header-container {
    display: flex;
    align-items: center;
    justify-content: center;
    margin-bottom: 20px;
}
.logo-container img {
    height: 80px;
}

/* Button container */
.block-container > div > div > div > div {
    display: flex;
    justify-content: center;
    flex-wrap: wrap;
    gap: 10px;
}

/* Nav button style */
button[kind="secondary"] {
    background-color: #f0f0f0;
    color: #333;
    padding: 10px 20px;
    border-radius: 8px;
    border: 1px solid #ccc;
    font-weight: 500;
    transition: background-color 0.3s ease, color 0.3s ease;
}

/* Active tab */
button.nav-active {
    background-color: #4CAF50 !important;
    color: white !important;
    border: 1px solid #4CAF50;
}
</style>
""", unsafe_allow_html=True)

# ---- Header Section with Logo ----
st.markdown(f"""
<div class="header-container">
    <div class="logo-container">
        <img src="{logo_url}" alt="Logo">
    </div>
</div>
""", unsafe_allow_html=True)

# ---- Navbar Buttons ----
cols = st.columns(len(tabs))
for i, tab in enumerate(tabs):
    # Track active state manually
    is_active = st.session_state.active_tab == tab and st.session_state.tab_index == i

    # Draw button
    if cols[i].button(tab, key=f"btn_{tab}_{i}"):
        if is_active:
            st.session_state.active_tab = None
            st.session_state.tab_index = None
        else:
            st.session_state.active_tab = tab
            st.session_state.tab_index = i

# ---- Display Content ----
if st.session_state.active_tab:
    st.title(f"{st.session_state.active_tab} Page")

    if st.session_state.active_tab == "Home":
        st.write("🏠 Welcome to the Home tab.")
    elif st.session_state.active_tab == "About":
        st.write("📖 This is the About section.")
    elif st.session_state.active_tab == "Projects":
        st.write("🛠️ Here are some cool projects.")
    elif st.session_state.active_tab == "Contact":
        st.write("📩 Get in touch with us!")
    elif st.session_state.active_tab == "next":
        st.write("⏭️ This is the next page content.")
