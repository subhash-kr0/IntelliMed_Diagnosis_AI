import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import streamlit as st
from src.chatbot import DiagnosisChatbot


def main():
    st.title("Medical Diagnosis Assistant")
    st.warning("This is not real medical advice")
    
    symptoms = st.multiselect("Select symptoms", ["fever", "cough", "headache", "nausea", "rash"])
    
    if st.button("Analyze"):
        bot = DiagnosisChatbot()
        result = bot.get_diagnosis(symptoms)
        
        st.subheader("Possible Conditions")
        st.write(result["possible_conditions"])
        
        st.subheader("Recommendation")
        st.error(result["recommendation"])

if __name__ == "__main__":
    main()
