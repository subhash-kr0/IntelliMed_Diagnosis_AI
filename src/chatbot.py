import os
import json
import google.generativeai as genai
import pandas as pd
from datetime import datetime
from typing import Dict, List
from config.settings import API_KEY

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-1.5-pro')

class DiagnosisChatbot:
    def __init__(self):
        self.model = model
        self.symptoms_db = self._load_symptoms_db()

    def _load_symptoms_db(self) -> Dict[str, List[str]]:
        with open("data/symptoms_db.json", "r") as f:
            return json.load(f)

    def _generate_differential_diagnosis(self, symptoms: List[str]) -> str:
        prompt = f"""
        Analyze these symptoms: {', '.join(symptoms)}.
        Match against this knowledge base: {self.symptoms_db}.
        List possible conditions in order of likelihood.
        """
        response = self.model.generate_content(prompt)
        return response.text

    def _generate_triage_recommendation(self, symptoms: List[str]) -> str:
        prompt = f"""
        Based on medical guidelines:
        - Chest pain: ER immediately
        - High fever (>104°F): Urgent care
        - Mild cough: Primary care
        
        Symptoms: {symptoms}
        Provide triage level (Emergency/Urgent/Routine).
        """
        return self.model.generate_content(prompt).text

    def get_diagnosis(self, symptoms: List[str]) -> Dict:
        self._validate_input(symptoms)
        diagnosis = self._generate_differential_diagnosis(symptoms)
        recommendation = self._generate_triage_recommendation(symptoms)
        return {"symptoms": symptoms, "possible_conditions": diagnosis, "recommendation": recommendation}
    
    def check_medication_interaction(self, meds: List[str], condition: str) -> str:
        prompt = f"""
        Check interactions between {meds} for {condition}.
        """
        return self.model.generate_content(prompt).text
    
    def _validate_input(self, symptoms: List[str]):
        if not symptoms:
            raise ValueError("No symptoms provided")
        if "chest pain" in symptoms:
            return "EMERGENCY: Seek immediate care"
    
    def log_interaction(self, query: str, response: str):
        with open("logs/audit.log", "a") as f:
            f.write(f"{datetime.now()}: {query} -> {response}\n")
