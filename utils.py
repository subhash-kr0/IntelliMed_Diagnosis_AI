import spacy

# Load the English model
nlp = spacy.load("en_core_web_sm")

# Define relevant medical/disease-related terms
MEDICAL_CATEGORIES = {
    "SYMPTOM", "DIAGNOSIS", "TREATMENT", "DRUG", "DISEASE", "BODY", "TEST", "ORGAN", "CONDITION"
}

# Keywords to help reinforce (can be enhanced further)
MEDICAL_KEYWORDS = [
    "pain", "fever", "cancer", "diabetes", "headache", "tumor", "x-ray",
    "scan", "bp", "sugar", "medicine", "tablet", "prescription", "treatment", 
    "diagnosis", "infection", "health", "doctor", "hospital"
]

def is_medical_query(text: str) -> bool:
    doc = nlp(text.lower())

    # Check if any known medical keywords exist
    if any(keyword in text.lower() for keyword in MEDICAL_KEYWORDS):
        return True

    # Also check using named entity recognition (NER) if applicable
    for ent in doc.ents:
        if ent.label_.upper() in MEDICAL_CATEGORIES:
            return True

    # Could not classify as medical
    return False
