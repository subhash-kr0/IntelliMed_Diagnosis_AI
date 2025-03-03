import joblib
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Load model and scaler
model = joblib.load('./models/diabetes_model.pkl')
scaler = joblib.load('./models/scaler.pkl')

# Feature names (adjust based on your dataset columns)
FEATURES = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
]

@app.route('/')
def home():
    return render_template('index.html', features=FEATURES)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form
        data = [float(request.form[feature]) for feature in FEATURES]
        
        # Preprocess
        scaled_data = scaler.transform([data])
        
        # Predict
        prediction = model.predict(scaled_data)
        probability = model.predict_proba(scaled_data)[0][1]
        
        # Convert to diagnosis
        diagnosis = "Malignant" if prediction[0] == 1 else "Benign"
        
        return render_template('index.html', 
                             prediction_text=f'Diagnosis: {diagnosis} (Confidence: {probability:.2%})',
                             features=FEATURES)
    
    except Exception as e:
        return render_template('index.html', 
                             prediction_text=f'Error: {str(e)}',
                             features=FEATURES)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)