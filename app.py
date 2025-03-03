import joblib
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Load model and scaler
model = joblib.load('./models/breast_cancer_rf_model.pkl')
scaler = joblib.load('./models/scaler.pkl')

# Feature names (adjust based on your dataset columns)
FEATURES = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 
    'smoothness_mean', 'compactness_mean', 'concavity_mean', 
    'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
    'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
    'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
    'fractal_dimension_se', 'radius_worst', 'texture_worst',
    'perimeter_worst', 'area_worst', 'smoothness_worst',
    'compactness_worst', 'concavity_worst', 'concave points_worst',
    'symmetry_worst', 'fractal_dimension_worst'
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