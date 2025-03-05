import joblib
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

# Load model and scaler
model = joblib.load('./models/kidneyDisease.pkl')
# scaler = joblib.load('./models/kidney_scaler.pkl')

# Feature names (adjust based on your dataset columns)
FEATURES = [
    'age', 'bp', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 'bu', 'sc', 'pot', 'wc', 'htn', 'dm', 'cad', 'pe', 'ane'
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
        # scaled_data = scaler.transform([data])
        
        data = np.array(data).reshape(1, -1)


        # Predict
        prediction = model.predict(data)
        probability = model.predict_proba(data)[0][1]
        
        # Convert to diagnosis
        diagnosis = "Kidney Disease Present" if prediction[0] == 1 else "Healthy"
        
        return render_template('index.html', 
                             prediction_text=f'Diagnosis: {diagnosis} (Confidence: {probability:.2%})',
                             features=FEATURES)
    
    except Exception as e:
        return render_template('index.html', 
                             prediction_text=f'Error: {str(e)}',
                             features=FEATURES)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)