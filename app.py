from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model, scaler, and label encoder
model = joblib.load('models/liver_model.joblib')
scaler = joblib.load('models/scaler.joblib')
le = joblib.load('models/label_encoder.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract form data
        gender = request.form['Gender']
        features = [
            float(request.form['Age']),
            float(request.form['Total_Bilirubin']),
            float(request.form['Direct_Bilirubin']),
            float(request.form['Alkaline_Phosphatase']),
            float(request.form['Alamine_Aminotransferase']),
            float(request.form['Aspartate_Aminotransferase']),
            float(request.form['Total_Proteins']),
            float(request.form['Albumin']),
            float(request.form['Albumin_and_Globulin_Ratio'])
        ]
        
        # Convert gender using label encoder
        gender_encoded = le.transform([gender])[0]
        features.insert(1, gender_encoded)  # Insert gender at index 1
        
        # Scale features
        final_features = scaler.transform(np.array(features).reshape(1, -1))
        
        # Predict
        prediction = model.predict(final_features)
        
        result = "Liver Disease Detected!" if prediction[0] == 1 else "No Liver Disease Detected"
        return render_template('result.html', prediction=result)
    
    except Exception as e:
        return render_template('error.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
