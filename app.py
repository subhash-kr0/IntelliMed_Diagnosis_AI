from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the model
try:
    model = joblib.load('models/best_rf_model.pkl')
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Convert input values to float
        features = [float(request.form.get(field, 0)) for field in ['age', 'bmi', 'systolic_bp', 'diastolic_bp', 'cholesterol']]
        final_features = [np.array(features)]
        
        # Make prediction
        if model:
            prediction = model.predict(final_features)
            result = "Hypertensive" if prediction[0] == 1 else "Non-Hypertensive"
        else:
            result = "Model not loaded. Prediction unavailable."
        
        return render_template('result.html', prediction=result)
    except Exception as e:
        return render_template('result.html', prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)