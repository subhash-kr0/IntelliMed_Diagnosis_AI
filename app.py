from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = joblib.load('models/random_forest_model.pkl')
scaler = joblib.load('models/scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    final_features = np.array(features).reshape(1, -1)
    scaled_features = scaler.transform(final_features)
    
    prediction = model.predict(scaled_features)
    
    result = "Heart Disease Detected!" if prediction[0] == 1 else "No Heart Disease Detected"
    
    return render_template('result.html', prediction_text=result)

if __name__ == '__main__':
    app.run(debug=True)