# from flask import Flask, request, render_template
# import joblib
# import numpy as np
# import pandas as pd


# app = Flask(__name__)

# # Load model and scaler
# model = joblib.load('./models/lung_cancer_model.pkl')
# scaler = joblib.load('./models/scaler.pkl')

# # Feature order must match training data columns (excluding LUNG_CANCER)
# FEATURE_ORDER = [
#     'GENDER', 'AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY',
#     'PEER_PRESSURE', 'CHRONIC_DISEASE', 'FATIGUE', 'ALLERGY',
#     'WHEEZING', 'ALCOHOL_CONSUMING', 'COUGHING',
#     'SHORTNESS_OF_BREATH', 'SWALLOWING_DIFFICULTY', 'CHEST_PAIN'
# ]

# @app.route('/')
# def home():
#     return render_template('predict.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         # Get form data and convert to dictionary
#         data = request.form.to_dict()
        
#         # Preprocess form data
#         processed = []
#         for feature in FEATURE_ORDER:
#             value = data[feature]
            
#             # Handle special cases
#             if feature == 'GENDER':
#                 processed.append(1 if value == 'M' else 0)
#             elif feature == 'AGE':
#                 processed.append(int(value))
#             else:  # All other binary features
#                 processed.append(1 if value == 'Yes' else 0)
        
#         # Scale features
#         scaled_data = scaler.transform(np.array(processed).reshape(1, -1))
        
#         # Make prediction
#         prediction = model.predict(scaled_data)
#         probability = model.predict_proba(scaled_data)[0][1]
        
#         result = "High risk of lung cancer" if prediction[0] == 1 \
#             else "Low risk of lung cancer"
        
#         return render_template('predict.html', 
#                              prediction=result,
#                              probability=f"{probability:.2%}")

#     except Exception as e:
#         return str(e)

# if __name__ == '__main__':
#     app.run(debug=True)










from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = joblib.load('./models/lung_cancer_model.pkl')
scaler = joblib.load('./models/scaler.pkl')

FEATURE_ORDER = [
    'GENDER', 'AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY',
    'PEER_PRESSURE', 'CHRONIC_DISEASE', 'FATIGUE', 'ALLERGY',
    'WHEEZING', 'ALCOHOL_CONSUMING', 'COUGHING',
    'SHORTNESS_OF_BREATH', 'SWALLOWING_DIFFICULTY', 'CHEST_PAIN'
]

@app.route('/')
def home():
    return render_template("predict.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # यूजर से इनपुट लें
        data = [request.form.get(feat, type=float) for feat in FEATURE_ORDER]

        # मॉडल के लिए डेटा को numpy array में बदलें
        scaler_data = scaler.transform(np.array([data]).reshape(1, -1))

        # मॉडल से भविष्यवाणी करें
        prediction = model.predict(scaler_data)[0]

        result = "Positive (लंग कैंसर का खतरा है)" if prediction == 1 else "Negative (लंग कैंसर का खतरा नहीं है)"

        return render_template("predict.html", prediction=result)

    except Exception as e:
        return render_template("predict.html", prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
