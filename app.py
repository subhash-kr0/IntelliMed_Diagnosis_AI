import joblib
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Load model and scaler
model = joblib.load('./models/diabetes_model.pkl')
scaler = joblib.load('./models/diabetes_scaler.pkl')

kidneyModel = joblib.load('./models/kidneyDisease_model.pkl')

# Diabetes feature names
DIABETES_FEATURES = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
]

# Kidney feature names
KIDNEY_FEATURES = [
    'age', 'bp', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr',
    'bu', 'sc', 'pot', 'wc', 'htn', 'dm', 'cad', 'pe', 'ane'
]

@app.route('/')
@app.route('/index')
def home():
    return render_template('index.html', features=DIABETES_FEATURES)


@app.route('/diabetes')
def diabetes():
    return render_template('diabetesPage.html', features=DIABETES_FEATURES)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form
        data = [float(request.form[feature]) for feature in DIABETES_FEATURES]

        # Preprocess
        scaled_data = scaler.transform([data])

        # Predict
        prediction = model.predict(scaled_data)
        probability = model.predict_proba(scaled_data)[0][1]

        # Convert to diagnosis
        diagnosis = "Diabetes Detected" if prediction[0] == 1 else "No Diabetes"

        return render_template('diabetesPage.html',
                               prediction_text=f'Diagnosis: {diagnosis} (Confidence: {probability:.2%})',
                               features=DIABETES_FEATURES)

    except Exception as e:
        return render_template('diabetesPage.html',
                               prediction_text=f'Error: {str(e)}',
                               features=DIABETES_FEATURES)


@app.route('/kidneyDisease')
def kidneyDisease():
    return render_template('kidneyDisease.html', features=KIDNEY_FEATURES)


@app.route('/kidneyDisease', methods=['POST'])
def kidneyDiseasePredict():
    try:
        # Get data from form
        data = [float(request.form[feature]) for feature in KIDNEY_FEATURES]
        data = np.array(data).reshape(1, -1)

        # Predict
        prediction = kidneyModel.predict(data)
        probability = kidneyModel.predict_proba(data)[0][1]

        # Convert to diagnosis
        diagnosis = "Kidney Disease Present" if prediction[0] == 1 else "Healthy"

        return render_template('kidneyDisease.html',
                               prediction_text=f'Diagnosis: {diagnosis} (Confidence: {probability:.2%})',
                               features=KIDNEY_FEATURES)

    except Exception as e:
        return render_template('kidneyDisease.html',
                               prediction_text=f'Error: {str(e)}',
                               features=KIDNEY_FEATURES)


heart_model = joblib.load('models/heartDisease_randomForest_model.pkl')
heart_scaler = joblib.load('models/heartDisease_scaler.pkl')

# @app.route('/')
# def home():
#     return render_template('index.html')

@app.route('/heartDisease', methods=['POST', 'GET'])
# def heartDisease():
#     return render_template('heartDisease.html')

@app.route('/heartDisease', methods=['POST', 'GET'])
def heartDiseasePredict():
    if request.method == 'POST':
        try:
            # Get values from form
            features = [float(x) for x in request.form.values()]
            print("Received features:", features)

            if len(features) == 0:
                raise ValueError("No input features received!")

            final_features = np.array(features).reshape(1, -1)
            scaled_features = heart_scaler.transform(final_features)

            prediction = heart_model.predict(scaled_features)
            result = "Heart Disease Detected!" if prediction[0] == 1 else "No Heart Disease Detected"

            return render_template('heartDisease.html', prediction_text=result)

        except Exception as e:
            return render_template('heartDisease.html', prediction_text=f"Error: {str(e)}")

    return render_template('heartDisease.html')


hypertension_model = joblib.load('models/hypertension_model.pkl')


@app.route('/hypertension', methods=['POST', 'GET'])
def hypertensionPredict():
    try:
        # Convert input values to float
        features = [float(request.form.get(field, 0)) for field in ['age', 'bmi', 'systolic_bp', 'diastolic_bp', 'cholesterol']]
        final_features = [np.array(features)]
        
        # Make prediction
        if model:
            prediction = hypertension_model.predict(final_features)
            result = "Hypertensive" if prediction[0] == 1 else "Non-Hypertensive"
        else:
            result = "Model not loaded. Prediction unavailable."
        
        return render_template('hypertensionPage.html', prediction=result)
    except Exception as e:
        return render_template('hypertensionPage.html', prediction=f"Error: {str(e)}")
    


breastModel = joblib.load('./models/breastCancer_randomForest_model.pkl')

BREAST_FEATURES = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 
    'smoothness_mean', 'compactness_mean', 'concavity_mean', 
    'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean'
]

@app.route('/breastCancer')
def breast():
    return render_template('breastCancer.html', features=BREAST_FEATURES)

@app.route('/breastCancerPredict', methods=['POST', 'GET'])
def BreastCancerPredict():
    try:
        # Get data from form
        data = [float(request.form[feature]) for feature in BREAST_FEATURES]

        # Reshape to 2D
        import numpy as np
        data_np = np.array(data).reshape(1, -1)

        # Predict
        prediction = breastModel.predict(data_np)
        probability = breastModel.predict_proba(data_np)[0][1]

        # Convert to diagnosis
        diagnosis = "Malignant" if prediction[0] == 1 else "Benign"

        return render_template('breastCancer.html', 
                             prediction_text=f'Diagnosis: {diagnosis} (Confidence: {probability:.2%})',
                             features=BREAST_FEATURES)

    except Exception as e:
        return render_template('breastCancer.html', 
                             prediction_text=f'Error: {str(e)}',
                             features=BREAST_FEATURES)




model = joblib.load('./models/lungCancer_XGBClassifier_model.pkl')
# scaler = joblib.load('./models/scaler.pkl')

LUNG_FEATURE_ORDER = [
    'GENDER', 'AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY',
    'PEER_PRESSURE', 'CHRONIC_DISEASE', 'FATIGUE', 'ALLERGY',
    'WHEEZING', 'ALCOHOL_CONSUMING', 'COUGHING',
    'SHORTNESS_OF_BREATH', 'SWALLOWING_DIFFICULTY', 'CHEST_PAIN'
]

# @app.route('/')
# def home():
#     return render_template("predict.html")

@app.route('/lungCancerPredict', methods=['POST', 'GET'])
def lungCancerPredict():
    try:
        data = [request.form.get(feat, type=float) for feat in LUNG_FEATURE_ORDER]


        # scaler_data = scaler.transform(np.array([data]).reshape(1, -1))

        prediction = model.predict([data])[0]

        result = "Positive (लंग कैंसर का खतरा है)" if prediction == 1 else "Negative (लंग कैंसर का खतरा नहीं है)"

        return render_template("lungPage.html", prediction=result)

    except Exception as e:
        return render_template("lungPage.html", prediction=f"Error: {str(e)}")



model = joblib.load('./models/liverDisease_rf_model.pkl')  # Make sure you have this model saved

@app.route('/liver')
def liver():
    return render_template('liverDisease.html')

@app.route('/liverDisease', methods=['POST'])
def liverDiseasePredict():
    try:
        data = [
            float(request.form['Age']),
            float(request.form['Gender']),
            float(request.form['Total_Bilirubin']),
            float(request.form['Alkaline_phosphate']),
            float(request.form['Alamine_Aminotransferace']),
            float(request.form['Aspartate_Amino']),
            float(request.form['Protien']),
            float(request.form['Albumin']),
            float(request.form['Albumin_Globulin_ratio'])
        ]
        prediction = model.predict([data])[0]
        result = "Liver Disease Detected" if prediction == 1 else "No Liver Disease"
        return render_template('liverDisease.html', prediction_text=result)
    except Exception as e:
        return render_template('liverDisease.html', prediction_text=f"Error: {str(e)}")
    


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
