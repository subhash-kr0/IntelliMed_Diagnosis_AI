from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load model
model = joblib.load('models/model.pkl')

# Gender mapping
gender_mapping = {
    'male': 0,
    'female': 1,
    'other': 2
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    age = int(request.form['age'])
    gender = gender_mapping[request.form['gender'].lower()]
    family_history = 1 if request.form['family_history'] == 'yes' else 0
    work_interfere = int(request.form['work_interfere'])
    remote_work = 1 if request.form['remote_work'] == 'yes' else 0
    tech_company = 1 if request.form['tech_company'] == 'yes' else 0

    # Create DataFrame
    data = [[age, gender, family_history, work_interfere, remote_work, tech_company]]
    columns = ['Age', 'Gender', 'family_history', 'work_interfere', 'remote_work', 'tech_company']
    
    # Predict
    prediction = model.predict(pd.DataFrame(data, columns=columns))
    result = "Likely to seek treatment" if prediction[0] == 1 else "Unlikely to seek treatment"
    
    return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)