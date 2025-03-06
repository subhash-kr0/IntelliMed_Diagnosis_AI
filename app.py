from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('./models/final_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from form
        age = float(request.form['age'])
        sex = int(request.form['sex'])  # Encoded as 0 or 1
        TSH = float(request.form['TSH'])
        T3 = float(request.form['T3'])
        TT4 = float(request.form['TT4'])
        T4U = float(request.form['T4U'])
        FTI = float(request.form['FTI'])
        TBG = float(request.form['TBG'])
        on_thyroxine = int(request.form['on_thyroxine'])
        pregnant = int(request.form['pregnant'])
        referral_source = int(request.form['referral_source'])
        lithium = float(request.form['lithium'])
        
        # Create feature array
        features = np.array([[age, sex, TSH, T3, TT4, T4U, FTI, TBG, on_thyroxine, pregnant, referral_source, lithium]])
        
        # Make prediction
        prediction = model.predict(features)
        result = 'Positive for Thyroid Disease' if prediction[0] == 1 else 'Negative for Thyroid Disease'
        
        return render_template('index.html', prediction=result)
    except Exception as e:
        return render_template('index.html', prediction=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)