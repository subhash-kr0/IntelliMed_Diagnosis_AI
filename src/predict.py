import pickle
import logging
import numpy as np

logging.basicConfig(filename='../logs', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_model():
    try:
        with open('../models/heart_model.pkl', 'rb') as f:
            model = pickle.load(f)
        logging.info("Model loaded successfully.")
        return model
    except Exception as e:
        logging.error(f"Error in load_model: {str(e)}")
        raise

def predict(features):
    try:
        model = load_model()
        prediction = model.predict([features])
        logging.info("Prediction made successfully.")
        return prediction[0]
    except Exception as e:
        logging.error(f"Error in predict: {str(e)}")
        raise