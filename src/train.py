import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
import logging

logging.basicConfig(filename='../logs', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_model():
    try:
        data = pd.read_csv('./data/heart.csv')
        X = data.drop(columns=['target'])
        y = data['target']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        accuracy = accuracy_score(y_test, model.predict(X_test))
        logging.info(f"Model trained successfully with accuracy: {accuracy}")
        
        with open('./models/heart_model.pkl', 'wb') as f:
            pickle.dump(model, f)
        logging.info("Model saved successfully.")
    except Exception as e:
        logging.error(f"Error in train_model: {str(e)}")
        raise


train_model()