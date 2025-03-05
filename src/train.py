import argparse
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(file_path):
    """Load and preprocess dataset"""
    try:
        data = pd.read_csv(file_path)
        logging.info("Data loaded successfully")

        # Handle missing values
        imputer = SimpleImputer(strategy='median')
        numerical_features = ['Age', 'BMI', 'Systolic_BP', 'Diastolic_BP', 'Total_Cholesterol']
        data[numerical_features] = imputer.fit_transform(data[numerical_features])

        # Convert categorical variables if applicable
        if 'Sex' in data.columns:
            data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
        
        if 'Hypertension' in data.columns:
            data['Hypertension'] = data['Hypertension'].map({0: 0, 1: 1})  # Ensure binary target

        return data
    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")
        raise

def train_model(data_path, model_path):
    """Main training function"""
    try:
        # Load and preprocess data
        data = load_data(data_path)

        # Selecting relevant numerical features for ML
        selected_features = ['Age', 'BMI', 'Systolic_BP', 'Diastolic_BP', 'Total_Cholesterol']
        
        # Split data
        X = data[selected_features]
        y = data['Hypertension']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Feature scaling (Apply only on X_train and X_test)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Initialize models
        models = {
            'Random Forest': RandomForestClassifier(random_state=42),
            'XGBoost': XGBClassifier(random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42),
            'SVM': SVC(random_state=42)
        }

        # Hyperparameter tuning for Random Forest
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5]
        }

        best_accuracy = 0
        best_model = None

        # Train and evaluate models
        for name, model in models.items():
            logging.info(f"Training {name}...")

            if name == 'Random Forest':
                grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
                grid_search.fit(X_train, y_train)
                model = grid_search.best_estimator_
                logging.info(f"Best parameters for Random Forest: {grid_search.best_params_}")

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            logging.info(f"{name} Accuracy: {accuracy:.4f}")
            logging.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model

        # Save best model and scaler
        joblib.dump(best_model, model_path)
        joblib.dump(scaler, 'models/scaler.pkl')
        logging.info(f"Best model saved to {model_path}")

    except Exception as e:
        logging.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Hypertension Prediction Model')
    parser.add_argument('--data_path', type=str, default='data/hypertension.csv', help='Path to input data')
    parser.add_argument('--model_path', type=str, default='models/best_model.pkl', help='Path to save trained model')
    
    args = parser.parse_args()
    
    train_model(args.data_path, args.model_path)
