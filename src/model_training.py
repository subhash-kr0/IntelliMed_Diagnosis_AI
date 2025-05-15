import os
import json
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SKPipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from ydata_profiling import ProfileReport

from logger import logger
from exception import CustomException

# Candidate models and parameters
CLASSIFIERS = {
    "RandomForest": (RandomForestClassifier(), {"model__n_estimators": [50, 100]}),
    "LogisticRegression": (LogisticRegression(max_iter=1000), {"model__C": [0.1, 1.0, 10.0]}),
    "SVM": (SVC(), {"model__C": [0.1, 1.0], "model__kernel": ["linear", "rbf"]}),
    "NaiveBayes": (GaussianNB(), {}),
    "DecisionTree": (DecisionTreeClassifier(), {"model__max_depth": [3, 5, 10]}),
}

REGRESSORS = {
    "RandomForest": (RandomForestRegressor(), {"model__n_estimators": [50, 100]}),
    "LinearRegression": (LinearRegression(), {}),
    "SVR": (SVR(), {"model__C": [0.1, 1.0], "model__kernel": ["linear", "rbf"]}),
}

def preprocess_data(df):
    df = df.dropna(axis=1, how='all')  # Drop columns with all NaNs
    df = df.dropna()  # Drop rows with any NaNs
    return df

def is_classification(df, target_column):
    target = df[target_column]
    return target.nunique() <= 15 or target.dtype == "object"

def train_pipeline(file_path, model_dir="models", report_dir="reports/metrics"):
    try:
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(report_dir, exist_ok=True)

        # Read and clean data
        df = pd.read_csv(file_path)
        df = preprocess_data(df)

        # Generate data profiling report
        profile = ProfileReport(df, title="Data Profiling Report", minimal=True)
        file_base = os.path.splitext(os.path.basename(file_path))[0]
        profile.to_file(os.path.join(report_dir, f"{file_base}_profile.html"))

        # Setup target and features
        target_column = df.columns[-1]
        X = df.drop(target_column, axis=1)
        y = df[target_column]

        # Task type
        classification_task = is_classification(df, target_column)
        models = CLASSIFIERS if classification_task else REGRESSORS

        # Identify feature types
        numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
        categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

        preprocessor = ColumnTransformer(transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ])

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42,
            stratify=y if classification_task else None
        )

        # Model training loop
        best_model = None
        best_score = -np.inf
        best_name = ""
        best_params = {}

        for name, (model, params) in models.items():
            pipeline = SKPipeline([
                ("preprocessor", preprocessor),
                ("model", model)
            ])
            grid = GridSearchCV(pipeline, param_grid=params, cv=5)
            grid.fit(X_train, y_train)
            cv_score = grid.best_score_

            logger.info(f"Model: {name}, CV Score: {cv_score:.4f}, Params: {grid.best_params_}")

            if cv_score > best_score:
                best_model = grid.best_estimator_
                best_score = cv_score
                best_name = name
                best_params = grid.best_params_

        # Final evaluation
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)

        # Save model
        model_path = os.path.join(model_dir, f"{file_base}_{best_name}.pkl")
        joblib.dump(best_model, model_path)

        # Save evaluation report
        report_path = os.path.join(report_dir, f"{file_base}_report.txt")
        metrics_path = os.path.join(report_dir, f"{file_base}_metrics.json")

        with open(report_path, "w") as f:
            f.write(f"Dataset: {file_path}\n")
            f.write(f"Target Column: {target_column}\n")
            f.write(f"Task Type: {'Classification' if classification_task else 'Regression'}\n")
            f.write(f"Best Model: {best_name}\n")
            f.write(f"Best Params: {best_params}\n")
            f.write(f"CV Score: {best_score:.4f}\n")

            if classification_task:
                report = classification_report(y_test, y_pred)
                f.write("\nClassification Report:\n" + report)
            else:
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                f.write(f"\nMSE: {mse:.4f}\nR2 Score: {r2:.4f}\n")

        # Save metrics as JSON
        metrics = {
            "model": best_name,
            "params": best_params,
            "cv_score": best_score,
            "task": "classification" if classification_task else "regression",
        }

        if classification_task:
            metrics["classification_report"] = classification_report(y_test, y_pred, output_dict=True)
        else:
            metrics["mse"] = mean_squared_error(y_test, y_pred)
            metrics["r2"] = r2_score(y_test, y_pred)

        with open(metrics_path, "w") as jf:
            json.dump(metrics, jf, indent=4)

        print(f"✅ Trained {best_name} on {file_path} | Saved model to {model_path}")

    except Exception as e:
        logger.error("An error occurred in the training pipeline.")
        raise CustomException(e)
