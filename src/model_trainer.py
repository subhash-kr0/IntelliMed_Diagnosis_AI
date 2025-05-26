# File: automated_disease_diagnosis/src/model_trainer.py
# =========================================
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
import joblib
import os
import time

def get_classification_models(task_type: str):
    """Returns a dictionary of classification models to try."""
    models = {
        "LogisticRegression": LogisticRegression(solver='liblinear', random_state=42, max_iter=200),
        "RandomForestClassifier": RandomForestClassifier(random_state=42),
        "GradientBoostingClassifier": GradientBoostingClassifier(random_state=42),
        "SVC": SVC(probability=True, random_state=42), # probability=True for roc_auc
        "KNeighborsClassifier": KNeighborsClassifier(),
        "DecisionTreeClassifier": DecisionTreeClassifier(random_state=42),
        "GaussianNB": GaussianNB(),
        "XGBClassifier": XGBClassifier(use_label_encoder=False, eval_metric='logloss' if task_type == "binary_classification" else 'mlogloss', random_state=42),
        "LGBMClassifier": LGBMClassifier(random_state=42, verbosity=-1)
    }
    return models

def get_param_grids():
    """Returns basic hyperparameter grids for GridSearchCV."""
    # Reduced grids for faster execution in this example
    param_grids = {
        "LogisticRegression": {'C': [0.1, 1.0]}, # Reduced from [0.01, 0.1, 1, 10, 100]
        "RandomForestClassifier": {'n_estimators': [50, 100], 'max_depth': [None, 10]}, # Reduced
        "GradientBoostingClassifier": {'n_estimators': [50, 100], 'learning_rate': [0.05, 0.1]}, # Reduced
        "SVC": {'C': [0.1, 1.0], 'kernel': ['linear', 'rbf']}, # Reduced
        "KNeighborsClassifier": {'n_neighbors': [3, 5]}, # Reduced
        "DecisionTreeClassifier": {'max_depth': [None, 10, 20]},
        "GaussianNB": {}, # No typical hyperparameters to tune with GridSearchCV
        "XGBClassifier": {'n_estimators': [50, 100], 'learning_rate': [0.05, 0.1], 'max_depth': [3, 5]}, # Reduced
        "LGBMClassifier": {'n_estimators': [50, 100], 'learning_rate': [0.05, 0.1], 'num_leaves': [20, 31]} # Reduced
    }
    return param_grids

def train_and_evaluate_models(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, task_type: str, models_dir="models"):
    """Trains, tunes, and evaluates multiple models."""
    
    models_to_try = get_classification_models(task_type)
    param_grids = get_param_grids()
    
    results = []
    best_model_overall = None
    best_f1_overall = -1.0 # Use F1 as primary metric for selection
    best_model_name = ""

    # Ensure y_train and y_test are 1D arrays
    y_train = np.ravel(y_train)
    y_test = np.ravel(y_test)

    num_classes = len(np.unique(y_train))
    print(f"Number of classes in target: {num_classes}")

    for name, model in models_to_try.items():
        start_time = time.time()
        print(f"\nTraining {name}...")
        
        # Use StratifiedKFold for classification
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42) # Reduced splits for speed
        
        scoring_metric = 'f1_weighted' if task_type == "multiclass_classification" or num_classes > 2 else 'f1'
        if task_type == "binary_classification" and num_classes == 2:
            scoring_metric = 'roc_auc' # Prefer ROC AUC for binary if possible

        try:
            grid_search = GridSearchCV(model, param_grids.get(name, {}), cv=cv, scoring=scoring_metric, n_jobs=-1, error_score='raise')
            grid_search.fit(X_train, y_train)
            
            best_model = grid_search.best_estimator_
            
            y_pred_train = best_model.predict(X_train)
            y_pred_test = best_model.predict(X_test)
            
            accuracy_train = accuracy_score(y_train, y_pred_train)
            accuracy_test = accuracy_score(y_test, y_pred_test)
            
            precision_test = precision_score(y_test, y_pred_test, average='weighted', zero_division=0)
            recall_test = recall_score(y_test, y_pred_test, average='weighted', zero_division=0)
            f1_test = f1_score(y_test, y_pred_test, average='weighted', zero_division=0)
            
            roc_auc_test = None
            if task_type == "binary_classification" and hasattr(best_model, "predict_proba") and num_classes == 2:
                y_pred_proba_test = best_model.predict_proba(X_test)[:, 1]
                roc_auc_test = roc_auc_score(y_test, y_pred_proba_test)
            elif task_type == "multiclass_classification" and hasattr(best_model, "predict_proba") and num_classes > 2:
                 y_pred_proba_test = best_model.predict_proba(X_test)
                 try:
                     roc_auc_test = roc_auc_score(y_test, y_pred_proba_test, multi_class='ovr', average='weighted')
                 except ValueError as e: # Handle cases where roc_auc_score might fail for multiclass (e.g. not all classes in y_pred_proba)
                     print(f"Warning: ROC AUC for multiclass {name} failed: {e}")
                     roc_auc_test = None


            model_results = {
                "model_name": name,
                "best_params": grid_search.best_params_,
                "train_accuracy": accuracy_train,
                "test_accuracy": accuracy_test,
                "test_precision": precision_test,
                "test_recall": recall_test,
                "test_f1_score": f1_test,
                "test_roc_auc": roc_auc_test,
                "classification_report_test": classification_report(y_test, y_pred_test, zero_division=0, output_dict=True),
                "training_time_seconds": time.time() - start_time
            }
            results.append(model_results)
            print(f"{name} trained. Test F1: {f1_test:.4f}, Test ROC AUC: {roc_auc_test if roc_auc_test is not None else 'N/A'}")

            # Update best model based on F1 score (or ROC AUC for binary)
            current_metric_for_selection = roc_auc_test if scoring_metric == 'roc_auc' and roc_auc_test is not None else f1_test

            if current_metric_for_selection > best_f1_overall: # best_f1_overall now stores the best primary metric
                best_f1_overall = current_metric_for_selection
                best_model_overall = best_model
                best_model_name = name

        except Exception as e:
            print(f"Error training or evaluating {name}: {e}")
            results.append({
                "model_name": name,
                "error": str(e),
                "training_time_seconds": time.time() - start_time
            })

    if best_model_overall:
        print(f"\nBest model overall: {best_model_name} with primary metric score: {best_f1_overall:.4f}")
        os.makedirs(models_dir, exist_ok=True)
        model_path = os.path.join(models_dir, f"best_{best_model_name.lower()}_model.joblib")
        joblib.dump(best_model_overall, model_path)
        print(f"Best model saved to: {model_path}")
    else:
        print("\nNo model was successfully trained and selected.")
        model_path = None
        
    return results, best_model_name, best_model_overall, model_path