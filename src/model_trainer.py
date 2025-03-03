# # Step 1: Import Required Libraries
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
#                              f1_score, roc_auc_score, confusion_matrix, 
#                              ConfusionMatrixDisplay, RocCurveDisplay)
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
# from xgboost import XGBClassifier
# from imblearn.over_sampling import SMOTE
# import joblib

# # Step 2: Load and Inspect Data
# df = pd.read_csv('./data/diabetes.csv')
# print(df.head())
# print("\nDataset Info:")
# print(df.info())
# print("\nDescriptive Statistics:")
# print(df.describe())

# # Step 3: Exploratory Data Analysis (EDA)
# # Check class distribution
# plt.figure(figsize=(6,4))
# sns.countplot(x='Outcome', data=df)
# plt.title('Class Distribution')
# plt.show()

# # Plot numerical features distributions
# num_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 
#  'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
# df[num_cols].hist(bins=20, figsize=(12,8))
# plt.tight_layout()
# plt.show()

# # Plot categorical features
# cat_cols = ['gender', 'hypertension', 'heart_disease', 'smoking_history']
# fig, axes = plt.subplots(2, 2, figsize=(12,8))
# for col, ax in zip(cat_cols, axes.flatten()):
#     sns.countplot(x=col, hue='diabetes', data=df, ax=ax)
# plt.tight_layout()
# plt.show()

# # Correlation matrix
# corr_matrix = df.corr()
# plt.figure(figsize=(10,8))
# sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
# plt.title('Correlation Matrix')
# plt.show()

# # Step 4: Data Preprocessing
# # Handle categorical variables
# df = pd.get_dummies(df, columns=['gender', 'smoking_history'], drop_first=True)

# # Handle potential missing values represented as 0 in biological features
# # Note: This dataset doesn't have explicit missing values, but we'll check for biological impossibilities
# biological_cols = ['blood_glucose_level', 'bmi', 'HbA1c_level']
# for col in biological_cols:
#     df[col] = df[col].replace(0, np.nan)
# df = df.dropna()

# # Split data into features and target
# X = df.drop('diabetes', axis=1)
# y = df['diabetes']

# # Handle class imbalance using SMOTE
# smote = SMOTE(random_state=42)
# X_res, y_res = smote.fit_resample(X, y)

# # Split data into train/test sets
# X_train, X_test, y_train, y_test = train_test_split(
#     X_res, y_res, test_size=0.2, random_state=42, stratify=y_res)

# # Feature Scaling
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# # Step 5: Model Training and Evaluation
# models = {
#     'Logistic Regression': LogisticRegression(max_iter=1000),
#     'Random Forest': RandomForestClassifier(),
#     'XGBoost': XGBClassifier(),
#     'SVM': SVC(probability=True)
# }

# results = {}

# for name, model in models.items():
#     model.fit(X_train_scaled, y_train)
#     y_pred = model.predict(X_test_scaled)
#     y_proba = model.predict_proba(X_test_scaled)[:,1]
    
#     metrics = {
#         'accuracy': accuracy_score(y_test, y_pred),
#         'precision': precision_score(y_test, y_pred),
#         'recall': recall_score(y_test, y_pred),
#         'f1': f1_score(y_test, y_pred),
#         'roc_auc': roc_auc_score(y_test, y_proba)
#     }
    
#     results[name] = metrics
    
#     # Plot confusion matrix
#     cm = confusion_matrix(y_test, y_pred)
#     disp = ConfusionMatrixDisplay(confusion_matrix=cm)
#     disp.plot()
#     plt.title(f'Confusion Matrix - {name}')
#     plt.show()
    
#     # Plot ROC curve
#     RocCurveDisplay.from_estimator(model, X_test_scaled, y_test)
#     plt.title(f'ROC Curve - {name}')
#     plt.show()

# # Compare model performance
# results_df = pd.DataFrame(results).T
# print("\nModel Performance Comparison:")
# print(results_df)

# # Step 6: Hyperparameter Tuning (Example with Random Forest)
# param_grid = {
#     'n_estimators': [100, 200, 300],
#     'max_depth': [None, 10, 20],
#     'min_samples_split': [2, 5, 10]
# }

# rf = RandomForestClassifier()
# grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='roc_auc')
# grid_search.fit(X_train_scaled, y_train)

# best_rf = grid_search.best_estimator_
# print(f"\nBest Random Forest Parameters: {grid_search.best_params_}")

# # Step 7: Feature Importance Analysis
# feature_importance = best_rf.feature_importances_
# features = X.columns
# importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importance})
# importance_df = importance_df.sort_values('Importance', ascending=False)

# plt.figure(figsize=(10,6))
# sns.barplot(x='Importance', y='Feature', data=importance_df)
# plt.title('Feature Importance')
# plt.show()

# # Step 8: Save Best Model
# joblib.dump(best_rf, 'diabetes_model.pkl')
# joblib.dump(scaler, 'scaler.pkl')





# Step 1: Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             ConfusionMatrixDisplay, RocCurveDisplay)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib

# Step 2: Load and Inspect Data
df = pd.read_csv('./data/diabetes.csv')
print(df.head())
print("\nDataset Info:")
print(df.info())
print("\nDescriptive Statistics:")
print(df.describe())

# Step 3: Exploratory Data Analysis (EDA)
# Check class distribution
plt.figure(figsize=(6,4))
sns.countplot(x='Outcome', data=df)
plt.title('Class Distribution')
plt.show()

# Plot numerical features distributions
num_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 
 'BMI', 'DiabetesPedigreeFunction', 'Age']

df[num_cols].hist(bins=20, figsize=(12,8))
plt.tight_layout()
plt.show()

# Correlation matrix
corr_matrix = df.corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Step 4: Data Preprocessing
# Handle potential missing values represented as 0 in biological features
biological_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in biological_cols:
    df[col] = df[col].replace(0, np.nan)

df.dropna(inplace=True)

# Split data into features and target
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

# Split data into train/test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42, stratify=y_res)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: Model Training and Evaluation
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(),
    'XGBoost': XGBClassifier(),
    'SVM': SVC(probability=True)  # Ensure probability=True for ROC AUC calculation
}

results = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test_scaled)[:,1]
    else:
        y_proba = model.decision_function(X_test_scaled)  # For SVM

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba)
    }
    
    results[name] = metrics
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f'Confusion Matrix - {name}')
    plt.show()
    
    # Plot ROC curve
    RocCurveDisplay.from_estimator(model, X_test_scaled, y_test)
    plt.title(f'ROC Curve - {name}')
    plt.show()

# Compare model performance
results_df = pd.DataFrame(results).T
print("\nModel Performance Comparison:")
print(results_df)

# Step 6: Hyperparameter Tuning (Example with Random Forest)
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

rf = RandomForestClassifier()
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='roc_auc')
grid_search.fit(X_train_scaled, y_train)

best_rf = grid_search.best_estimator_
print(f"\nBest Random Forest Parameters: {grid_search.best_params_}")

# Step 7: Feature Importance Analysis
feature_importance = best_rf.feature_importances_
features = X.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importance})
importance_df = importance_df.sort_values('Importance', ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Feature Importance')
plt.show()

# Step 8: Save Best Model
joblib.dump(best_rf, './models/diabetes_model.pkl')
joblib.dump(scaler, './models/scaler.pkl')
