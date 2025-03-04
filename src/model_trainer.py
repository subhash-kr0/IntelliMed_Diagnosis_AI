import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



# Load dataset (Note: Kaggle datasets require authentication)

# Consider downloading the file locally if URL doesn't work

df = pd.read_csv('data/Lung_Cancer_dataset.csv')  # Update with correct path/URL

df.head(10)

df.info()



# Data Preprocessing

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()



# Encode categorical columns (including binary features)

categorical_cols = ['GENDER','SMOKING', 'YELLOW_FINGERS', 'ANXIETY',

                    'PEER_PRESSURE', 'CHRONIC_DISEASE', 'FATIGUE', 'ALLERGY', 

                    'WHEEZING', 'ALCOHOL_CONSUMING', 'COUGHING', 

                    'SHORTNESS_OF_BREATH', 'SWALLOWING_DIFFICULTY', 

                    'CHEST_PAIN', 'LUNG_CANCER']



for col in categorical_cols:

    df[col] = le.fit_transform(df[col])



# Check encoded values

for col in categorical_cols:

    print(f"\n{col} value counts:")

    print(df[col].value_counts())



# Target distribution visualization

plt.figure(figsize=(6, 4))

plt.pie(df['LUNG_CANCER'].value_counts(), 

        labels=['No Cancer' if x == 0 else 'Cancer' for x in df['LUNG_CANCER'].value_counts().index],

        autopct='%1.1f%%')

plt.title('Lung Cancer Distribution')

plt.show()



# Check duplicate rows (entire row duplicates)

print(f"\nTotal duplicate rows: {df.duplicated().sum()}")



# Data Summary

print("\nData summary:")

print(df.describe())



# Visualization: Age distribution by cancer status

plt.figure(figsize=(10, 6))

sns.histplot(data=df, x='AGE', hue='LUNG_CANCER', element='step', bins=20, 

             palette={0: 'skyblue', 1: 'salmon'})

plt.title('Age Distribution by Lung Cancer Status')

plt.xlabel('Age')

plt.ylabel('Count')

plt.legend(['Healthy', 'Cancer'])

plt.show()



# Feature distributions

fig, axes = plt.subplots(4, 4, figsize=(20, 20))

axes = axes.flatten()



for idx, col in enumerate(df.columns):

    if col != 'LUNG_CANCER' and idx < len(axes):

        sns.histplot(data=df, x=col, bins=20, ax=axes[idx], 

                     hue='LUNG_CANCER', multiple="stack", 

                     palette={0: 'skyblue', 1: 'salmon'})

        axes[idx].set_title(f'{col} Distribution')

        

plt.tight_layout()

plt.show()



# Additional visualizations

plt.figure(figsize=(10, 6))

sns.boxplot(data=df, x='LUNG_CANCER', y='AGE', hue='GENDER', 

            palette={0: 'lightblue', 1: 'pink'})

plt.title('Age Distribution by Cancer Status and Gender')

plt.xticks([0, 1], ['No Cancer', 'Cancer'])

plt.xlabel('Lung Cancer Status')

plt.legend(title='Gender', labels=['Male', 'Female'])

plt.show()



# Correlation matrix

plt.figure(figsize=(16, 12))

corr = df.corr()

sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", 

            annot_kws={'size': 8}, linewidths=0.5)

plt.title('Feature Correlation Matrix')

plt.show()



# Model Training

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score, confusion_matrix



# Prepare data using ACTUAL dataset (removed make_classification)

X = df.drop(columns='LUNG_CANCER')

y = df['LUNG_CANCER']



# Split data

X_train, X_test, y_train, y_test = train_test_split(

    X, y, test_size=0.2, random_state=42, stratify=y

)



# Scale numerical features

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)



# Initialize models

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier

from xgboost import XGBClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier



models = {

    "LogisticRegression": LogisticRegression(max_iter=1000),

    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),

    "XGBClassifier": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),

    "KNN": KNeighborsClassifier(n_neighbors=5),

    "DecisionTree": DecisionTreeClassifier(random_state=42),

    "GradientBoosting": GradientBoostingClassifier(random_state=42),

    "AdaBoost": AdaBoostClassifier(random_state=42)

}



# Model training and evaluation

results = []

for name, model in models.items():

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    

    metrics = {

        'Model': name,

        'Accuracy': accuracy_score(y_test, y_pred),

        'Precision': precision_score(y_test, y_pred),

        'Recall': recall_score(y_test, y_pred),

        'F1': f1_score(y_test, y_pred)

    }

    results.append(metrics)

    

    print(f"\n{name} Classification Report:")

    print(classification_report(y_test, y_pred))



# Create results dataframe

results_df = pd.DataFrame(results).sort_values('F1', ascending=False)

print("\nModel Performance Comparison:")

print(results_df)



# Best model analysis (using XGBoost as example)

best_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

best_model.fit(X_train, y_train)

y_pred = best_model.predict(X_test)



# Confusion matrix

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 

            xticklabels=['Predicted Healthy', 'Predicted Cancer'],

            yticklabels=['Actual Healthy', 'Actual Cancer'])

plt.title('XGBoost Confusion Matrix')

plt.ylabel('True Label')

plt.xlabel('Predicted Label')

plt.show()




# Add at the end of your training script
import joblib

# Save the best model and scaler
joblib.dump(best_model, 'lung_cancer_model.pkl')
joblib.dump(scaler, 'scaler.pkl')