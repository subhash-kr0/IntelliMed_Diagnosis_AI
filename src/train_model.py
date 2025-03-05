import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
df = pd.read_csv('./data/indian_liver_patient.csv')
print(df.head())

# Data preprocessing
# Handle missing values
df['Albumin_and_Globulin_Ratio'].fillna(df['Albumin_and_Globulin_Ratio'].mean(), inplace=True)

# Convert gender to numerical
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])

# Save label encoder
joblib.dump(le, './models/label_encoder.joblib')

# Define features and target
X = df.drop(['Dataset'], axis=1)
y = df['Dataset'].map({2: 0, 1: 1})  # Convert to binary classification

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save scaler
joblib.dump(scaler, './models/scaler.joblib')

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

# Save model
joblib.dump(model, './models/liver_model.joblib')