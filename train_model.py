import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
import joblib

# Load data
df = pd.read_csv('./data/survey.csv')

# Preprocessing
def preprocess_data(df):
    # Handle gender column
    df['Gender'] = df['Gender'].str.lower().str.strip()
    gender_map = {
        'female': 'female',
        'male': 'male',
        'm': 'male',
        'f': 'female',
        'woman': 'female',
        'man': 'male'
    }
    df['Gender'] = df['Gender'].apply(lambda x: gender_map.get(x, 'other'))
    
    # Select features and target
    features = ['Age', 'Gender', 'family_history', 'work_interfere', 'remote_work', 'tech_company']
    target = 'treatment'
    
    # Filter data
    df = df[features + [target]].dropna()
    
    # Convert categorical variables
    le = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        df[col] = le.fit_transform(df[col])
    
    return df

# Process data
processed_df = preprocess_data(df)
X = processed_df.drop('treatment', axis=1)
y = processed_df['treatment']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, './models/model.pkl')