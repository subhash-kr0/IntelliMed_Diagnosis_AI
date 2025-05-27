import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np

def preprocess_features(df: pd.DataFrame, target_column: str, task_type: str):
    """
    Performs automated feature engineering:
    - Splits data into train/test.
    - Handles missing values.
    - Encodes categorical features.
    - Scales numerical features.
    - Encodes target variable for classification.
    Returns X_train, X_test, y_train, y_test, and the preprocessing pipeline.
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame.")

    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # Identify feature types
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

    print(f"Numerical features: {numerical_features}")
    print(f"Categorical features: {categorical_features}")

    # Create preprocessing pipelines for numerical and categorical features
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')), # Median is robust to outliers
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')), # Or a constant like 'Missing'
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) # sparse_output=False for easier handling with pandas
    ])

    # Create a column transformer to apply different transformations to different columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, numerical_features),
            ('cat', categorical_pipeline, categorical_features)
        ], 
        remainder='passthrough' # Keep other columns (if any) not specified, though ideally all are handled
    )
    
    # Split data BEFORE applying the preprocessor fit to avoid data leakage
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if task_type != "regression" and y.nunique() > 1 else None) 

    # Fit the preprocessor on the training data and transform both train and test
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Get feature names after one-hot encoding
    try:
        # For scikit-learn >= 0.23
        feature_names_out = preprocessor.get_feature_names_out()
        X_train_processed_df = pd.DataFrame(X_train_processed, columns=feature_names_out, index=X_train.index)
        X_test_processed_df = pd.DataFrame(X_test_processed, columns=feature_names_out, index=X_test.index)
    except AttributeError: # Older scikit-learn
        # Fallback for older versions (less ideal, manual construction)
        # This part can be complex if there are many transformations.
        # For simplicity, we'll assume one-hot encoder is the main name changer.
        ohe_feature_names = []
        if categorical_features:
            ohe_feature_names = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)
        
        all_feature_names = numerical_features + list(ohe_feature_names)
        # This might not be perfectly robust for all preprocessor structures in older sklearn.
        # Consider updating sklearn or handling names more carefully if issues arise.
        if X_train_processed.shape[1] == len(all_feature_names):
             X_train_processed_df = pd.DataFrame(X_train_processed, columns=all_feature_names, index=X_train.index)
             X_test_processed_df = pd.DataFrame(X_test_processed, columns=all_feature_names, index=X_test.index)
        else: # Fallback to generic names if mismatch
            print("Warning: Could not accurately get feature names after preprocessing. Using generic names.")
            X_train_processed_df = pd.DataFrame(X_train_processed, index=X_train.index)
            X_test_processed_df = pd.DataFrame(X_test_processed, index=X_test.index)
            X_train_processed_df.columns = [f"feature_{i}" for i in range(X_train_processed_df.shape[1])]
            X_test_processed_df.columns = [f"feature_{i}" for i in range(X_test_processed_df.shape[1])]


    # Encode target variable (y) if it's classification and not already numeric
    label_encoder = None
    if task_type in ["binary_classification", "multiclass_classification"]:
        if not pd.api.types.is_numeric_dtype(y_train):
            label_encoder = LabelEncoder()
            y_train = label_encoder.fit_transform(y_train)
            y_test = label_encoder.transform(y_test)
            print(f"Target variable label encoded. Classes: {label_encoder.classes_}")
        else: # Ensure it's int if numeric
            y_train = y_train.astype(int)
            y_test = y_test.astype(int)


    print("Feature engineering complete.")
    print(f"X_train_processed shape: {X_train_processed_df.shape}")
    print(f"X_test_processed shape: {X_test_processed_df.shape}")
    
    return X_train_processed_df, X_test_processed_df, y_train, y_test, preprocessor, label_encoder
