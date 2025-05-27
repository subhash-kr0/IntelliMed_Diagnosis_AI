import pandas as pd
from pymongo import MongoClient
import os

def load_csv_data(file_path: str):
    """Loads data from a CSV file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Error: CSV file not found at {file_path}")
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded data from {file_path}. Shape: {df.shape}")
        return df
    except Exception as e:
        raise Exception(f"Error loading CSV file {file_path}: {e}")

def load_mongodb_data(uri: str, db_name: str, collection_name: str):
    """Loads data from a MongoDB collection."""
    try:
        client = MongoClient(uri)
        db = client[db_name]
        collection = db[collection_name]
        data = list(collection.find())
        df = pd.DataFrame(data)
        
        if '_id' in df.columns:
            df.drop('_id', axis=1, inplace=True)
            
        print(f"Successfully loaded data from MongoDB: {db_name}.{collection_name}. Shape: {df.shape}")
        return df
    except Exception as e:
        raise Exception(f"Error loading data from MongoDB: {e}")

def load_data(data_path: str = None, mongo_uri: str = None, db_name: str = None, collection_name: str = None, target_column: str = None):
    """Loads data from CSV or MongoDB and validates target column."""
    if data_path:
        df = load_csv_data(data_path)
    elif mongo_uri and db_name and collection_name:
        df = load_mongodb_data(mongo_uri, db_name, collection_name)
    else:
        raise ValueError("Error: Either CSV data_path or MongoDB details (uri, db_name, collection_name) must be provided.")

    if target_column:
        if target_column not in df.columns:
            raise ValueError(f"Error: Target column '{target_column}' not found in the loaded dataset. Available columns: {df.columns.tolist()}")
    else:
        print("Warning: No target column specified. Some operations might be limited.")
        # Attempt to infer target if not provided (could be the last column, but risky)
        # For this project, we'll require it to be specified for clarity.
        raise ValueError("Error: Target column must be specified for diagnosis tasks.")
        
    return df