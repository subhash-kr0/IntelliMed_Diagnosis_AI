import pandas as pd

def identify_task_type(target_series: pd.Series):
    """
    Identifies the task type based on the target variable.
    Focuses on binary and multiclass classification.
    """
    if target_series is None:
        return "unknown_no_target"

    num_unique_values = target_series.nunique()
    dtype = target_series.dtype

    print(f"Target column analysis: Dtype={dtype}, Unique_Values={num_unique_values}")

    if num_unique_values == 2:
        print("Task identified as: Binary Classification")
        return "binary_classification"
    elif num_unique_values > 2:
        if pd.api.types.is_numeric_dtype(dtype) and num_unique_values > 20 and num_unique_values / len(target_series) > 0.2: 
             print(f"Task identified as: Potentially Regression (but treated as Multiclass Classification for this project if classes are distinct integers). Unique values: {num_unique_values}")

             return "multiclass_classification" 
        print("Task identified as: Multiclass Classification")
        return "multiclass_classification"
    elif num_unique_values == 1:
        print("Warning: Target column has only one unique value. Model training will not be meaningful.")
        return "single_class_error"
    else: 
        print("Warning: Target column is empty or has no unique values.")
        return "empty_target_error"

