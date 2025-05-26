# File: automated_disease_diagnosis/src/task_identifier.py
# =========================================
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
        # Heuristic: if it's object/category or integer with relatively few unique values, it's likely multiclass.
        # If it's float or int with many unique values, it could be regression, but this project focuses on classification.
        if pd.api.types.is_numeric_dtype(dtype) and num_unique_values > 20 and num_unique_values / len(target_series) > 0.2: # Arbitrary threshold
             print(f"Task identified as: Potentially Regression (but treated as Multiclass Classification for this project if classes are distinct integers). Unique values: {num_unique_values}")
             # For this project, we'll still treat it as multiclass if values are discrete.
             # A more robust system would differentiate regression better.
             return "multiclass_classification" # Or "regression" if explicitly handled
        print("Task identified as: Multiclass Classification")
        return "multiclass_classification"
    elif num_unique_values == 1:
        print("Warning: Target column has only one unique value. Model training will not be meaningful.")
        return "single_class_error"
    else: # 0 unique values (empty series)
        print("Warning: Target column is empty or has no unique values.")
        return "empty_target_error"

# =========================================
# File: automated_disease_diagnosis/src/eda_generator.py
# =========================================
import pandas as pd
from pandas_profiling import ProfileReport
import os

def generate_eda_report(df: pd.DataFrame, target_column: str, report_title: str = "Automated EDA Report", output_dir: str = "reports/eda"):
    """Generates an EDA report using pandas-profiling."""
    if df is None or df.empty:
        print("Error: DataFrame is empty. Cannot generate EDA report.")
        return None
        
    os.makedirs(output_dir, exist_ok=True)
    
    profile_title = f"{report_title} (Target: {target_column})" if target_column else report_title
    
    try:
        profile = ProfileReport(df, title=profile_title, explorative=True, minimal=False) # Use minimal=True for faster, smaller reports
        
        # Sanitize filename
        safe_title = "".join(c if c.isalnum() else "_" for c in report_title)
        report_filename = os.path.join(output_dir, f"{safe_title}.html")
        
        profile.to_file(report_filename)
        print(f"EDA report generated and saved to: {report_filename}")
        return report_filename
    except Exception as e:
        print(f"Error generating EDA report with pandas-profiling: {e}")
        print("Attempting with minimal=True as a fallback...")
        try:
            profile = ProfileReport(df, title=profile_title, explorative=True, minimal=True)
            report_filename = os.path.join(output_dir, f"{safe_title}_minimal.html")
            profile.to_file(report_filename)
            print(f"Minimal EDA report generated and saved to: {report_filename}")
            return report_filename
        except Exception as e_minimal:
            print(f"Error generating minimal EDA report: {e_minimal}")
            return None