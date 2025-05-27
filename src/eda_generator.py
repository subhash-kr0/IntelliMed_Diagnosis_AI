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


