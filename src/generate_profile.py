# generate_profile.py

import os
import pandas as pd
from ydata_profiling import ProfileReport

def generate_profile_report(csv_path, output_path):
    df = pd.read_csv(csv_path)
    profile = ProfileReport(df, title=f"Profiling Report - {os.path.basename(csv_path)}", explorative=True)
    profile.to_file(output_path)
    print(f"✅ Generated: {output_path}")

def generate_all_profiles(data_dir="data", report_dir="reports/profiling"):
    os.makedirs(report_dir, exist_ok=True)
    for file in os.listdir(data_dir):
        if file.endswith(".csv"):
            input_path = os.path.join(data_dir, file)
            output_path = os.path.join(report_dir, file.replace(".csv", "_profile.html"))
            generate_profile_report(input_path, output_path)

if __name__ == "__main__":
    generate_all_profiles()
