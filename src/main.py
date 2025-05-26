import argparse
import os
import pandas as pd
import joblib
from datetime import datetime

# Import from other modules in the src package
from .data_loader import load_data
from .task_identifier import identify_task_type
from .eda_generator import generate_eda_report
from .feature_engineer import preprocess_features
from .model_trainer import train_and_evaluate_models
from .reporter import generate_markdown_report

def main():
    parser = argparse.ArgumentParser(description="Automated Disease Diagnosis System")
    parser.add_argument("--data_path", type=str, help="Path to the CSV data file.")
    parser.add_argument("--target_column", required=True, type=str, help="Name of the target column in the dataset.")
    
    # MongoDB arguments (optional)
    parser.add_argument("--mongo_uri", type=str, help="MongoDB connection URI.")
    parser.add_argument("--db_name", type=str, help="MongoDB database name.")
    parser.add_argument("--collection_name", type=str, help="MongoDB collection name.")

    # Output directories
    parser.add_argument("--models_dir", type=str, default="models", help="Directory to save trained models.")
    parser.add_argument("--reports_dir", type=str, default="reports", help="Directory to save reports.")
    
    args = parser.parse_args()

    # Create output directories if they don't exist
    os.makedirs(args.models_dir, exist_ok=True)
    os.makedirs(args.reports_dir, exist_ok=True)
    os.makedirs(os.path.join(args.reports_dir, "eda"), exist_ok=True) # For EDA HTML

    # --- Initialize Report Data ---
    report_data_payload = {}
    start_time_main = datetime.now()
    print(f"Starting Automated Disease Diagnosis System at {start_time_main.strftime('%Y-%m-%d %H:%M:%S')}...")

    try:
        # 1. Load Data
        print("\n--- Step 1: Loading Data ---")
        df = load_data(
            data_path=args.data_path, 
            mongo_uri=args.mongo_uri, 
            db_name=args.db_name, 
            collection_name=args.collection_name,
            target_column=args.target_column
        )
        report_data_payload['data_info'] = {
            'data_path': args.data_path if args.data_path else f"MongoDB: {args.db_name}.{args.collection_name}",
            'shape': df.shape,
            'target_column': args.target_column,
            'columns': df.columns.tolist()
        }

        # 2. Identify Task Type
        print("\n--- Step 2: Identifying Task Type ---")
        task_type = identify_task_type(df[args.target_column])
        report_data_payload['task_type'] = task_type
        if task_type in ["single_class_error", "empty_target_error", "unknown_no_target"]:
            print(f"Exiting due to issue with target variable: {task_type}")
            generate_markdown_report(report_data_payload, output_dir=args.reports_dir)
            return

        # 3. Generate EDA Report
        print("\n--- Step 3: Generating EDA Report ---")
        eda_report_filename = f"eda_report_{args.target_column}"
        eda_report_path = generate_eda_report(df, args.target_column, report_title=eda_report_filename, output_dir=os.path.join(args.reports_dir, "eda"))
        report_data_payload['eda_report_path'] = eda_report_path

        # 4. Feature Engineering
        print("\n--- Step 4: Performing Feature Engineering ---")
        X_train, X_test, y_train, y_test, preprocessor, label_encoder = preprocess_features(df, args.target_column, task_type)
        
        # Save the preprocessor
        preprocessor_path = os.path.join(args.models_dir, "preprocessor_pipeline.joblib")
        joblib.dump(preprocessor, preprocessor_path)
        print(f"Preprocessor pipeline saved to: {preprocessor_path}")
        report_data_payload['saved_preprocessor_path'] = preprocessor_path

        # Summarize feature engineering for report
        fe_summary = {
            'num_features_count': len(preprocessor.transformers_[0][2]) if preprocessor.transformers_ and len(preprocessor.transformers_)>0 else 'N/A', # Assuming num is first
            'cat_features_count': len(preprocessor.transformers_[1][2]) if preprocessor.transformers_ and len(preprocessor.transformers_)>1 else 'N/A', # Assuming cat is second
            'X_train_shape': X_train.shape,
            'X_test_shape': X_test.shape,
            'label_encoder_classes': label_encoder.classes_.tolist() if label_encoder else None
        }
        report_data_payload['feature_engineering_summary'] = fe_summary


        # 5. Train and Evaluate Models
        print("\n--- Step 5: Training and Evaluating Models ---")
        model_results, best_model_name, _, saved_model_path = train_and_evaluate_models(
            X_train, y_train, X_test, y_test, task_type, models_dir=args.models_dir
        )
        report_data_payload['model_training_results'] = model_results
        report_data_payload['best_model_name'] = best_model_name
        report_data_payload['saved_model_path'] = saved_model_path
        
        # 6. Generate Final Report
        print("\n--- Step 6: Generating Final Report ---")
        final_report_path = generate_markdown_report(report_data_payload, output_dir=args.reports_dir)
        if final_report_path:
            print(f"Final Markdown report generated: {final_report_path}")
        else:
            print("Failed to generate final Markdown report.")

    except Exception as e:
        print(f"\nAn error occurred during the ADDS pipeline: {e}")
        import traceback
        traceback.print_exc()
        # Try to generate a partial report if an error occurs
        report_data_payload['error_summary'] = str(e)
        generate_markdown_report(report_data_payload, output_dir=args.reports_dir, filename_prefix="error_diagnosis_report")
        print("Partial error report attempted.")

    finally:
        end_time_main = datetime.now()
        print(f"\nAutomated Disease Diagnosis System finished at {end_time_main.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total execution time: {end_time_main - start_time_main}")

if __name__ == "__main__":
    main()





















#     **To use this:**
# 1.  Create the directory structure as commented at the beginning of the code block (`automated_disease_diagnosis/`, `src/`, `data/`, etc.).
# 2.  Save each Python script content into its respective file within the `src/` directory (e.g., `data_loader.py`, `main.py`).
# 3.  Save `README.md` and `requirements.txt` in the root `automated_disease_diagnosis/` directory.
# 4.  Follow the setup instructions in the `README.md` to install dependencies and run the system.
# 5.  Place your data CSV (e.g., `heart_disease.csv`) in the `data/` folder.
# 6.  Run `python src/main.py --data_path data/your_dataset.csv --target_column your_target_column_name` from the `automated_disease_diagnosis` directory.



# python src/main.py \
#     --mongo_uri "mongodb://your_mongodb_host:port/" \
#     --db_name "your_database_name" \
#     --collection_name "your_collection_name" \
#     --target_column "your_target_column_name_in_mongo_docs"