import pandas as pd
import logging

logging.basicConfig(filename='../logs/app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_clean_data(filepath):
    try:
        df = pd.read_csv(filepath)
        logging.info("Dataset loaded successfully.")
        # Data cleaning steps
        df.dropna(inplace=True)
        logging.info("Missing values handled.")
        return df
    except Exception as e:
        logging.error(f"Error in load_and_clean_data: {str(e)}")
        raise