import re

import pandas as pd
import requests
import os

def preprocess_diabetes_data(raw_data_path, processed_data_path):
    """Performs automatic preprocessing on the diabetes dataset.
    Expects raw data at raw_data_path and saves preprocessed data to processed_data_path.
    Args:
        raw_data_path (str): The path to the raw diabetes dataset CSV file.
        processed_data_path (str): The path where the preprocessed data will be saved.
    Returns:
        pandas.DataFrame: The preprocessed DataFrame.
    """
    import requests # Ensure requests is imported locally if not global
    import os # Ensure os is imported locally if not global
    import pandas as pd # Ensure pandas is imported locally

    # Ensure raw data directory exists
    os.makedirs(os.path.dirname(raw_data_path), exist_ok=True)

    # Download the dataset if it doesn't exist
    if not os.path.exists(raw_data_path):
        print(f"Downloading dataset from: https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv to {raw_data_path}")
        try:
            response = requests.get("https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv")
            response.raise_for_status() # Raise an exception for bad status codes
            with open(raw_data_path, 'wb') as f:
                f.write(response.content)
            print("Dataset downloaded successfully.")
        except requests.exceptions.RequestException as e:
            print(f"Error downloading the dataset: {e}")
            return None # Or raise an error
    else:
        print(f"Raw data already exists at: {raw_data_path}")

    print(f"Loading raw data from: {raw_data_path}")
    df_processed = pd.read_csv(raw_data_path)

    print("Applying median imputation for zero values...")
    columns_to_impute = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

    for col in columns_to_impute:
        non_zero_values = df_processed[df_processed[col] != 0][col]
        if not non_zero_values.empty:
            median_val = non_zero_values.median()
            df_processed[col] = df_processed[col].replace(0, median_val)
        else:
            print(f"  - Warning: Column '{col}' contains only zero values or is empty. No imputation performed.")

    print("Preprocessing complete.")
    print(f"Saving preprocessed data to: {processed_data_path}")
    os.makedirs(os.path.dirname(processed_data_path), exist_ok=True) # Ensure directory exists before saving
    df_processed.to_csv(processed_data_path, index=False)
    return df_processed

if __name__ == '__main__':
    # Define paths relative to the MLflow project root
    raw_data_file = './data/raw/diabetes_prediction_dataset.csv'
    processed_data_file = './data/processed/diabetes_processed_data.csv'

    # Run the preprocessing function
    processed_df = preprocess_diabetes_data(raw_data_file, processed_data_file)

    if processed_df is not None:
        print("
First 5 rows of preprocessed data:")
        print(processed_df.head())
        print("
Verification of zero values after preprocessing:")
        for col in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
            print(f"  - Column '{col}': {(processed_df[col] == 0).sum()} zero values")
    else:
        print("Preprocessing failed due to data download error or other issues.")