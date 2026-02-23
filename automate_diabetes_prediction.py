import pandas as pd
import os

def preprocess_diabetes_data(raw_data_path, processed_data_path):
    """Performs automatic preprocessing on the diabetes dataset.
    Expects a DataFrame with raw data and saves preprocessed data.
    Args:
        raw_data_path (str): The path to the raw diabetes dataset CSV file.
        processed_data_path (str): The path where the preprocessed data will be saved.
    Returns:
        pandas.DataFrame: The preprocessed DataFrame.
    """
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
    print("--- Running example usage of preprocess_diabetes_data ---")
    # Define example paths for raw and processed data
    example_raw_data_path = "/content/drive/MyDrive/Colab Notebooks/Demo7/diabetes_prediction_dataset.csv"
    example_processed_data_path = "/content/drive/MyDrive/Colab Notebooks/Demo7/data/processed/diabetes_processed_data.csv"

    # Ensure the output directory exists for the example
    os.makedirs(os.path.dirname(example_processed_data_path), exist_ok=True)

    processed_df_example = preprocess_diabetes_data(example_raw_data_path, example_processed_data_path)
    print(f"\nFirst 5 rows of preprocessed data (saved to {example_processed_data_path}):")
    print(processed_df_example.head())
    print("\nVerification of zero values after example preprocessing:")
    for col in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
        print(f"  - Column '{col}': {(processed_df_example[col] == 0).sum()} zero values")
