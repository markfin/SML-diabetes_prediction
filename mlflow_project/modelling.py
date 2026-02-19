
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

# Configure MLflow to log to a local directory
MLFLOW_TRACKING_URI = "./output/mlruns"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

def load_data(file_path):
    """Loads the preprocessed diabetes dataset."""
    print(f"Loading preprocessed data from: {file_path}")
    df = pd.read_csv(file_path)
    return df

def train_model(df):
    """Splits data, trains a Logistic Regression model with GridSearchCV, and logs with MLflow."""
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Define the experiment name
    mlflow.set_experiment("Diabetes_Prediction_Model_Tuning")

    with mlflow.start_run():
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)

        # Define the model and parameters for GridSearchCV
        model = LogisticRegression(solver='liblinear', random_state=42)
        param_grid = {
            'C': [0.01, 0.1, 1, 10, 100],
            'max_iter': [100, 200, 300]
        }

        print("Starting GridSearchCV for Logistic Regression...")
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_

        print(f"Best parameters found: {best_params}")
        print(f"Best F1-score: {grid_search.best_score_:.4f}")

        # Log best parameters
        mlflow.log_params(best_params)

        # Make predictions
        y_pred = best_model.predict(X_test)

        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")

        # Log metrics
        mlflow.log_metrics({
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        })

        # Log the best model
        mlflow.sklearn.log_model(best_model, "best_logistic_regression_model")

        # Generate and log classification report as JSON
        report = classification_report(y_test, y_pred, output_dict=True)
        report_path = "classification_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=4)
        mlflow.log_artifact(report_path, "classification_reports")
        os.remove(report_path) # Clean up local file

        # Generate and log confusion matrix as PNG
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        cm_path = "confusion_matrix.png"
        plt.savefig(cm_path)
        mlflow.log_artifact(cm_path, "confusion_matrices")
        plt.close() # Close the plot to free memory
        os.remove(cm_path) # Clean up local file

        print("MLflow Run completed. View with: mlflow ui")


if __name__ == '__main__':
    # Define the path to the preprocessed data
    processed_data_path = "./data/processed/diabetes_processed_data.csv"

    # Ensure the MLflow output directory exists
    os.makedirs(MLFLOW_TRACKING_URI, exist_ok=True)

    # Check if the processed data file exists before attempting to load
    if os.path.exists(processed_data_path):
        df_processed = load_data(processed_data_path)
        train_model(df_processed)
    else:
        print(f"Error: Processed data file not found at {processed_data_path}.")
        print("Please ensure the preprocessing step has been executed.")
