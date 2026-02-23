
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import os

# Define paths relative to the script's location
# Use os.path.dirname(os.path.abspath(__file__)) to get the script's directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT_DIR = os.path.join(SCRIPT_DIR, '..') # Go up one level from MLProject to project root

PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT_DIR, 'data', 'processed', 'diabetes_processed_data.csv')
MLRUNS_DIR = os.path.join(PROJECT_ROOT_DIR, 'mlruns')

# Set MLflow tracking URI
mlflow.set_tracking_uri(f"file://{MLRUNS_DIR}")
mlflow.set_experiment("Diabetes_Prediction_Model_Training")

def train_and_log_model(data_path):
    print(f"
Loading processed data from: {data_path}")
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: Processed data file not found at {data_path}. Please ensure preprocessing is done.")
        return

    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define model parameters
    C_param = 0.1 # Regularization parameter
    solver_param = 'liblinear'
    random_state_param = 42
    max_iter_param = 1000

    with mlflow.start_run(run_name="Single_Logistic_Regression_Training") as run:
        print(f"
Starting MLflow Run for C={C_param}, solver={solver_param}")

        # Manually log parameters
        mlflow.log_param("C", C_param)
        mlflow.log_param("solver", solver_param)
        mlflow.log_param("random_state", random_state_param)
        mlflow.log_param("max_iter", max_iter_param)

        # Create and train the model
        model = LogisticRegression(C=C_param, solver=solver_param, random_state=random_state_param, max_iter=max_iter_param)
        model.fit(X_train, y_train)

        # Make predictions and evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        # Manually log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # Log the model
        mlflow.sklearn.log_model(model, "logistic_regression_model")

        print(f"  Run ID: {run.info.run_id}")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print("  Model and metrics logged to MLflow.")

    print("
Model training and MLflow logging complete.")

if __name__ == '__main__':
    print("--- Running train_model.py ---")
    train_and_log_model(PROCESSED_DATA_PATH)
