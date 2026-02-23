
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import os

# Define paths
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DATA_PATH = os.path.join(PROJECT_DIR, 'data', 'processed', 'diabetes_processed_data.csv')
MLRUNS_DIR = os.path.join(PROJECT_DIR, 'mlruns')

# Set MLflow tracking URI
mlflow.set_tracking_uri("file://{}".format(MLRUNS_DIR))
mlflow.set_experiment("Diabetes_Prediction_Hyperparameter_Tuning")

def train_evaluate_log_model(data_path):
    print('Loading processed data from: {}'.format(data_path))
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print('Error: Processed data file not found at {}. Please ensure preprocessing is done.'.format(data_path))
        return

    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define hyperparameter grid for Logistic Regression
    hyperparameter_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'solver': ['liblinear', 'lbfgs']
    }

    run_count = 0
    for C_val in hyperparameter_grid['C']:
        for solver_val in hyperparameter_grid['solver']:
            run_count += 1
            with mlflow.start_run(run_name='LogisticRegression_C{}_Solver{}'.format(C_val, solver_val)) as run:
                print('Starting MLflow Run {} for C={}, solver={}'.format(run_count, C_val, solver_val))

                # Log hyperparameters manually
                mlflow.log_param("C", C_val)
                mlflow.log_param("solver", solver_val)

                # Train model
                model = LogisticRegression(C=C_val, solver=solver_val, random_state=42, max_iter=1000)
                model.fit(X_train, y_train)

                # Evaluate model
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, zero_division=0)
                recall = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)

                # Log metrics manually
                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("precision", precision)
                mlflow.log_metric("recall", recall)
                mlflow.log_metric("f1_score", f1)

                # Log model
                mlflow.sklearn.log_model(model, "logistic_regression_model")

                print('  Run ID: {}'.format(run.info.run_id))
                print('  Accuracy: {:.4f}'.format(accuracy))
                print('  Precision: {:.4f}'.format(precision))
                print('  Recall: {:.4f}'.format(recall))
                print('  F1-Score: {:.4f}'.format(f1))
                print('  Model and metrics logged to MLflow.')

    print('Hyperparameter tuning and MLflow logging complete.')

if __name__ == '__main__':
    print('--- Running modelling_tuning.py ---')
    train_evaluate_log_model(PROCESSED_DATA_PATH)
