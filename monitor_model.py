
import requests
import time
import json
import random
import pandas as pd
import numpy as np
import os
from flask import Flask, request, jsonify
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CollectorRegistry, CONTENT_TYPE_LATEST
from prometheus_flask_exporter import PrometheusMetrics
import threading

# Configuration
MODEL_SERVER_URL = "http://localhost:5001/invocations"
# The Prometheus metrics will be served on a separate Flask app, e.g., port 8000
PROMETHEUS_PORT = 8000
PREDICTIONS_TO_GENERATE = 100 # Number of prediction requests to send for continuous monitoring

# Placeholder for the actual feature names (ensure they match your model's expected input)
FEATURE_NAMES = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

# Initialize Flask app for Prometheus metrics
app = Flask(__name__)
metrics = PrometheusMetrics(app)

# Define Prometheus metrics
# Counter for total prediction requests
PREDICTION_REQUESTS_TOTAL = Counter(
    'prediction_requests_total', 'Total number of prediction requests'
)

# Histogram for prediction latency
PREDICTION_LATENCY_SECONDS = Histogram(
    'prediction_latency_seconds', 'Latency of prediction requests in seconds',
    buckets=[.005, .01, .025, .05, .075, .1, .25, .5, .75, 1.0, 2.5, 5.0, 10.0]
)

# Gauge for distribution of predicted classes (0 and 1)
PREDICTION_CLASS_DISTRIBUTION = Gauge(
    'prediction_class_distribution', 'Distribution of predicted classes',
    ['class']
)

def generate_random_input(feature_names):
    """Generates a random input sample for the diabetes prediction model."""
    data = {
        'Pregnancies': random.randint(0, 17),
        'Glucose': random.randint(40, 200),
        'BloodPressure': random.randint(20, 122),
        'SkinThickness': random.randint(0, 99),
        'Insulin': random.randint(0, 846),
        'BMI': round(random.uniform(15.0, 60.0), 1),
        'DiabetesPedigreeFunction': round(random.uniform(0.07, 2.5), 3),
        'Age': random.randint(21, 81)
    }
    return [data[f] for f in feature_names]

def send_prediction_request_and_monitor():
    """Sends a prediction request and updates Prometheus metrics."""
    PREDICTION_REQUESTS_TOTAL.inc()
    input_sample = generate_random_input(FEATURE_NAMES)
    start_time = time.time()

    headers = {"Content-Type": "application/json"}
    payload = {
        "dataframe_split": {
            "columns": FEATURE_NAMES,
            "data": [input_sample]
        }
    }
    prediction = None
    try:
        response = requests.post(MODEL_SERVER_URL, headers=headers, json=payload)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        response_json = response.json()
        if 'predictions' in response_json:
            prediction = response_json['predictions'][0]
    except requests.exceptions.RequestException as e:
        print(f'Error sending request: {e}')

    latency = time.time() - start_time
    PREDICTION_LATENCY_SECONDS.observe(latency)

    # Update class distribution metrics
    if prediction is not None:
        # Increment counts for both classes to ensure both labels are always present
        # This is a common pattern to avoid missing labels when one class has 0 predictions
        if prediction == 0:
            PREDICTION_CLASS_DISTRIBUTION.labels(class='0').set(PREDICTION_CLASS_DISTRIBUTION.labels(class='0')._value + 1 if PREDICTION_CLASS_DISTRIBUTION.labels(class='0')._value is not None else 1)
            PREDICTION_CLASS_DISTRIBUTION.labels(class='1').set(PREDICTION_CLASS_DISTRIBUTION.labels(class='1')._value if PREDICTION_CLASS_DISTRIBUTION.labels(class='1')._value is not None else 0)
        elif prediction == 1:
            PREDICTION_CLASS_DISTRIBUTION.labels(class='1').set(PREDICTION_CLASS_DISTRIBUTION.labels(class='1')._value + 1 if PREDICTION_CLASS_DISTRIBUTION.labels(class='1')._value is not None else 1)
            PREDICTION_CLASS_DISTRIBUTION.labels(class='0').set(PREDICTION_CLASS_DISTRIBUTION.labels(class='0')._value if PREDICTION_CLASS_DISTRIBUTION.labels(class='0')._value is not None else 0)
    else:
        # If prediction failed, don't update distribution for this request
        pass

    return prediction

def continuous_prediction_generator():
    """Continuously generates predictions to simulate traffic for monitoring."""
    print(f'
Starting continuous prediction generator. Sending {PREDICTIONS_TO_GENERATE} requests...')
    for _ in range(PREDICTIONS_TO_GENERATE):
        send_prediction_request_and_monitor()
        time.sleep(random.uniform(0.1, 0.5)) # Simulate varying request intervals
    print('
Continuous prediction generator finished its cycle.')

@app.route('/metrics')
def metrics_endpoint():
    return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}

@app.route('/ping')
def ping():
    return jsonify({'status': 'ok'}), 200

def run_flask_app():
    print(f"
Starting Prometheus metrics server on http://0.0.0.0:{PROMETHEUS_PORT}")
    app.run(host='0.0.0.0', port=PROMETHEUS_PORT, debug=False, use_reloader=False)

if __name__ == '__main__':
    # Initialize gauges to 0 to ensure labels are always present
    PREDICTION_CLASS_DISTRIBUTION.labels(class='0').set(0)
    PREDICTION_CLASS_DISTRIBUTION.labels(class='1').set(0)

    # Start the Flask app in a separate thread
    flask_thread = threading.Thread(target=run_flask_app)
    flask_thread.daemon = True # Allow the main program to exit even if the thread is running
    flask_thread.start()

    # Give the Flask app a moment to start up
    time.sleep(2)

    # Start generating predictions
    continuous_prediction_generator()

    # Keep the main thread alive to allow Flask thread to run (optional, useful for colab)
    # In a real daemon, you might not need this if the Flask app is the main process
    # For demonstration, we'll keep it running for a while to allow checking metrics
    print(f"
Monitor script is running. Access Prometheus metrics at http://localhost:{PROMETHEUS_PORT}/metrics")
    print("To terminate, interrupt the kernel.")
    try:
        while True:
            time.sleep(1) # Keep main thread alive
    except KeyboardInterrupt:
        print("
Monitor script terminated.")
