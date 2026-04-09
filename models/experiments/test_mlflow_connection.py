import mlflow

TRACKING_URI = "http://35.232.122.64:5000"  # shared GCP MLflow server
EXPERIMENT   = "lunch-buddy"

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT)              # creates experiment if it doesn't exist yet

with mlflow.start_run(run_name="test-run"):
    mlflow.log_param("model",    "logistic_regression")   # dummy param
    mlflow.log_param("engineer", "Andrew")                # so we know who ran it
    mlflow.log_metric("accuracy", 0.92)                   # dummy metric

print(f"Run logged. View at {TRACKING_URI}/#/experiments/1")
