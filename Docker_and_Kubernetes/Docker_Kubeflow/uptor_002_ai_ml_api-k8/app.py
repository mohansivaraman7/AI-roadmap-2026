import os
from datetime import datetime, timezone

from flask import Flask, request, jsonify
import joblib
import mlflow

app = Flask(__name__)

# Load model
model = joblib.load("house_model.pkl")

# Only set the tracking URI here — a pure in-memory config call, never
# touches the network. Anything that actually talks to MLflow (like
# set_experiment or start_run) happens lazily inside predict(), never at
# import time, so a down/misconfigured/unreachable MLflow server can
# never stop the API itself from starting.
mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
if mlflow_tracking_uri:
    mlflow.set_tracking_uri(mlflow_tracking_uri)


@app.route("/")
def home():
    return "House Price Prediction API"


@app.route("/predict", methods=["POST"])
def predict():

    data = request.json

    size = data["size"]

    prediction = model.predict([[size]])
    predicted_price = float(prediction[0])

    # Log every live prediction to MLflow for basic production monitoring —
    # never let a logging failure (MLflow server unreachable, misconfigured,
    # or any other MLflow-side error) break the actual API response.
    try:
        mlflow.set_experiment("house-price-predictions")
        run_name = f"predict-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
        with mlflow.start_run(run_name=run_name):
            mlflow.log_param("house_size", size)
            mlflow.log_metric("predicted_price", predicted_price)
    except Exception as exc:
        app.logger.warning("MLflow logging failed: %s", exc)

    return jsonify({
        "house_size": size,
        "predicted_price": predicted_price
    })


if __name__ == "__main__":
    app.run(debug=True)
