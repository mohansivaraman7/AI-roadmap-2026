"""
Real-time inference API for the Diabetes Risk Prediction model.

Run:
    python app.py
Then:
    curl -X POST http://127.0.0.1:5001/predict -H "Content-Type: application/json" \
         -d '{"Pregnancies":6,"Glucose":148,"BloodPressure":72,"SkinThickness":35,
              "Insulin":0,"BMI":33.6,"DiabetesPedigreeFunction":0.627,"Age":50}'
"""
from flask import Flask, jsonify, request

from src.predict import FEATURE_ORDER, predict_one

app = Flask(__name__)


@app.get("/health")
def health():
    return jsonify({"status": "ok", "service": "diabetes-risk-prediction"})


@app.get("/schema")
def schema():
    return jsonify({"required_features": FEATURE_ORDER})


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True)
    if not payload:
        return jsonify({"error": "Send a JSON body with the required features."}), 400
    try:
        result = predict_one(payload)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)
