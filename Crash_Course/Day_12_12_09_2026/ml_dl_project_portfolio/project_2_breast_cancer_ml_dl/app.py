"""
Real-time inference API for the Breast Cancer Diagnosis (ML + DL) model.

Run:
    python app.py
Then:
    curl -X POST http://127.0.0.1:5002/predict -H "Content-Type: application/json" \
         -d '{"Clump_Thickness":5,"Uniformity_Cell_Size":4,"Uniformity_Cell_Shape":4,
              "Marginal_Adhesion":5,"Single_Epithelial_Cell_Size":7,"Bare_Nuclei":10,
              "Bland_Chromatin":3,"Normal_Nucleoli":2,"Mitoses":1}'
"""
from flask import Flask, jsonify, request

from src.predict import FEATURE_ORDER, predict_one

app = Flask(__name__)


@app.get("/health")
def health():
    return jsonify({"status": "ok", "service": "breast-cancer-diagnosis-ml-dl"})


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
    app.run(host="0.0.0.0", port=5002, debug=False)
