"""
Real-time inference API for the AAPL next-day closing price predictor
(ML + DL + sequential/LSTM).

Run:
    python app.py
Then:
    curl -X POST http://127.0.0.1:5003/predict -H "Content-Type: application/json" \
         -d '{"recent_closes": [128.72, 128.45, 129.5, 130.29, 131.53,
                                 132.04, 132.42, 130.32, 130.29, 133.29]}'

`recent_closes` must contain exactly WINDOW_SIZE (see /schema) closing
prices in chronological order, oldest first.
"""
from flask import Flask, jsonify, request

from src import config
from src.predict import predict_next_close

app = Flask(__name__)


@app.get("/health")
def health():
    return jsonify({"status": "ok", "service": "aapl-stock-price-prediction-ml-dl-lstm"})


@app.get("/schema")
def schema():
    return jsonify(
        {
            "required_field": "recent_closes",
            "required_length": config.WINDOW_SIZE,
            "order": "chronological, oldest to newest",
        }
    )


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True)
    if not payload or "recent_closes" not in payload:
        return jsonify({"error": "Send a JSON body with a 'recent_closes' list."}), 400
    try:
        result = predict_next_close(payload["recent_closes"])
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5003, debug=False)
