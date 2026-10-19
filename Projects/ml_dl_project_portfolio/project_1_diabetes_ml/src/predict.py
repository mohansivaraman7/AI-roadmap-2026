"""
Real-time inference using the pickled diabetes-risk model.

This module loads the pickled artifacts exactly once (module-level cache)
so repeated calls -- e.g. from the Flask API -- are fast, which is what
makes the API usable for real-time / low-latency predictions.
"""
import pickle
from functools import lru_cache

import pandas as pd

from . import config

FEATURE_ORDER = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]


@lru_cache(maxsize=1)
def _load_artifacts():
    with open(config.MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(config.SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    return model, scaler


def predict_one(features: dict) -> dict:
    """Predict diabetes risk for a single patient record.

    Parameters
    ----------
    features: dict with keys matching FEATURE_ORDER.

    Returns
    -------
    dict with the predicted class and the model's probability estimate.
    """
    model, scaler = _load_artifacts()

    missing = [c for c in FEATURE_ORDER if c not in features]
    if missing:
        raise ValueError(f"Missing required feature(s): {missing}")

    row = pd.DataFrame([[features[c] for c in FEATURE_ORDER]], columns=FEATURE_ORDER)
    row_scaled = scaler.transform(row)

    prediction = int(model.predict(row_scaled)[0])
    probability = float(model.predict_proba(row_scaled)[0, 1])

    return {
        "prediction": prediction,
        "label": "diabetic_risk" if prediction == 1 else "low_risk",
        "probability_diabetic": round(probability, 4),
    }


if __name__ == "__main__":
    sample = {
        "Pregnancies": 6,
        "Glucose": 148,
        "BloodPressure": 72,
        "SkinThickness": 35,
        "Insulin": 0,
        "BMI": 33.6,
        "DiabetesPedigreeFunction": 0.627,
        "Age": 50,
    }
    print(predict_one(sample))
