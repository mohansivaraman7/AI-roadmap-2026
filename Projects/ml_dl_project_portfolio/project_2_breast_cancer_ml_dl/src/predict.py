"""
Real-time inference for breast cancer diagnosis, combining the pickled
classical ML model AND the pickled deep-learning (Keras) model into a
single ensemble prediction -- demonstrating ML + DL working together.
"""
import os
import pickle
from functools import lru_cache

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import pandas as pd

from . import config

FEATURE_ORDER = [
    "Clump_Thickness",
    "Uniformity_Cell_Size",
    "Uniformity_Cell_Shape",
    "Marginal_Adhesion",
    "Single_Epithelial_Cell_Size",
    "Bare_Nuclei",
    "Bland_Chromatin",
    "Normal_Nucleoli",
    "Mitoses",
]


@lru_cache(maxsize=1)
def _load_artifacts():
    with open(config.ML_MODEL_PATH, "rb") as f:
        ml_model = pickle.load(f)
    with open(config.DL_MODEL_PATH, "rb") as f:
        dl_model = pickle.load(f)
    with open(config.SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    return ml_model, dl_model, scaler


def predict_one(features: dict) -> dict:
    """Predict benign/malignant for a single cytology record using both
    the classical ML model and the deep learning model, plus a simple
    averaged ensemble."""
    ml_model, dl_model, scaler = _load_artifacts()

    missing = [c for c in FEATURE_ORDER if c not in features]
    if missing:
        raise ValueError(f"Missing required feature(s): {missing}")

    row = pd.DataFrame([[features[c] for c in FEATURE_ORDER]], columns=FEATURE_ORDER)
    row_scaled = scaler.transform(row)

    ml_proba = float(ml_model.predict_proba(row_scaled)[0, 1])
    dl_proba = float(dl_model.predict(row_scaled, verbose=0).ravel()[0])
    ensemble_proba = (ml_proba + dl_proba) / 2.0

    def to_label(p: float) -> str:
        return config.CLASS_LABELS[1] if p >= 0.5 else config.CLASS_LABELS[0]

    return {
        "ml_model": {"probability_malignant": round(ml_proba, 4), "label": to_label(ml_proba)},
        "dl_model": {"probability_malignant": round(dl_proba, 4), "label": to_label(dl_proba)},
        "ensemble": {
            "probability_malignant": round(ensemble_proba, 4),
            "label": to_label(ensemble_proba),
        },
    }


if __name__ == "__main__":
    sample = {
        "Clump_Thickness": 5,
        "Uniformity_Cell_Size": 4,
        "Uniformity_Cell_Shape": 4,
        "Marginal_Adhesion": 5,
        "Single_Epithelial_Cell_Size": 7,
        "Bare_Nuclei": 10,
        "Bland_Chromatin": 3,
        "Normal_Nucleoli": 2,
        "Mitoses": 1,
    }
    print(predict_one(sample))
