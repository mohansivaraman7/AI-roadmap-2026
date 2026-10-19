"""
Real-time next-day AAPL closing-price prediction from the pickled
Random Forest (ML) baseline and the pickled LSTM (DL, sequential) model.

Feed in the most recent `WINDOW_SIZE` trading days' closing prices
(oldest -> newest) and get back a same-instant forecast for the next
trading day from both models plus a simple ensemble average.
"""
import os
import pickle
from functools import lru_cache

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np

from . import config


@lru_cache(maxsize=1)
def _load_artifacts():
    with open(config.ML_MODEL_PATH, "rb") as f:
        ml_model = pickle.load(f)
    with open(config.DL_MODEL_PATH, "rb") as f:
        dl_model = pickle.load(f)
    with open(config.SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    return ml_model, dl_model, scaler


def predict_next_close(recent_closes: list) -> dict:
    """Predict tomorrow's AAPL closing price from the last WINDOW_SIZE
    days of actual closing prices (in USD, oldest -> newest)."""
    ml_model, dl_model, scaler = _load_artifacts()

    if len(recent_closes) != config.WINDOW_SIZE:
        raise ValueError(
            f"Expected exactly {config.WINDOW_SIZE} recent closing prices, "
            f"got {len(recent_closes)}."
        )

    window = np.array(recent_closes, dtype=float).reshape(-1, 1)
    scaled_window = scaler.transform(window).ravel()

    ml_pred_scaled = ml_model.predict(scaled_window.reshape(1, -1))[0]
    dl_pred_scaled = dl_model.predict(
        scaled_window.reshape(1, config.WINDOW_SIZE, 1), verbose=0
    ).ravel()[0]

    ml_price = float(scaler.inverse_transform([[ml_pred_scaled]])[0, 0])
    dl_price = float(scaler.inverse_transform([[dl_pred_scaled]])[0, 0])
    ensemble_price = (ml_price + dl_price) / 2.0

    last_close = recent_closes[-1]

    def direction(pred_price: float) -> str:
        return "up" if pred_price > last_close else ("down" if pred_price < last_close else "flat")

    return {
        "last_known_close": round(float(last_close), 2),
        "ml_model": {"predicted_next_close": round(ml_price, 2), "direction": direction(ml_price)},
        "dl_model": {"predicted_next_close": round(dl_price, 2), "direction": direction(dl_price)},
        "ensemble": {
            "predicted_next_close": round(ensemble_price, 2),
            "direction": direction(ensemble_price),
        },
        "disclaimer": "Educational project only -- not financial advice.",
    }


if __name__ == "__main__":
    from .data_loader import load_close_series

    series = load_close_series()
    last_window = series.values[-config.WINDOW_SIZE :].tolist()
    print(predict_next_close(last_window))
