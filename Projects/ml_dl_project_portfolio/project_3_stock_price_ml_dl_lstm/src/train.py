"""
Train a classical ML baseline (RandomForestRegressor on lag features)
AND a deep-learning LSTM (on proper sequential windows) to predict the
next day's AAPL closing price, then pickle both models.

Usage:
    python -m src.train
"""
import json
import logging
import os
import pickle
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from . import config
from .data_loader import download_dataset, load_close_series
from .preprocess import build_datasets

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def evaluate_in_price_scale(scaler, y_true_scaled, y_pred_scaled) -> dict:
    y_true = scaler.inverse_transform(y_true_scaled.reshape(-1, 1)).ravel()
    y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
    return {
        "rmse_usd": round(rmse, 4),
        "mae_usd": round(mae, 4),
        "r2_score": round(r2, 4),
        "mape_percent": round(mape, 4),
    }


def build_lstm_model(window_size: int):
    import keras
    from keras import layers

    inputs = layers.Input(shape=(window_size, 1))
    x = layers.LSTM(config.LSTM_UNITS, return_sequences=False)(inputs)
    x = layers.Dense(config.DENSE_UNITS, activation="relu")(x)
    outputs = layers.Dense(1)(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.LSTM_LEARNING_RATE),
        loss="mse",
        metrics=["mae"],
    )
    return model


def main() -> None:
    started = time.time()

    download_dataset()
    series = load_close_series()
    logger.info(
        "Loaded %d trading days of AAPL closes (%s -> %s)",
        len(series),
        series.index.min().date(),
        series.index.max().date(),
    )

    data = build_datasets(series, window_size=config.WINDOW_SIZE)
    scaler = data["scaler"]

    # ---------------------------------------------------------------
    # 1) Classical ML baseline: Random Forest on flattened lag features
    # ---------------------------------------------------------------
    ml_model = RandomForestRegressor(
        n_estimators=300, max_depth=8, random_state=config.RANDOM_STATE
    )
    ml_model.fit(data["X_train_ml"], data["y_train"])
    ml_pred = ml_model.predict(data["X_test_ml"])
    ml_metrics = evaluate_in_price_scale(scaler, data["y_test"], ml_pred)
    logger.info("[ML]  random_forest_lag_features -> %s", ml_metrics)

    # ---------------------------------------------------------------
    # 2) Deep learning: LSTM over genuine sequential windows
    # ---------------------------------------------------------------
    dl_model = build_lstm_model(config.WINDOW_SIZE)
    history = dl_model.fit(
        data["X_train_lstm"],
        data["y_train"],
        validation_split=0.1,
        epochs=config.LSTM_EPOCHS,
        batch_size=config.LSTM_BATCH_SIZE,
        verbose=0,
    )
    dl_pred = dl_model.predict(data["X_test_lstm"], verbose=0).ravel()
    dl_metrics = evaluate_in_price_scale(scaler, data["y_test"], dl_pred)
    dl_metrics["final_train_loss_mse_scaled"] = round(float(history.history["loss"][-1]), 6)
    dl_metrics["final_val_loss_mse_scaled"] = round(float(history.history["val_loss"][-1]), 6)
    logger.info("[DL]  lstm_sequential            -> %s", dl_metrics)

    # ---------------------------------------------------------------
    # 3) Persist everything with pickle
    # ---------------------------------------------------------------
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    with open(config.ML_MODEL_PATH, "wb") as f:
        pickle.dump(ml_model, f)
    with open(config.DL_MODEL_PATH, "wb") as f:
        pickle.dump(dl_model, f)  # Keras 3 LSTM model, pickled natively
    with open(config.SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    overall_best = "dl_model (LSTM)" if dl_metrics["rmse_usd"] <= ml_metrics["rmse_usd"] else "ml_model (RandomForest)"

    metadata = {
        "window_size": config.WINDOW_SIZE,
        "date_range": [str(series.index.min().date()), str(series.index.max().date())],
        "n_trading_days": len(series),
        "test_fraction": config.TEST_FRACTION,
        "ml_metrics": ml_metrics,
        "dl_metrics": dl_metrics,
        "overall_best_model": overall_best,
        "pickle_protocol": pickle.HIGHEST_PROTOCOL,
        "training_seconds": round(time.time() - started, 2),
        "disclaimer": (
            "Educational project only. Historical price prediction with "
            "these small models is NOT financial advice and should not be "
            "used for real trading decisions."
        ),
    }
    with open(config.METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Saved ML model  -> %s", config.ML_MODEL_PATH)
    logger.info("Saved DL model  -> %s", config.DL_MODEL_PATH)
    logger.info("Saved scaler    -> %s", config.SCALER_PATH)
    logger.info("Saved metadata  -> %s", config.METADATA_PATH)
    logger.info("Overall best model (lower RMSE): %s", overall_best)


if __name__ == "__main__":
    main()
