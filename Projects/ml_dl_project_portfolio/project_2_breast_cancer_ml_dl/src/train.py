"""
Train classical ML models AND a Keras deep-learning model on the breast
cancer dataset, compare them, and persist both the best classical model
and the neural network with pickle.

Usage:
    python -m src.train
"""
import json
import logging
import os
import pickle
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # quiet TensorFlow logs

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from . import config
from .data_loader import download_dataset, load_raw_dataframe
from .preprocess import clean, save_processed_splits, split

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CANDIDATE_ML_MODELS = {
    "logistic_regression": LogisticRegression(max_iter=1000, random_state=config.RANDOM_STATE),
    "random_forest": RandomForestClassifier(
        n_estimators=300, max_depth=6, random_state=config.RANDOM_STATE
    ),
    "svm_rbf": SVC(kernel="rbf", probability=True, random_state=config.RANDOM_STATE),
}


def evaluate(y_test, y_pred, y_proba) -> dict:
    return {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
    }


def build_dl_model(input_dim: int):
    """A small fully-connected feed-forward neural network -- the 'deep
    learning' half of this project. Kept intentionally compact since the
    dataset is small (~700 rows); depth here means multiple non-linear
    hidden layers rather than sheer parameter count."""
    import keras
    from keras import layers

    inputs = layers.Input(shape=(input_dim,))
    x = inputs
    for units in config.DL_HIDDEN_UNITS:
        x = layers.Dense(units, activation="relu")(x)
        x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.DL_LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main() -> None:
    started = time.time()

    download_dataset()
    df = clean(load_raw_dataframe())
    X_train, X_test, y_train, y_test = split(df)
    save_processed_splits(X_train, X_test, y_train, y_test)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ---------------------------------------------------------------
    # 1) Classical ML models
    # ---------------------------------------------------------------
    ml_results = {}
    fitted_ml_models = {}
    for name, model in CANDIDATE_ML_MODELS.items():
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring="f1")
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        metrics = evaluate(y_test, y_pred, y_proba)
        metrics["cv_f1_mean"] = round(float(cv_scores.mean()), 4)
        metrics["cv_f1_std"] = round(float(cv_scores.std()), 4)
        ml_results[name] = metrics
        fitted_ml_models[name] = model
        logger.info("[ML] %-20s -> %s", name, metrics)

    best_ml_name = max(ml_results, key=lambda n: ml_results[n]["f1_score"])
    best_ml_model = fitted_ml_models[best_ml_name]
    logger.info("Best classical ML model: %s", best_ml_name)

    # ---------------------------------------------------------------
    # 2) Deep learning model (Keras dense neural network)
    # ---------------------------------------------------------------
    import numpy as np

    dl_model = build_dl_model(input_dim=X_train_scaled.shape[1])
    history = dl_model.fit(
        X_train_scaled,
        y_train.values,
        validation_split=0.15,
        epochs=config.DL_EPOCHS,
        batch_size=config.DL_BATCH_SIZE,
        verbose=0,
    )
    dl_proba = dl_model.predict(X_test_scaled, verbose=0).ravel()
    dl_pred = (dl_proba >= 0.5).astype(int)
    dl_metrics = evaluate(y_test, dl_pred, dl_proba)
    dl_metrics["final_train_accuracy"] = round(float(history.history["accuracy"][-1]), 4)
    dl_metrics["final_val_accuracy"] = round(float(history.history["val_accuracy"][-1]), 4)
    logger.info("[DL] dense_neural_network -> %s", dl_metrics)

    # ---------------------------------------------------------------
    # 3) Persist everything with pickle
    # ---------------------------------------------------------------
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    with open(config.ML_MODEL_PATH, "wb") as f:
        pickle.dump(best_ml_model, f)
    with open(config.DL_MODEL_PATH, "wb") as f:
        pickle.dump(dl_model, f)  # Keras 3 models support native pickling
    with open(config.SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    overall_best = "dl_model" if dl_metrics["f1_score"] >= ml_results[best_ml_name]["f1_score"] else f"ml_model ({best_ml_name})"

    metadata = {
        "feature_columns": list(X_train.columns),
        "target_column": config.TARGET_COLUMN,
        "class_labels": config.CLASS_LABELS,
        "best_ml_model_name": best_ml_name,
        "ml_results": ml_results,
        "dl_results": dl_metrics,
        "overall_best_model": overall_best,
        "pickle_protocol": pickle.HIGHEST_PROTOCOL,
        "training_seconds": round(time.time() - started, 2),
    }
    with open(config.METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Saved ML model  -> %s", config.ML_MODEL_PATH)
    logger.info("Saved DL model  -> %s", config.DL_MODEL_PATH)
    logger.info("Saved scaler    -> %s", config.SCALER_PATH)
    logger.info("Saved metadata  -> %s", config.METADATA_PATH)
    logger.info("Overall best model: %s", overall_best)


if __name__ == "__main__":
    main()
