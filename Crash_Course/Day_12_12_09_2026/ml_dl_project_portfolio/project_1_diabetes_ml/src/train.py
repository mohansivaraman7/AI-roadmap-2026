"""
Train several classical ML models on the diabetes dataset, evaluate them,
and persist the best one (plus its feature scaler) with pickle.

Usage:
    python -m src.train
"""
import json
import logging
import pickle
import time

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

CANDIDATE_MODELS = {
    "logistic_regression": LogisticRegression(max_iter=1000, random_state=config.RANDOM_STATE),
    "random_forest": RandomForestClassifier(
        n_estimators=300, max_depth=6, random_state=config.RANDOM_STATE
    ),
    "svm_rbf": SVC(kernel="rbf", probability=True, random_state=config.RANDOM_STATE),
}


def evaluate(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
    }


def main() -> None:
    started = time.time()

    download_dataset()
    df = clean(load_raw_dataframe())
    X_train, X_test, y_train, y_test = split(df)
    save_processed_splits(X_train, X_test, y_train, y_test)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}
    fitted_models = {}
    for name, model in CANDIDATE_MODELS.items():
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring="f1")
        model.fit(X_train_scaled, y_train)
        metrics = evaluate(model, X_test_scaled, y_test)
        metrics["cv_f1_mean"] = round(cv_scores.mean(), 4)
        metrics["cv_f1_std"] = round(cv_scores.std(), 4)
        results[name] = metrics
        fitted_models[name] = model
        logger.info("%-20s -> %s", name, metrics)

    best_name = max(results, key=lambda n: results[n]["f1_score"])
    best_model = fitted_models[best_name]
    logger.info("Best model: %s (f1_score=%.4f)", best_name, results[best_name]["f1_score"])

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # --- Persist the trained model and its scaler with pickle ---
    with open(config.MODEL_PATH, "wb") as f:
        pickle.dump(best_model, f)
    with open(config.SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    metadata = {
        "best_model": best_name,
        "feature_columns": list(X_train.columns),
        "target_column": config.TARGET_COLUMN,
        "all_model_results": results,
        "chosen_model_metrics": results[best_name],
        "sklearn_pickle_protocol": pickle.HIGHEST_PROTOCOL,
        "training_seconds": round(time.time() - started, 2),
    }
    with open(config.METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Saved model -> %s", config.MODEL_PATH)
    logger.info("Saved scaler -> %s", config.SCALER_PATH)
    logger.info("Saved metadata -> %s", config.METADATA_PATH)


if __name__ == "__main__":
    main()
