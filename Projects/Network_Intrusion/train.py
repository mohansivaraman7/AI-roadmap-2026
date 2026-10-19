"""
Network Intrusion Detection System (NIDS) - training & evaluation pipeline.

Trains multiple classical ML models to classify network connection records
(as logged by a firewall / router / IDS) as normal or an attack, using the
NSL-KDD benchmark dataset. Produces:
  - A binary classifier (normal vs attack)
  - A multiclass classifier (normal / dos / probe / r2l / u2r)
  - Evaluation metrics, confusion matrices, ROC curve, feature importances
"""

import time
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)

from data_loader import load_dataset, COLUMN_NAMES

RANDOM_STATE = 42
CATEGORICAL_COLS = ["protocol_type", "service", "flag"]
FEATURE_COLS = [c for c in COLUMN_NAMES if c != "label"]

sns.set_style("whitegrid")


def preprocess(train_df, test_df):
    """One-hot encode categoricals (aligned across train/test), scale numerics."""
    X_train_raw = train_df[FEATURE_COLS].copy()
    X_test_raw = test_df[FEATURE_COLS].copy()

    X_train_enc = pd.get_dummies(X_train_raw, columns=CATEGORICAL_COLS)
    X_test_enc = pd.get_dummies(X_test_raw, columns=CATEGORICAL_COLS)

    # Align columns: test set may be missing some rare service/flag categories
    X_train_enc, X_test_enc = X_train_enc.align(X_test_enc, join="left", axis=1, fill_value=0)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_enc)
    X_test_scaled = scaler.transform(X_test_enc)

    return X_train_scaled, X_test_scaled, list(X_train_enc.columns), scaler


def train_and_evaluate_binary(X_train, y_train, X_test, y_test, feature_names, outdir):
    print("\n" + "=" * 60)
    print("BINARY CLASSIFICATION: normal vs attack")
    print("=" * 60)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(max_depth=15, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=20, n_jobs=-1, random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, max_depth=4, random_state=RANDOM_STATE),
    }

    results = {}
    roc_data = {}
    for name, model in models.items():
        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        results[name] = {
            "accuracy": acc, "precision": prec, "recall": rec, "f1_score": f1,
            "train_time_sec": round(train_time, 2)
        }
        print(f"\n{name}  (trained in {train_time:.1f}s)")
        print(f"  Accuracy={acc:.4f}  Precision={prec:.4f}  Recall={rec:.4f}  F1={f1:.4f}")

        if y_proba is not None:
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_data[name] = (fpr, tpr, auc(fpr, tpr))

        if name == "Random Forest":
            best_model = model
            best_pred = y_pred

    # Confusion matrix for best model (Random Forest)
    cm = confusion_matrix(y_test, best_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal", "Attack"], yticklabels=["Normal", "Attack"])
    plt.title("Confusion Matrix - Random Forest (Binary)")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(f"{outdir}/confusion_matrix_binary.png", dpi=150)
    plt.close()

    # ROC curves
    plt.figure(figsize=(6, 5))
    for name, (fpr, tpr, auc_val) in roc_data.items():
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves - Binary Classification")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{outdir}/roc_curves.png", dpi=150)
    plt.close()

    # Feature importance (Random Forest)
    importances = pd.Series(best_model.feature_importances_, index=feature_names)
    top_features = importances.sort_values(ascending=False).head(15)
    plt.figure(figsize=(7, 6))
    sns.barplot(x=top_features.values, y=top_features.index, color="#4C72B0")
    plt.title("Top 15 Feature Importances - Random Forest")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(f"{outdir}/feature_importance.png", dpi=150)
    plt.close()

    # Model comparison bar chart
    comp_df = pd.DataFrame(results).T[["accuracy", "precision", "recall", "f1_score"]]
    comp_df.plot(kind="bar", figsize=(8, 5))
    plt.title("Model Comparison - Binary Classification")
    plt.ylabel("Score")
    plt.ylim(0.8, 1.0)
    plt.xticks(rotation=20)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{outdir}/model_comparison.png", dpi=150)
    plt.close()

    return results, best_model


def train_and_evaluate_multiclass(X_train, y_train, X_test, y_test, outdir):
    print("\n" + "=" * 60)
    print("MULTICLASS CLASSIFICATION: normal / dos / probe / r2l / u2r")
    print("=" * 60)

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    model = RandomForestClassifier(n_estimators=200, max_depth=20, n_jobs=-1, random_state=RANDOM_STATE)
    model.fit(X_train, y_train_enc)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test_enc, y_pred)
    print(f"\nRandom Forest multiclass accuracy: {acc:.4f}\n")
    report = classification_report(y_test_enc, y_pred, target_names=le.classes_, zero_division=0)
    print(report)

    cm = confusion_matrix(y_test_enc, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges",
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title("Confusion Matrix - Random Forest (Multiclass)")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(f"{outdir}/confusion_matrix_multiclass.png", dpi=150)
    plt.close()

    return acc, report


def main():
    print("Loading NSL-KDD dataset...")
    train_df, test_df = load_dataset("data/NSL_KDD_Train.csv", "data/NSL_KDD_Test.csv")

    print("Preprocessing (one-hot encoding + scaling)...")
    X_train, X_test, feature_names, scaler = preprocess(train_df, test_df)
    print(f"Feature matrix shape after encoding: {X_train.shape}")

    outdir = "outputs"

    # Binary task
    y_train_bin = train_df["binary_label"].values
    y_test_bin = test_df["binary_label"].values
    binary_results, best_binary_model = train_and_evaluate_binary(
        X_train, y_train_bin, X_test, y_test_bin, feature_names, outdir
    )

    # Multiclass task
    y_train_multi = train_df["attack_category"].values
    y_test_multi = test_df["attack_category"].values
    multi_acc, multi_report = train_and_evaluate_multiclass(
        X_train, y_train_multi, X_test, y_test_multi, outdir
    )

    # Save metrics to JSON for the report
    summary = {
        "binary_classification": binary_results,
        "multiclass_accuracy": multi_acc,
        "multiclass_report": multi_report,
        "dataset": {
            "train_records": len(train_df),
            "test_records": len(test_df),
            "features_after_encoding": X_train.shape[1],
        }
    }
    with open(f"{outdir}/results_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nAll done. Plots and results saved to outputs/")


if __name__ == "__main__":
    main()
