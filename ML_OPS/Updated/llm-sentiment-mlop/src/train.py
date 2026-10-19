"""
Training stage.

Fine-tunes a small open-source transformer LLM (DistilBERT) for binary
sentiment classification, logging every run to MLflow (params, metrics,
and the model artifact itself) and registering the result in the MLflow
Model Registry. This is the piece that turns "a notebook that trains a
model" into something reproducible and comparable across runs.
"""
import argparse
from pathlib import Path

import mlflow
import mlflow.transformers
import numpy as np
import pandas as pd
import yaml
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

# See data_prep.py for why paths are resolved against PROJECT_ROOT rather
# than trusted to be relative to whatever the caller's cwd happens to be.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = str(PROJECT_ROOT / "configs" / "config.yaml")


def resolve_path(path: str) -> str:
    p = Path(path)
    return str(p if p.is_absolute() else PROJECT_ROOT / p)


def resolve_tracking_uri(uri: str) -> str:
    """Like resolve_path, but URI-aware.

    A bare path (legacy folder file-store, e.g. "mlruns") goes through
    resolve_path unchanged. A "sqlite:///relative/path.db" URI has a
    *relative* path by SQLAlchemy's own convention (3 slashes; 4 slashes
    means absolute) -- left alone it's just as cwd-dependent as a bare
    path was, so we rewrite it to an absolute 4-slash form anchored at
    PROJECT_ROOT. Any other URI (already-absolute sqlite, postgresql://,
    a remote tracking server) is left exactly as configured.
    """
    if "://" not in uri:
        return resolve_path(uri)
    if uri.startswith("sqlite:///") and not uri.startswith("sqlite:////"):
        rel = uri[len("sqlite:///"):]
        abs_path = (PROJECT_ROOT / rel).resolve()
        return f"sqlite:///{abs_path}"
    return uri


def ensure_experiment(name: str, artifact_location: str) -> None:
    """Create the experiment with an explicit, PROJECT_ROOT-anchored
    artifact location if it doesn't exist yet, then make it active.

    Without this, a SQL-backed tracking store picks a default artifact
    root ("./mlartifacts") relative to the *caller's* cwd the first time
    the experiment is created -- the same cwd-dependence problem the
    tracking URI itself has, just one level down.
    """
    if mlflow.get_experiment_by_name(name) is None:
        mlflow.create_experiment(name, artifact_location=artifact_location)
    mlflow.set_experiment(name)


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_split(processed_dir: str, split: str) -> Dataset:
    df = pd.read_csv(f"{processed_dir}/{split}.csv")
    return Dataset.from_pandas(df.reset_index(drop=True))


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
        "precision": precision_score(labels, preds, average="weighted", zero_division=0),
        "recall": recall_score(labels, preds, average="weighted", zero_division=0),
    }


def train(config: dict):
    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["training"]
    mlflow_cfg = config["mlflow"]

    mlflow.set_tracking_uri(resolve_tracking_uri(mlflow_cfg["tracking_uri"]))
    artifact_location = Path(resolve_path(mlflow_cfg["artifact_location"])).as_uri()
    ensure_experiment(mlflow_cfg["experiment_name"], artifact_location)

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["base_model"])
    model = AutoModelForSequenceClassification.from_pretrained(
        model_cfg["base_model"], num_labels=model_cfg["num_labels"]
    )

    def tokenize(batch):
        return tokenizer(
            batch["text"], truncation=True, max_length=model_cfg["max_length"]
        )

    processed_dir = resolve_path(data_cfg["processed_dir"])
    train_ds = load_split(processed_dir, "train").map(tokenize, batched=True)
    val_ds = load_split(processed_dir, "val").map(tokenize, batched=True)

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    args = TrainingArguments(
        output_dir=resolve_path(train_cfg["output_dir"]),
        num_train_epochs=train_cfg["epochs"],
        per_device_train_batch_size=train_cfg["batch_size"],
        per_device_eval_batch_size=train_cfg["batch_size"],
        learning_rate=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
        eval_strategy="epoch",
        save_strategy="no",
        logging_strategy="epoch",
        report_to=[],  # we log to MLflow ourselves for full control
        disable_tqdm=True,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    with mlflow.start_run() as run:
        mlflow.log_params(
            {
                "base_model": model_cfg["base_model"],
                "epochs": train_cfg["epochs"],
                "batch_size": train_cfg["batch_size"],
                "learning_rate": train_cfg["learning_rate"],
                "max_length": model_cfg["max_length"],
            }
        )

        trainer.train()
        eval_metrics = trainer.evaluate()
        mlflow.log_metrics(
            {k.replace("eval_", ""): v for k, v in eval_metrics.items() if isinstance(v, (int, float))}
        )

        # Log the fine-tuned model + tokenizer as an MLflow "transformers" model
        # and register it in the Model Registry in one step.
        model_info = mlflow.transformers.log_model(
            transformers_model={"model": model, "tokenizer": tokenizer},
            artifact_path="model",
            task="text-classification",  # explicit: avoids an HF Hub lookup to infer
            # the task, which fails for any model not pulled from the public hub.
            registered_model_name=mlflow_cfg["registered_model_name"],
        )

        print(f"Run ID: {run.info.run_id}")
        print(f"Registered model URI: {model_info.model_uri}")
        print(f"Validation metrics: {eval_metrics}")

    return run.info.run_id, eval_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()

    cfg = load_config(args.config)
    train(cfg)
