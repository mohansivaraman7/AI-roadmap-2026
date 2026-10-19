"""
Training stage.

Fine-tunes a small open-source transformer LLM (DistilBERT) for binary
sentiment classification, logging every run to MLflow (params, metrics,
and the model artifact itself) and registering the result in the MLflow
Model Registry. This is the piece that turns "a notebook that trains a
model" into something reproducible and comparable across runs.
"""
import argparse

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

    mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
    mlflow.set_experiment(mlflow_cfg["experiment_name"])

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["base_model"])
    model = AutoModelForSequenceClassification.from_pretrained(
        model_cfg["base_model"], num_labels=model_cfg["num_labels"]
    )

    def tokenize(batch):
        return tokenizer(
            batch["text"], truncation=True, max_length=model_cfg["max_length"]
        )

    train_ds = load_split(data_cfg["processed_dir"], "train").map(tokenize, batched=True)
    val_ds = load_split(data_cfg["processed_dir"], "val").map(tokenize, batched=True)

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    args = TrainingArguments(
        output_dir=train_cfg["output_dir"],
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
    parser.add_argument("--config", default="../configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train(cfg)
