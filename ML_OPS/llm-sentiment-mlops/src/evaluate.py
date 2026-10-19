"""
Evaluation / promotion stage.

Loads the latest registered model version, scores it on the held-out
test set, and compares it against whichever version currently holds the
"champion" alias (i.e. what's serving production traffic). Only promotes
the new version if it's actually better -- this is the guardrail that
stops a worse model from silently replacing a good one.
"""
import argparse

import mlflow
import pandas as pd
import yaml
from mlflow import MlflowClient
from sklearn.metrics import accuracy_score, f1_score


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def score_model(model_uri: str, test_df: pd.DataFrame) -> dict:
    model = mlflow.transformers.load_model(model_uri, return_type="pipeline")
    preds = model(list(test_df["text"]))
    label_map = {"LABEL_0": 0, "LABEL_1": 1, "NEGATIVE": 0, "POSITIVE": 1}
    # NB: dict.get's default arg is evaluated eagerly, so int(p["label"]) can't
    # be passed as the default -- it would blow up on string labels like
    # "LABEL_1" even when that key IS in label_map. Use an explicit branch instead.
    pred_labels = [
        label_map[p["label"]] if p["label"] in label_map else int(p["label"])
        for p in preds
    ]

    return {
        "accuracy": accuracy_score(test_df["label"], pred_labels),
        "f1": f1_score(test_df["label"], pred_labels, average="weighted"),
    }


def evaluate_and_promote(config: dict, candidate_run_id: str = None):
    mlflow_cfg = config["mlflow"]
    mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
    client = MlflowClient()
    registered_name = mlflow_cfg["registered_model_name"]
    alias = mlflow_cfg["production_alias"]

    test_df = pd.read_csv(f"{config['data']['processed_dir']}/test.csv")

    # Candidate = the version just trained (latest), unless one is specified.
    versions = client.search_model_versions(f"name='{registered_name}'")
    if not versions:
        raise RuntimeError("No registered model versions found -- run train.py first")
    candidate_version = max(versions, key=lambda v: int(v.version))
    if candidate_run_id:
        candidate_version = next(v for v in versions if v.run_id == candidate_run_id)

    candidate_uri = f"models:/{registered_name}/{candidate_version.version}"
    candidate_metrics = score_model(candidate_uri, test_df)
    print(f"Candidate (v{candidate_version.version}) test metrics: {candidate_metrics}")

    # Compare against whatever currently holds the production alias, if any.
    try:
        current_champion = client.get_model_version_by_alias(registered_name, alias)
        champion_uri = f"models:/{registered_name}/{current_champion.version}"
        champion_metrics = score_model(champion_uri, test_df)
        print(f"Current champion (v{current_champion.version}) test metrics: {champion_metrics}")
    except mlflow.exceptions.MlflowException:
        champion_metrics = None
        print("No existing champion model -- candidate will be promoted by default")

    promote = champion_metrics is None or candidate_metrics["f1"] >= champion_metrics["f1"]

    if promote:
        client.set_registered_model_alias(registered_name, alias, candidate_version.version)
        print(f"Promoted v{candidate_version.version} to '{alias}'")
    else:
        print(f"Candidate v{candidate_version.version} did not beat champion -- not promoted")

    return promote, candidate_metrics, champion_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--run-id", default=None, help="Specific run to evaluate (defaults to latest)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    evaluate_and_promote(cfg, candidate_run_id=args.run_id)
