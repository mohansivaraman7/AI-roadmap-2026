"""
Data preparation stage.

Reads the raw labeled CSV, does a stratified train/val/test split, and
writes the splits to disk as CSVs. Kept as its own pipeline stage (rather
than inline in train.py) so it can be run, cached, and versioned
independently of training -- in a larger project this is the stage you'd
point DVC or a feature store at.
"""
import argparse
import os

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def prepare_data(config: dict) -> dict:
    raw_path = config["data"]["raw_path"]
    processed_dir = config["data"]["processed_dir"]
    test_size = config["data"]["test_size"]
    val_size = config["data"]["val_size"]
    seed = config["data"]["random_seed"]

    df = pd.read_csv(raw_path)
    if not {"text", "label"}.issubset(df.columns):
        raise ValueError("Expected columns 'text' and 'label' in raw data")

    # First split off the test set, then split remaining into train/val.
    train_val_df, test_df = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df["label"]
    )
    relative_val_size = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=relative_val_size,
        random_state=seed,
        stratify=train_val_df["label"],
    )

    os.makedirs(processed_dir, exist_ok=True)
    paths = {}
    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        out_path = os.path.join(processed_dir, f"{name}.csv")
        split_df.to_csv(out_path, index=False)
        paths[name] = out_path

    print(
        f"Prepared data -> train: {len(train_df)}, "
        f"val: {len(val_df)}, test: {len(test_df)} rows"
    )
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    prepare_data(cfg)
