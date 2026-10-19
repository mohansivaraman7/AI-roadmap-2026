"""Unit tests for the data prep stage -- the kind of fast, no-GPU test
that should run on every commit in CI, before any expensive training job."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_prep import load_config, prepare_data  # noqa: E402


def test_prepare_data_creates_expected_splits(tmp_path):
    config = load_config("configs/config.yaml")
    # Redirect output to a temp dir so the test doesn't clobber real data/processed
    config["data"]["processed_dir"] = str(tmp_path)

    paths = prepare_data(config)

    assert set(paths.keys()) == {"train", "val", "test"}
    for path in paths.values():
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0


def test_split_sizes_roughly_match_config(tmp_path):
    import pandas as pd

    config = load_config("configs/config.yaml")
    config["data"]["processed_dir"] = str(tmp_path)
    total_rows = len(pd.read_csv(config["data"]["raw_path"]))

    paths = prepare_data(config)
    train_n = len(pd.read_csv(paths["train"]))
    val_n = len(pd.read_csv(paths["val"]))
    test_n = len(pd.read_csv(paths["test"]))

    assert train_n + val_n + test_n == total_rows
    # test split should be roughly test_size of the total (allow rounding slack)
    assert abs(test_n / total_rows - config["data"]["test_size"]) < 0.1
