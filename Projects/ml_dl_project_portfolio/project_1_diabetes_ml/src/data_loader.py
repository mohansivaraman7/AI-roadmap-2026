"""
Download and load the Pima Indians Diabetes dataset.

Real-world source: UCI Machine Learning Repository / National Institute of
Diabetes and Digestive and Kidney Diseases, mirrored as plain CSV on GitHub
so it can be fetched with a simple HTTP GET (no API key required).
"""
import logging
import urllib.request

import pandas as pd

from . import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def download_dataset(force: bool = False) -> None:
    """Download the raw dataset from the internet if not already present."""
    config.DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)

    if config.RAW_CSV_PATH.exists() and not force:
        logger.info("Raw dataset already present at %s (skip download).", config.RAW_CSV_PATH)
        return

    logger.info("Downloading dataset from %s ...", config.DATASET_URL)
    urllib.request.urlretrieve(config.DATASET_URL, config.RAW_CSV_PATH)
    logger.info("Saved raw dataset to %s", config.RAW_CSV_PATH)


def load_raw_dataframe() -> pd.DataFrame:
    """Load the raw CSV (no header) and attach human-readable column names."""
    if not config.RAW_CSV_PATH.exists():
        download_dataset()
    df = pd.read_csv(config.RAW_CSV_PATH, header=None, names=config.COLUMN_NAMES)
    return df


if __name__ == "__main__":
    download_dataset()
    frame = load_raw_dataframe()
    logger.info("Loaded %d rows, %d columns", *frame.shape)
    logger.info("\n%s", frame.head())
