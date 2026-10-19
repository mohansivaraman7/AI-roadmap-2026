"""
Download and load real historical AAPL (Apple Inc.) daily stock prices.

This is genuine sequential / time-series data: each row is one trading
day, and order matters -- unlike the tabular datasets in the other two
projects, rows cannot be shuffled without destroying the signal.
"""
import logging
import urllib.request

import pandas as pd

from . import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def download_dataset(force: bool = False) -> None:
    config.DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    if config.RAW_CSV_PATH.exists() and not force:
        logger.info("Raw dataset already present at %s (skip download).", config.RAW_CSV_PATH)
        return
    logger.info("Downloading dataset from %s ...", config.DATASET_URL)
    urllib.request.urlretrieve(config.DATASET_URL, config.RAW_CSV_PATH)
    logger.info("Saved raw dataset to %s", config.RAW_CSV_PATH)


def load_close_series() -> pd.Series:
    """Return the closing price as a chronologically-sorted pandas Series
    indexed by date."""
    if not config.RAW_CSV_PATH.exists():
        download_dataset()
    df = pd.read_csv(config.RAW_CSV_PATH, parse_dates=[config.DATE_COLUMN])
    df = df.sort_values(config.DATE_COLUMN).reset_index(drop=True)
    series = df.set_index(config.DATE_COLUMN)[config.CLOSE_COLUMN]
    return series


if __name__ == "__main__":
    download_dataset()
    s = load_close_series()
    logger.info("Loaded %d trading days: %s -> %s", len(s), s.index.min().date(), s.index.max().date())
    logger.info("\n%s", s.head())
