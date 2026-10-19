"""
Cleaning, class remapping and train/test splitting for the breast
cancer dataset.
"""
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Impute the small number of missing `Bare_Nuclei` values with the
    column median, and remap the target to {0, 1}."""
    df = df.copy()
    df["Bare_Nuclei"] = df["Bare_Nuclei"].fillna(df["Bare_Nuclei"].median())
    df[config.TARGET_COLUMN] = df[config.TARGET_COLUMN].map(config.CLASS_MAPPING)
    return df


def split(df: pd.DataFrame):
    X = df.drop(columns=[config.TARGET_COLUMN])
    y = df[config.TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test


def save_processed_splits(X_train, X_test, y_train, y_test) -> None:
    config.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train_df = X_train.copy()
    train_df[config.TARGET_COLUMN] = y_train.values
    test_df = X_test.copy()
    test_df[config.TARGET_COLUMN] = y_test.values
    train_df.to_csv(config.TRAIN_CSV_PATH, index=False)
    test_df.to_csv(config.TEST_CSV_PATH, index=False)
