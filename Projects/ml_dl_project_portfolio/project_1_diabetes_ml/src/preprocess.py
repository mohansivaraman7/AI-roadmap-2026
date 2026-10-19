"""
Cleaning, imputation and train/test splitting for the diabetes dataset.
"""
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Replace physiologically-impossible zeros with NaN, then impute with
    the column median (robust to outliers, simple and reproducible)."""
    df = df.copy()
    for col in config.ZERO_AS_MISSING_COLUMNS:
        df[col] = df[col].replace(0, pd.NA)
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())
    return df


def split(df: pd.DataFrame):
    """Stratified train/test split so the class balance is preserved."""
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
