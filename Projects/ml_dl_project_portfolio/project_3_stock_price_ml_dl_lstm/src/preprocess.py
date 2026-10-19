"""
Turn a raw closing-price time series into supervised-learning examples:
a sliding window of the past `WINDOW_SIZE` days predicts the next day's
closing price. Used by both the ML baseline (flattened lag features)
and the LSTM (kept as a proper (timesteps, features) sequence).

Scaling is fit ONLY on the training portion and then applied to the
whole series, which avoids leaking information from the future (the
test period) into the transform -- a standard time-series hygiene rule.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from . import config


def chronological_split_index(n_rows: int) -> int:
    """Return the index that separates train from test, preserving time
    order (never shuffle time series data)."""
    return int(n_rows * (1 - config.TEST_FRACTION))


def fit_scaler_on_train(series: pd.Series) -> MinMaxScaler:
    split_idx = chronological_split_index(len(series))
    train_values = series.values[:split_idx].reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_values)
    return scaler


def make_windows(scaled_values: np.ndarray, window_size: int):
    """scaled_values: 1-D array. Returns X of shape (n, window_size) and
    y of shape (n,) where y[i] is the value immediately after window i."""
    X, y = [], []
    for i in range(len(scaled_values) - window_size):
        X.append(scaled_values[i : i + window_size])
        y.append(scaled_values[i + window_size])
    return np.array(X), np.array(y)


def build_datasets(series: pd.Series, window_size: int = config.WINDOW_SIZE):
    """Full pipeline: fit scaler on train portion, scale the whole
    series, window it, and split (chronologically) into train/test for
    both the flattened-ML view and the 3D LSTM view."""
    scaler = fit_scaler_on_train(series)
    scaled = scaler.transform(series.values.reshape(-1, 1)).ravel()

    X, y = make_windows(scaled, window_size)

    # windows are offset by `window_size` relative to the raw series, so
    # recompute the split index in windowed-sample space
    split_idx = chronological_split_index(len(series)) - window_size
    split_idx = max(split_idx, 1)

    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    return {
        "scaler": scaler,
        "X_train_ml": X_train,                                   # (n, window_size)
        "X_test_ml": X_test,
        "X_train_lstm": X_train.reshape(-1, window_size, 1),      # (n, window_size, 1)
        "X_test_lstm": X_test.reshape(-1, window_size, 1),
        "y_train": y_train,
        "y_test": y_test,
    }
