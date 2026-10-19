"""
Central configuration for the Stock Price Prediction (ML + DL +
Sequential data / LSTM) project.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

RAW_CSV_PATH = DATA_RAW_DIR / "aapl_stock.csv"
PROCESSED_CSV_PATH = DATA_PROCESSED_DIR / "aapl_close_series.csv"

ML_MODEL_PATH = MODELS_DIR / "ml_model.pkl"    # RandomForestRegressor on lag features
DL_MODEL_PATH = MODELS_DIR / "dl_model.pkl"    # Keras LSTM sequential model
SCALER_PATH = MODELS_DIR / "scaler.pkl"        # MinMaxScaler fit on the training close prices
METADATA_PATH = MODELS_DIR / "metadata.json"

# ---------------------------------------------------------------------------
# Dataset source (public, no authentication required)
# ---------------------------------------------------------------------------
# Real daily OHLCV history for Apple Inc. (AAPL), Feb 2015 - Feb 2017,
# published by Plotly as a sample finance dataset.
DATASET_URL = "https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv"

DATE_COLUMN = "Date"
CLOSE_COLUMN = "AAPL.Close"

# Sequential-data hyperparameters: how many past trading days feed one
# prediction of the next day's closing price.
WINDOW_SIZE = 10
TEST_FRACTION = 0.2  # chronological split -- the *last* 20% of days, never shuffled

RANDOM_STATE = 42

# LSTM hyperparameters
LSTM_UNITS = 64
DENSE_UNITS = 32
LSTM_EPOCHS = 80
LSTM_BATCH_SIZE = 8
LSTM_LEARNING_RATE = 1e-3
