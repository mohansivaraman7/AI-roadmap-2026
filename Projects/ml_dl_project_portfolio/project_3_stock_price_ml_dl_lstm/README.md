# AAPL Stock Price Prediction (Machine Learning + Deep Learning + Sequential Data)

A real-world time-series forecasting project: predict tomorrow's AAPL
closing price from the last 10 trading days. This is the most advanced
of the three projects — it adds genuine **sequential data** handling
(an LSTM recurrent network) on top of the ML + DL combination from
project 2.

> **Disclaimer:** this is an educational demonstration of ML/DL
> forecasting techniques on historical data. It is **not financial
> advice** and should never be used to make real trading or investment
> decisions — short lightweight models like these do not reliably
> predict markets.

## Problem statement

Financial time series are sequential by nature: today's price depends
on a recent trajectory, not just isolated features. This project
builds two forecasters that consume the same 10-day window and compares
a classical lag-feature regressor against a recurrent neural network
that treats the window as a true sequence.

## Dataset

**AAPL (Apple Inc.) daily OHLCV history**, Feb 2015 – Feb 2017 — a real
market dataset published by Plotly as a finance sample.

- Source (downloaded automatically at train time, no API key needed):
  `https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv`
- 507 trading days; this project uses the `AAPL.Close` column as the
  target series.
- Data is **never shuffled** — a chronological 80/20 split is used so
  the model is always evaluated on days strictly after its training
  period, exactly like a real deployment would see the future.

## Project structure

```
project_3_stock_price_ml_dl_lstm/
├── README.md
├── requirements.txt
├── app.py                  # Flask real-time inference API (ML + LSTM ensemble)
├── data/
│   ├── raw/                 # downloaded CSV (git-ignored)
│   └── processed/           # (reserved for any exported intermediate series)
├── models/
│   ├── ml_model.pkl         # pickled RandomForestRegressor (lag-feature baseline)
│   ├── dl_model.pkl         # pickled Keras LSTM sequential model
│   ├── scaler.pkl           # pickled MinMaxScaler (fit on the training window only)
│   └── metadata.json        # metrics for both models + run info
├── src/
│   ├── config.py            # paths, URL, window size, hyperparameters
│   ├── data_loader.py        # downloads + loads the close-price time series
│   ├── preprocess.py         # scaling + sliding-window sequence construction
│   ├── train.py              # trains RF + LSTM, pickles both
│   └── predict.py            # loads both pickles, real-time next-day forecast
└── tests/
    └── test_predict.py       # smoke tests for the inference path
```

## How it works

1. `data_loader.py` downloads the raw OHLCV CSV and extracts the
   closing price as a date-sorted series — genuine sequential data,
   where row order carries information.
2. `preprocess.py` fits a `MinMaxScaler` **only on the training
   portion** (never on test-period values, to avoid look-ahead leakage)
   and builds sliding windows: each training example is 10 consecutive
   scaled closes (`X`) paired with the very next day's scaled close
   (`y`).
3. **Classical ML baseline:** a `RandomForestRegressor` treats the
   10-day window as 10 flat lag features.
4. **Deep learning / sequential model:** a Keras `LSTM(64) -> Dense(32)
   -> Dense(1)` network consumes the window as an actual
   `(timesteps, features)` sequence — the architecture built
   specifically for ordered data, unlike the tree model.
5. Both models are evaluated on the held-out chronological test period
   after inverse-scaling back to real USD prices (RMSE, MAE, R², MAPE).
6. Both models — the Random Forest **and** the LSTM — are serialized
   with plain `pickle`. Keras 3's LSTM models pickle natively, so no
   separate `.h5`/SavedModel export step is required; loading is a
   single `pickle.load` for either model.
7. `predict.py` accepts the most recent 10 real closing prices and
   returns a same-instant next-day forecast from the ML model, the
   LSTM, and a simple averaged ensemble — this is the "real-time" path:
   feed it live/latest prices and get an immediate forecast back.
8. `app.py` exposes this over a Flask REST API.

## Setup

```bash
cd project_3_stock_price_ml_dl_lstm
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Train

```bash
python -m src.train
```

Writes `models/ml_model.pkl`, `models/dl_model.pkl`, `models/scaler.pkl`
and `models/metadata.json` (RMSE/MAE/R²/MAPE for both models, in real
USD terms).

## Run real-time predictions

CLI (uses the most recent 10 real closes from the dataset):
```bash
python -m src.predict
```

REST API:
```bash
python app.py
curl -X POST http://127.0.0.1:5003/predict \
     -H "Content-Type: application/json" \
     -d '{"recent_closes": [128.72, 128.45, 129.5, 130.29, 131.53,
                             132.04, 132.42, 130.32, 130.29, 133.29]}'
```

Response:
```json
{
  "last_known_close": 133.29,
  "ml_model": {"predicted_next_close": 132.87, "direction": "down"},
  "dl_model": {"predicted_next_close": 133.94, "direction": "up"},
  "ensemble": {"predicted_next_close": 133.41, "direction": "up"},
  "disclaimer": "Educational project only -- not financial advice."
}
```

Use `/schema` to see how many trading days `recent_closes` must contain
(`WINDOW_SIZE`, currently 10).

## Tests

```bash
python -m pytest tests/ -v
```

## Standards followed

- Time series discipline: chronological split (no shuffling), scaler
  fit strictly on the training window to prevent data leakage from the
  future into preprocessing.
- Same config/data/preprocess/train/predict/serve layout as the other
  two projects for a consistent portfolio structure.
- Metrics are reported in real, interpretable units (USD, %) rather
  than only scaled-space loss values.
- Every trained artifact (including the recurrent network) is
  pickled and versioned in `models/`, loaded once and cached in the
  serving layer for real-time latency.
- A clear, prominent disclaimer accompanies every prediction, since
  stock forecasting carries real-world financial risk if misused.
