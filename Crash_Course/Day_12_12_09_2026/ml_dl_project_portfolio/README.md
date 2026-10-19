# ML / DL Project Portfolio

Three progressively more advanced, real-world machine learning projects,
each trained on a real public dataset fetched programmatically from the
internet, each shipping its trained model(s) as **pickle** artifacts,
and each exposed as a small real-time Flask inference API.

| # | Project | Paradigm | Dataset | Task |
|---|---------|----------|---------|------|
| 1 | [`project_1_diabetes_ml`](project_1_diabetes_ml/) | Machine Learning | Pima Indians Diabetes (UCI) | Binary classification — diabetes risk |
| 2 | [`project_2_breast_cancer_ml_dl`](project_2_breast_cancer_ml_dl/) | Machine Learning **+** Deep Learning | Breast Cancer Wisconsin (UCI) | Binary classification — benign vs malignant, ML + Keras DNN ensemble |
| 3 | [`project_3_stock_price_ml_dl_lstm`](project_3_stock_price_ml_dl_lstm/) | Machine Learning **+** Deep Learning **+** Sequential Data | AAPL daily closes (real market data) | Time-series regression — next-day closing price, Random Forest + LSTM |

Each project is self-contained (its own `requirements.txt`, `README.md`,
`src/`, `models/`, `tests/`) so they can be copied, run, or deployed
independently.

## Why these three, in this order

- **Project 1** is pure classical ML on static tabular data: the
  simplest end-to-end pattern (clean → split → train → pick best →
  pickle → serve).
- **Project 2** keeps tabular data but adds a second paradigm: a Keras
  neural network trained alongside the classical models, combined into
  an ensemble at inference time — showing ML and DL used together, not
  as alternatives.
- **Project 3** adds the third dimension the prompt asked for —
  **sequential data** — via an LSTM that consumes real time-ordered
  windows, compared against a lag-feature ML baseline. This is also the
  most "real-time" of the three in the literal sense: you feed it the
  latest known prices and get an instant next-step forecast.

## Common project layout (all three)

```
project_N_xxx/
├── README.md              # dataset, architecture, how to run, standards
├── requirements.txt        # pinned minimum dependency versions
├── app.py                  # Flask API: /health, /schema, /predict
├── data/
│   ├── raw/                 # downloaded dataset (git-ignored, fetched at train time)
│   └── processed/           # train/test splits written by the pipeline
├── models/
│   ├── *.pkl                 # pickled trained model(s) + preprocessing objects
│   └── metadata.json         # metrics for every candidate model
├── src/
│   ├── config.py             # single source of truth for paths/URLs/hyperparameters
│   ├── data_loader.py         # downloads + loads the raw dataset (no manual steps)
│   ├── preprocess.py          # cleaning / scaling / splitting, isolated from training
│   ├── train.py               # trains, evaluates, selects, and pickles the model(s)
│   └── predict.py             # loads pickles once, exposes a real-time predict function
└── tests/
    └── test_predict.py        # smoke tests for the inference contract
```

This mirrors standard applied-ML project structure (à la cookiecutter
data science): data, code and trained artifacts are kept in clearly
separated folders, configuration is centralized, and the inference path
is decoupled from both training and the web layer so it can be tested
and reused independently of Flask.

## Running any project

```bash
cd project_N_xxx
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.train        # downloads data, trains, evaluates, pickles the model(s)
python -m src.predict       # one real-time prediction from the command line
python app.py                # start the REST API
python -m pytest tests/ -v   # run the smoke tests
```

All three pipelines were run end-to-end while building this portfolio;
each `models/metadata.json` contains the actual metrics obtained.

## On "pickling"

Every project persists its trained model(s) with Python's `pickle`
module — including the deep-learning models. Keras 3 (bundled with the
`tensorflow-cpu` dependency used here) supports pickling its model
objects directly, so both the classical scikit-learn estimators and the
neural networks (dense network in project 2, LSTM in project 3) use the
exact same `pickle.dump` / `pickle.load` pattern. Any fitted
preprocessing object that must travel with a model (a `StandardScaler`
or `MinMaxScaler`) is pickled alongside it, since applying training-time
scaling consistently at inference time is required for correct
predictions.

## On "real-time"

Each project ships a small Flask API (`app.py`) that loads its pickled
artifacts once at process start (cached via `functools.lru_cache`) and
answers `/predict` requests immediately from in-memory models — this is
the real-time serving pattern: no retraining or batch job, just a
sub-second forward pass per request. Project 3 goes a step further: its
input is literally "the latest known values," so it is designed to be
called with live/just-observed prices to get an immediate forecast.

## Disclaimers

- Project 1 and 2 are simplified educational analogues of clinical
  decision-support tools — they are **not** validated medical devices
  and must never be used for real diagnosis or treatment decisions.
- Project 3 is an educational demonstration of time-series ML/DL
  techniques — it is **not financial advice** and must never be used
  for real trading or investment decisions.
