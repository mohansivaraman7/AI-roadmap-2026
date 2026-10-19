# Diabetes Risk Prediction (Machine Learning)

A real-world clinical decision-support project: given a patient's basic
vitals, predict the risk of diabetes in real time. This is the "pure
ML" project of the three — classical, interpretable models trained on
tabular health data.

## Problem statement

Type 2 diabetes screening often relies on a handful of easily-measured
vitals. This project trains a binary classifier that flags high-risk
patients from 8 routine measurements, so a clinic could plug it into an
intake form and get an instant risk flag instead of waiting for lab
review.

## Dataset

**Pima Indians Diabetes Database** — National Institute of Diabetes and
Digestive and Kidney Diseases (via UCI Machine Learning Repository).

- Source (downloaded automatically at train time, no API key needed):
  `https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv`
- 768 patient records, 8 numeric features + binary outcome
  (`0` = no diabetes, `1` = diabetes).
- Features: `Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
  BMI, DiabetesPedigreeFunction, Age`.

The dataset is downloaded programmatically by `src/data_loader.py` — it is
not committed to the repo, in line with standard practice (see
`.gitignore` at the workspace root).

## Project structure

```
project_1_diabetes_ml/
├── README.md
├── requirements.txt
├── app.py                 # Flask real-time inference API
├── data/
│   ├── raw/                # downloaded CSV (git-ignored)
│   └── processed/          # train/test splits written by train.py
├── models/
│   ├── diabetes_model.pkl  # pickled best classifier
│   ├── scaler.pkl          # pickled StandardScaler (must match training)
│   └── metadata.json       # metrics for every candidate model
├── src/
│   ├── config.py           # paths, URLs, hyperparameters
│   ├── data_loader.py       # downloads + loads the raw dataset
│   ├── preprocess.py        # cleaning, imputation, train/test split
│   ├── train.py             # trains & compares models, pickles the best
│   └── predict.py           # loads pickle, real-time single-record inference
└── tests/
    └── test_predict.py      # smoke tests for the inference path
```

## How it works

1. `train.py` downloads the dataset, treats physiologically-impossible
   zeros (e.g. `Glucose == 0`) as missing and imputes them with the
   column median, then splits 80/20 (stratified).
2. Three candidate models are trained and 5-fold cross-validated:
   Logistic Regression, Random Forest, and an RBF-kernel SVM.
3. The model with the best held-out F1 score is selected automatically.
4. The winning model **and** the fitted `StandardScaler` are serialized
   with `pickle` into `models/`. The scaler must travel with the model
   because inference has to apply the identical transform used at
   training time.
5. `predict.py` loads both pickles once (cached) and exposes
   `predict_one(features: dict) -> dict` for low-latency, real-time
   scoring.
6. `app.py` wraps `predict.py` in a small Flask API so the model can be
   queried over HTTP, the way a real clinic intake system would call it.

## Setup

```bash
cd project_1_diabetes_ml
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Train the model

```bash
python -m src.train
```

This prints per-model metrics, then writes `models/diabetes_model.pkl`,
`models/scaler.pkl`, and `models/metadata.json`.

## Run real-time predictions

CLI:
```bash
python -m src.predict
```

REST API:
```bash
python app.py
# in another shell:
curl -X POST http://127.0.0.1:5001/predict \
     -H "Content-Type: application/json" \
     -d '{"Pregnancies":6,"Glucose":148,"BloodPressure":72,"SkinThickness":35,
          "Insulin":0,"BMI":33.6,"DiabetesPedigreeFunction":0.627,"Age":50}'
```

Response:
```json
{"prediction": 1, "label": "diabetic_risk", "probability_diabetic": 0.71}
```

## Tests

```bash
python -m pytest tests/ -v
```

## Standards followed

- Separation of concerns: config / data / preprocessing / training /
  inference / serving are each in their own module.
- No hard-coded paths — everything routes through `src/config.py`.
- Reproducibility: fixed `random_state`, stratified split, deterministic
  preprocessing.
- The trained artifact (model + scaler) is versionable and portable via
  `pickle`, decoupled from the training code.
- A thin serving layer (`app.py`) that only orchestrates
  request/response — all real logic lives in `src/`, so it stays testable
  outside of Flask.
- Basic automated tests guard the inference contract.
