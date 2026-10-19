# Breast Cancer Diagnosis (Machine Learning + Deep Learning)

A real-world medical diagnostic-support project: given 9 cytology
measurements from a fine-needle aspirate, predict whether a tumor is
benign or malignant in real time. This project deliberately trains
**both** a classical ML model and a deep learning model and combines
them into a single ensemble, so it demonstrates the two paradigms
working together rather than picking one.

## Problem statement

A pathologist digitizes a handful of measurements from a biopsy sample.
Instead of waiting on manual review, this system gives an instant
first-pass risk classification that a clinician can use to triage
cases — a genuine (simplified) analogue of decision-support tools used
in real diagnostic pipelines.

## Dataset

**Breast Cancer Wisconsin (Original)** — donated by Dr. William H.
Wolberg, University of Wisconsin Hospitals, via the UCI Machine Learning
Repository.

- Source (downloaded automatically at train time, no API key needed):
  `https://raw.githubusercontent.com/jbrownlee/Datasets/master/breast-cancer-wisconsin.csv`
- 699 records, 9 integer cytology features (1–10 scale), target class
  `2` = benign / `4` = malignant (remapped to `0`/`1`).
- 16 records have a missing `Bare_Nuclei` value (marked `?` in the
  source); these are imputed with the column median.

## Project structure

```
project_2_breast_cancer_ml_dl/
├── README.md
├── requirements.txt
├── app.py                  # Flask real-time inference API (ML + DL ensemble)
├── data/
│   ├── raw/                 # downloaded CSV (git-ignored)
│   └── processed/           # train/test splits written by train.py
├── models/
│   ├── ml_model.pkl         # pickled best classical classifier
│   ├── dl_model.pkl         # pickled Keras dense neural network
│   ├── scaler.pkl           # pickled StandardScaler shared by both models
│   └── metadata.json        # metrics for every candidate + the DL network
├── src/
│   ├── config.py            # paths, URLs, hyperparameters
│   ├── data_loader.py        # downloads + loads the raw dataset
│   ├── preprocess.py         # cleaning, class remap, train/test split
│   ├── train.py              # trains ML models + Keras DNN, pickles both
│   └── predict.py            # loads both pickles, real-time ensemble inference
└── tests/
    └── test_predict.py       # smoke tests for the inference path
```

## How it works

1. `train.py` downloads the dataset, imputes the missing `Bare_Nuclei`
   values, remaps the target, and splits 80/20 (stratified).
2. **Classical ML side:** Logistic Regression, Random Forest and an
   RBF-kernel SVM are trained and 5-fold cross-validated; the best by
   held-out F1 is kept.
3. **Deep learning side:** a small fully-connected Keras network
   (`Dense(32) -> Dropout -> Dense(16) -> Dropout -> Dense(1, sigmoid)`)
   is trained with the Adam optimizer and binary cross-entropy loss.
4. Both the winning classical model and the neural network are
   serialized with **pickle** (Keras 3 models support native pickling,
   so the exact same `pickle.dump` / `pickle.load` pattern used for the
   scikit-learn model works for the neural network too — no separate
   `.h5`/SavedModel format needed).
5. `predict.py` loads all three pickles (ML model, DL model, shared
   scaler) once and returns predictions from **both** models plus a
   simple averaged-probability ensemble — a lightweight but genuine
   example of combining ML and DL at inference time.
6. `app.py` exposes this over a Flask REST API for real-time scoring.

## Setup

```bash
cd project_2_breast_cancer_ml_dl
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Train

```bash
python -m src.train
```

Writes `models/ml_model.pkl`, `models/dl_model.pkl`, `models/scaler.pkl`
and `models/metadata.json` with metrics for every model.

## Run real-time predictions

CLI:
```bash
python -m src.predict
```

REST API:
```bash
python app.py
curl -X POST http://127.0.0.1:5002/predict \
     -H "Content-Type: application/json" \
     -d '{"Clump_Thickness":5,"Uniformity_Cell_Size":4,"Uniformity_Cell_Shape":4,
          "Marginal_Adhesion":5,"Single_Epithelial_Cell_Size":7,"Bare_Nuclei":10,
          "Bland_Chromatin":3,"Normal_Nucleoli":2,"Mitoses":1}'
```

Response:
```json
{
  "ml_model": {"probability_malignant": 0.94, "label": "malignant"},
  "dl_model": {"probability_malignant": 0.97, "label": "malignant"},
  "ensemble": {"probability_malignant": 0.955, "label": "malignant"}
}
```

## Tests

```bash
python -m pytest tests/ -v
```

## Standards followed

- Same modular layout as the other two projects (config / data /
  preprocess / train / predict / serve) for consistency across the
  portfolio.
- A single shared `StandardScaler` feeds both the ML and DL models, so
  they see identical inputs — a common real-world requirement when
  ensembling heterogeneous models.
- TensorFlow log noise is suppressed at the module boundary
  (`TF_CPP_MIN_LOG_LEVEL`) rather than globally, keeping side effects
  local.
- Every trained artifact is pickled, versioned in `models/`, and loaded
  lazily/cached (`lru_cache`) in the serving layer for real-time
  latency.
- Automated smoke tests validate the inference contract for both
  models and the ensemble output.
