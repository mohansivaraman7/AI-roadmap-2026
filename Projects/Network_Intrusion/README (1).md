# Network Intrusion Detection with Machine Learning (NSL-KDD)

A complete ML project that classifies network connection records — the kind of
data a firewall, router, or IDS sensor logs (protocol, service, flags, byte
counts, connection rates, etc.) — as **normal traffic** or an **attack**.

## Dataset

**NSL-KDD**, the standard benchmark for network intrusion detection research
(an improved version of the older KDD Cup 1999 dataset with duplicate records
removed). Each row = one network connection.

| | Records | Normal | Attack |
|---|---|---|---|
| Train (`KDDTrain+`) | 125,973 | 67,343 | 58,630 |
| Test (`KDDTest+`) | 22,544 | 9,711 | 12,833 |

41 features per connection, covering three groups:
- **Basic connection features**: duration, protocol_type, service, flag, src_bytes, dst_bytes...
- **Content features**: num_failed_logins, root_shell, num_file_creations... (signs of suspicious behavior within a connection)
- **Traffic features**: count, srv_count, serror_rate, same_srv_rate... (statistics over recent connections to the same host/service — this is what a real IDS computes from a sliding window of traffic)

Attacks fall into 4 categories: **DoS** (denial of service), **Probe**
(scanning/surveillance), **R2L** (remote-to-local unauthorized access), **U2R**
(privilege escalation).

Note: the official test set deliberately includes attack *patterns not seen in
training* — this is what makes NSL-KDD a realistic benchmark and why accuracy
is meaningfully below 99% (a detector that memorizes training attacks won't
generalize, just like in the real world).

## Project structure

```
nids_project/
├── data/
│   ├── NSL_KDD_Train.csv
│   └── NSL_KDD_Test.csv
├── data_loader.py     # loads CSVs, assigns column names, maps attack labels
├── train.py            # preprocessing + training + evaluation pipeline
├── outputs/             # generated plots and metrics
└── README.md
```

## Pipeline

1. **Load & label** — assign the 41 official feature names; map ~22 fine-grained
   attack names (`neptune`, `smurf`, `satan`, `rootkit`, ...) to the 4 standard
   categories; derive a binary label (normal=0 / attack=1).
2. **Preprocess** — one-hot encode the 3 categorical features
   (`protocol_type`, `service`, `flag`) → 122 total features; standard-scale
   everything; align train/test columns (test set is missing a few rare
   `service` values).
3. **Train** 4 models for binary detection: Logistic Regression, Decision
   Tree, Random Forest, Gradient Boosting. Also train a Random Forest for the
   5-class (normal/dos/probe/r2l/u2r) problem.
4. **Evaluate** — accuracy/precision/recall/F1, confusion matrices, ROC/AUC,
   feature importances.

## Results (binary: normal vs. attack)

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 0.754 | 0.917 | 0.624 | 0.743 |
| Decision Tree | 0.783 | 0.958 | 0.647 | 0.772 |
| Random Forest | 0.771 | 0.968 | 0.618 | 0.755 |
| **Gradient Boosting** | **0.815** | **0.970** | **0.697** | **0.811** |

Precision is consistently high (>90%) across all models — when a model flags
a connection as an attack, it's almost always right. Recall is the harder
problem: catching *novel* attack variants in the test set that never
appeared in training, which is exactly what makes this benchmark realistic.

**Multiclass Random Forest**: 74.9% accuracy. DoS and Normal are detected
well (F1 ≈ 0.87 / 0.78); R2L and U2R — rare, subtle attacks with very few
training examples (995 and 52 records respectively) — are much harder to
catch (F1 ≈ 0.07 / 0.06), which mirrors real published results on this
dataset and reflects genuine class imbalance rather than a bug.

## Generated files (in `outputs/`)

- `model_comparison.png` — bar chart comparing all 4 models
- `confusion_matrix_binary.png` — Random Forest confusion matrix
- `confusion_matrix_multiclass.png` — 5-class confusion matrix
- `roc_curves.png` — ROC/AUC comparison across models
- `feature_importance.png` — top 15 most predictive features
- `results_summary.json` — all metrics in machine-readable form

## How to run

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
python3 train.py
```

## Ideas for extending this project

- Add **XGBoost/LightGBM** for a stronger boosted-tree baseline
- Handle class imbalance with **SMOTE** for the R2L/U2R classes
- Try **CICIDS2017** or **UNSW-NB15** for more modern attack types (DDoS, botnets, exploits)
- Build a **streaming/real-time** version using `river` for online learning on live traffic
- Add **SHAP** values for per-prediction explainability (useful for a real SOC analyst)
