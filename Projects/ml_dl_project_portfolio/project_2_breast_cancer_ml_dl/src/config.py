"""
Central configuration for the Breast Cancer Diagnosis (ML + DL) project.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

RAW_CSV_PATH = DATA_RAW_DIR / "breast_cancer_wisconsin.csv"
TRAIN_CSV_PATH = DATA_PROCESSED_DIR / "train.csv"
TEST_CSV_PATH = DATA_PROCESSED_DIR / "test.csv"

ML_MODEL_PATH = MODELS_DIR / "ml_model.pkl"          # best classical ML model
DL_MODEL_PATH = MODELS_DIR / "dl_model.pkl"          # Keras neural network (pickled)
SCALER_PATH = MODELS_DIR / "scaler.pkl"
METADATA_PATH = MODELS_DIR / "metadata.json"

# ---------------------------------------------------------------------------
# Dataset source (public, no authentication required)
# ---------------------------------------------------------------------------
DATASET_URL = (
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/"
    "breast-cancer-wisconsin.csv"
)

# The raw mirror has no header and no sample-id column. These are the
# standard UCI "Breast Cancer Wisconsin (Original)" attribute names.
COLUMN_NAMES = [
    "Clump_Thickness",
    "Uniformity_Cell_Size",
    "Uniformity_Cell_Shape",
    "Marginal_Adhesion",
    "Single_Epithelial_Cell_Size",
    "Bare_Nuclei",
    "Bland_Chromatin",
    "Normal_Nucleoli",
    "Mitoses",
    "Class",
]

TARGET_COLUMN = "Class"
# Original encoding: 2 = benign, 4 = malignant. We remap to 0/1 for modeling.
CLASS_MAPPING = {2: 0, 4: 1}
CLASS_LABELS = {0: "benign", 1: "malignant"}

RANDOM_STATE = 42
TEST_SIZE = 0.2

# Keras Dense Neural Network hyperparameters
DL_HIDDEN_UNITS = [32, 16]
DL_EPOCHS = 60
DL_BATCH_SIZE = 16
DL_LEARNING_RATE = 1e-3
