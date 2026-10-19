"""
Central configuration for the Diabetes Risk Prediction project.

Keeping paths, URLs and hyperparameters in one place is a basic project
standard: it avoids magic strings scattered across scripts and gives a
single point of change when the dataset source or storage layout changes.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

RAW_CSV_PATH = DATA_RAW_DIR / "pima_diabetes.csv"
TRAIN_CSV_PATH = DATA_PROCESSED_DIR / "train.csv"
TEST_CSV_PATH = DATA_PROCESSED_DIR / "test.csv"

MODEL_PATH = MODELS_DIR / "diabetes_model.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
METADATA_PATH = MODELS_DIR / "metadata.json"

# ---------------------------------------------------------------------------
# Dataset source (public, no authentication required)
# ---------------------------------------------------------------------------
DATASET_URL = (
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/"
    "pima-indians-diabetes.data.csv"
)

# The raw file ships without a header row; these are the well-known
# column names for the Pima Indians Diabetes Database (UCI / National
# Institute of Diabetes and Digestive and Kidney Diseases).
COLUMN_NAMES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome",
]

TARGET_COLUMN = "Outcome"

# Columns where a recorded 0 is not physiologically possible and really
# means "missing" (a well-documented quirk of this dataset).
ZERO_AS_MISSING_COLUMNS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

RANDOM_STATE = 42
TEST_SIZE = 0.2
