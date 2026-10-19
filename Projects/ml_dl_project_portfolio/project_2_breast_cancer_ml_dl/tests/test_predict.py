"""
Minimal smoke tests for the breast cancer ML+DL inference pipeline.
Run with: python -m pytest tests/ -v   (from the project root)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.predict import predict_one  # noqa: E402

SAMPLE = {
    "Clump_Thickness": 3,
    "Uniformity_Cell_Size": 1,
    "Uniformity_Cell_Shape": 1,
    "Marginal_Adhesion": 1,
    "Single_Epithelial_Cell_Size": 2,
    "Bare_Nuclei": 2,
    "Bland_Chromatin": 3,
    "Normal_Nucleoli": 1,
    "Mitoses": 1,
}


def test_predict_one_has_ml_dl_and_ensemble_keys():
    result = predict_one(SAMPLE)
    for key in ("ml_model", "dl_model", "ensemble"):
        assert key in result
        assert 0.0 <= result[key]["probability_malignant"] <= 1.0
        assert result[key]["label"] in ("benign", "malignant")


def test_predict_one_missing_feature_raises():
    try:
        predict_one({"Clump_Thickness": 3})
        assert False, "Expected ValueError for missing features"
    except ValueError:
        pass


if __name__ == "__main__":
    test_predict_one_has_ml_dl_and_ensemble_keys()
    test_predict_one_missing_feature_raises()
    print("All tests passed.")
