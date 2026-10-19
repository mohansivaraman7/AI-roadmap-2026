"""
Minimal smoke tests for the diabetes prediction pipeline.
Run with: python -m pytest tests/ -v   (from the project root)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.predict import predict_one  # noqa: E402


def test_predict_one_returns_expected_keys():
    sample = {
        "Pregnancies": 2,
        "Glucose": 120,
        "BloodPressure": 70,
        "SkinThickness": 25,
        "Insulin": 80,
        "BMI": 28.5,
        "DiabetesPedigreeFunction": 0.35,
        "Age": 30,
    }
    result = predict_one(sample)
    assert "prediction" in result
    assert "probability_diabetic" in result
    assert result["prediction"] in (0, 1)
    assert 0.0 <= result["probability_diabetic"] <= 1.0


def test_predict_one_missing_feature_raises():
    incomplete = {"Pregnancies": 2, "Glucose": 120}
    try:
        predict_one(incomplete)
        assert False, "Expected ValueError for missing features"
    except ValueError:
        pass


if __name__ == "__main__":
    test_predict_one_returns_expected_keys()
    test_predict_one_missing_feature_raises()
    print("All tests passed.")
