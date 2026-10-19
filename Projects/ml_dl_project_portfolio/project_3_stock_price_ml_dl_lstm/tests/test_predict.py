"""
Minimal smoke tests for the stock price ML+DL+LSTM inference pipeline.
Run with: python -m pytest tests/ -v   (from the project root)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.predict import predict_next_close  # noqa: E402


def test_predict_next_close_returns_expected_keys():
    fake_window = [120.0 + i * 0.5 for i in range(config.WINDOW_SIZE)]
    result = predict_next_close(fake_window)
    for key in ("ml_model", "dl_model", "ensemble"):
        assert key in result
        assert "predicted_next_close" in result[key]
        assert result[key]["predicted_next_close"] > 0


def test_predict_next_close_wrong_length_raises():
    try:
        predict_next_close([1.0, 2.0, 3.0])
        assert False, "Expected ValueError for wrong window length"
    except ValueError:
        pass


if __name__ == "__main__":
    test_predict_next_close_returns_expected_keys()
    test_predict_next_close_wrong_length_raises()
    print("All tests passed.")
