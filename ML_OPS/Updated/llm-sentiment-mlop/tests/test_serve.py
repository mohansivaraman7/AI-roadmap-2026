"""API-level tests for the serving layer. The model itself is monkeypatched
out so this suite runs in CI in seconds without needing a trained model or
MLflow artifacts on disk -- only test_predict_endpoint (skipped by default)
needs a real registered model."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from fastapi.testclient import TestClient

from src import serve  # noqa: E402


@pytest.fixture
def client():
    return TestClient(serve.app)


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_uses_mocked_model(client, monkeypatch):
    def fake_get_model():
        return lambda text: [{"label": "LABEL_1", "score": 0.987}]

    monkeypatch.setattr(serve, "get_model", fake_get_model)

    response = client.post("/predict", json={"text": "I love this!"})
    assert response.status_code == 200
    body = response.json()
    assert body["label"] == "positive"
    assert body["score"] == pytest.approx(0.987)
