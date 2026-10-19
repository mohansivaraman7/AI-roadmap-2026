"""
Serving stage.

A minimal FastAPI app that loads whichever model version currently holds
the "champion" alias in the MLflow Model Registry and exposes it as a
REST endpoint. Because it resolves the model by alias rather than a
hardcoded path, redeploying after evaluate.py promotes a new version
requires no code change -- just a restart (or a reload endpoint, for a
more advanced setup).
"""
import os
from pathlib import Path

import mlflow
import yaml
from fastapi import FastAPI
from pydantic import BaseModel

# See data_prep.py for why paths are resolved against PROJECT_ROOT rather
# than trusted to be relative to whatever the caller's cwd happens to be
# (this matters here too: `uvicorn src.serve:app` from the repo root vs.
# `uvicorn serve:app` from inside src/ leave the process with different cwds).
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def resolve_path(path: str) -> str:
    p = Path(path)
    return str(p if p.is_absolute() else PROJECT_ROOT / p)


def resolve_tracking_uri(uri: str) -> str:
    """See train.py for why this exists (URI-aware version of resolve_path)."""
    if "://" not in uri:
        return resolve_path(uri)
    if uri.startswith("sqlite:///") and not uri.startswith("sqlite:////"):
        rel = uri[len("sqlite:///"):]
        abs_path = (PROJECT_ROOT / rel).resolve()
        return f"sqlite:///{abs_path}"
    return uri


CONFIG_PATH = os.environ.get("CONFIG_PATH", str(PROJECT_ROOT / "configs" / "config.yaml"))

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

mlflow.set_tracking_uri(resolve_tracking_uri(config["mlflow"]["tracking_uri"]))

app = FastAPI(title="Sentiment Classifier API", version="1.0")

_model = None
_label_map = {"LABEL_0": "negative", "LABEL_1": "positive", "NEGATIVE": "negative", "POSITIVE": "positive"}


def get_model():
    """Lazy-load the model on first request so `import serve` (e.g. for
    tests) doesn't require MLflow artifacts to be present."""
    global _model
    if _model is None:
        registered_name = config["mlflow"]["registered_model_name"]
        alias = config["mlflow"]["production_alias"]
        model_uri = f"models:/{registered_name}@{alias}"
        _model = mlflow.transformers.load_model(model_uri, return_type="pipeline")
    return _model


class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    label: str
    score: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    model = get_model()
    result = model(req.text)[0]
    label = _label_map.get(result["label"], result["label"])
    return PredictResponse(label=label, score=result["score"])


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
