# House Price Prediction API

A small Flask API that serves a scikit-learn linear regression model predicting house price from house size. Built as a learning project covering the full path from a plain Python script to CI, containerization, orchestration, and MLOps (experiment tracking + live monitoring with MLflow).

## Features

- `POST /predict` — returns a predicted price for a given house size
- Model trained with scikit-learn (`train_model.py`) and served with Flask + gunicorn
- **MLflow** tracks every training run (params, metrics, the model itself) and every live `/predict` call, for experiment tracking and basic production monitoring
- Jenkins pipeline that installs dependencies, retrains the model, and runs a live smoke test on every build
- Dockerfile to package the app into a portable container image
- `docker-compose.yml` to run the API, an MLflow tracking server, and a one-off training job together
- Kubernetes manifests to run and scale the containerized app
- Deployed on [Render](https://render.com)

## Project structure

```
.
├── app.py                              # Flask API: loads house_model.pkl, exposes POST /predict,
│                                        #   logs every prediction to MLflow
├── train_model.py                      # Trains the LinearRegression model, logs the run to MLflow,
│                                        #   saves house_model.pkl
├── house_model.pkl                     # Trained model artifact
├── requirements.txt                    # Python dependencies (includes mlflow-skinny)
├── test_prediction.py                  # Sends a sample request to /predict and prints the result
├── Jenkinsfile                         # CI pipeline: install deps -> retrain -> smoke test
├── Dockerfile                          # API container image
├── Dockerfile.mlflow                   # MLflow tracking server container image
├── docker-compose.yml                  # Runs api + mlflow + a one-off train job together
├── .dockerignore                       # Files excluded from the Docker build context
├── k8s/
│   ├── deployment.yaml                 # Runs the app as a Kubernetes Deployment (2 replicas)
│   └── service.yaml                    # Exposes the Deployment via a NodePort Service
└── docker-kubernetes-kubeflow-guide.md # Step-by-step guide: Docker, Kubernetes, and where Kubeflow fits
```

## Requirements

- Python 3.11+
- pip
- Docker + Docker Compose (optional, for the containerized / MLflow workflow)
- A local Kubernetes cluster such as Minikube or Kind + `kubectl` (optional, for the Kubernetes workflow)

## Running locally (no MLflow server)

```bash
python3 -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate

pip install -r requirements.txt

# (re)train the model — creates/overwrites house_model.pkl
python train_model.py

# start the API
python app.py
```

The API is now available at `http://127.0.0.1:5000`. Without `MLFLOW_TRACKING_URI` set, MLflow just writes run data into a local `./mlruns` folder instead of a server — training and prediction both still work, you just won't get the web UI.

## API usage

**Health check**

```bash
curl http://127.0.0.1:5000/
```

**Predict**

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"size": 1700}'
```

Response:

```json
{
  "house_size": 1700,
  "predicted_price": 85.0
}
```

## Testing

With the app running locally, run the sample client script:

```bash
python test_prediction.py
```

## Monitoring & MLOps with MLflow

This is the recommended way to run the project — it gives you a full loop: **train → track → serve → monitor**.

```bash
# 1. Start the MLflow tracking server (persists to a Docker volume)
docker compose up -d mlflow

# 2. Train the model — this run gets logged to the "house-price-training"
#    experiment (params, metrics, and the model artifact itself)
docker compose run --rm train

# 3. Start the API — every /predict call gets logged to the
#    "house-price-predictions" experiment for monitoring
docker compose up -d api
```

Open the MLflow UI at **http://localhost:5001** to see:

- **`house-price-training`** experiment — one run per time you retrained the model, with `model_type`, `training_rows`, `mse`, and `r2` logged, plus the serialized model as a downloadable artifact. Useful for comparing runs if you ever change the training data or try a different model.
- **`house-price-predictions`** experiment — one run per live prediction the API served, logging the requested `house_size` and the `predicted_price` returned. This is a lightweight stand-in for production monitoring: you can see prediction volume and watch for input values (e.g. unusually large or small house sizes) drifting outside the range the model was trained on (1000–2000 in this toy dataset).

Send a few requests and refresh the UI to watch new runs appear:

```bash
curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d '{"size": 1700}'
curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d '{"size": 2100}'
```

Tear everything down (and optionally wipe tracked history):

```bash
docker compose down          # stop containers, keep MLflow data
docker compose down -v       # also delete the mlflow_data volume
```

Notes on this setup: MLflow logging in `app.py` is wrapped in a try/except, so if the MLflow server is ever down, `/predict` still returns a prediction — it just skips logging that call, rather than failing the request. Also, logging happens synchronously inside the request today (simplest to follow for a learning project); a production system handling real traffic would typically push these logs asynchronously (e.g. via a queue) instead of logging inline on every request.

## Running with Docker (API only, no MLflow)

```bash
docker build -t house-price-api:1.0 .
docker run -p 5000:5000 house-price-api:1.0
```

This works fine — `app.py` and `train_model.py` fall back to local file-based MLflow tracking if no `MLFLOW_TRACKING_URI` is set, so the API still runs without the `mlflow` service. You just won't have a UI to see logged runs; use `docker compose` (above) for that.

## Running on Kubernetes

Requires a local cluster (Minikube or Kind) and `kubectl`.

```bash
docker build -t house-price-api:1.0 .

# make the image visible to the cluster
minikube image load house-price-api:1.0      # Minikube
# or: kind load docker-image house-price-api:1.0   # Kind

kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

kubectl get pods
minikube service house-price-api-service --url
```

The current manifests run the API only (MLflow isn't included in the Kubernetes setup yet — the docker-compose flow above is the way to run the full API + MLflow stack today). Full Docker/Kubernetes walkthrough, including scaling and self-healing demos, is in [`docker-kubernetes-kubeflow-guide.md`](./docker-kubernetes-kubeflow-guide.md).

## CI/CD

The `Jenkinsfile` runs on every build: sets up a virtualenv, installs dependencies, retrains the model, starts the API, runs a live smoke test against `/predict`, then tears everything down and archives `house_model.pkl` and the run log as build artifacts. It does not currently start MLflow — training and the smoke test fall back to local file-based tracking during CI runs.

## Deployment

The app is deployed on [Render](https://render.com), which builds and runs it directly from this repository.

## Learn more

See [`docker-kubernetes-kubeflow-guide.md`](./docker-kubernetes-kubeflow-guide.md) for a hands-on walkthrough of how this project moves from plain Python, through Docker and Kubernetes, and where a tool like Kubeflow would fit in at larger scale.
