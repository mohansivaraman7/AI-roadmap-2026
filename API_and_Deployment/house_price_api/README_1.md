# House Price Prediction API

A small Flask API that serves a scikit-learn linear regression model predicting house price from house size. Built as a learning project covering the full path from a plain Python script to CI, containerization, and orchestration.

## Features

- `POST /predict` — returns a predicted price for a given house size
- Model trained with scikit-learn (`train_model.py`) and served with Flask + gunicorn
- Jenkins pipeline that installs dependencies, retrains the model, and runs a live smoke test on every build
- Dockerfile to package the app into a portable container image
- Kubernetes manifests to run and scale the containerized app
- Deployed on [Render](https://render.com)

## Project structure

```
.
├── app.py                              # Flask API: loads house_model.pkl, exposes POST /predict
├── train_model.py                      # Trains the LinearRegression model, saves house_model.pkl
├── house_model.pkl                     # Trained model artifact
├── requirements.txt                    # Python dependencies
├── test_prediction.py                  # Sends a sample request to /predict and prints the result
├── Jenkinsfile                         # CI pipeline: install deps -> retrain -> smoke test
├── Dockerfile                          # Container image definition
├── .dockerignore                       # Files excluded from the Docker build context
├── k8s/
│   ├── deployment.yaml                 # Runs the app as a Kubernetes Deployment (2 replicas)
│   └── service.yaml                    # Exposes the Deployment via a NodePort Service
└── docker-kubernetes-kubeflow-guide.md # Step-by-step guide: Docker, Kubernetes, and where Kubeflow fits
```

## Requirements

- Python 3.11+
- pip
- Docker (optional, for the containerized workflow)
- A local Kubernetes cluster such as Minikube or Kind + `kubectl` (optional, for the Kubernetes workflow)

## Running locally

```bash
python3 -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate

pip install -r requirements.txt

# (re)train the model — creates/overwrites house_model.pkl
python train_model.py

# start the API
python app.py
```

The API is now available at `http://127.0.0.1:5000`.

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

## Running with Docker

```bash
docker build -t house-price-api:1.0 .
docker run -p 5000:5000 house-price-api:1.0
```

Then hit it the same way as above (`curl -X POST http://127.0.0.1:5000/predict ...`).

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

Full walkthrough, including scaling and self-healing demos, is in [`docker-kubernetes-kubeflow-guide.md`](./docker-kubernetes-kubeflow-guide.md).

## CI/CD

The `Jenkinsfile` runs on every build: sets up a virtualenv, installs dependencies, retrains the model, starts the API, runs a live smoke test against `/predict`, then tears everything down and archives `house_model.pkl` and the run log as build artifacts.

## Deployment

The app is deployed on [Render](https://render.com), which builds and runs it directly from this repository.

## Learn more

See [`docker-kubernetes-kubeflow-guide.md`](./docker-kubernetes-kubeflow-guide.md) for a hands-on walkthrough of how this project moves from plain Python, through Docker and Kubernetes, and where a tool like Kubeflow would fit in at larger scale.
