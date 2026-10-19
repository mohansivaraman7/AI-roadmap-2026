# From Code to Cluster: Docker, Kubernetes, and Kubeflow, Explained Through One Real Project

This guide uses one small, real project — a Flask API that predicts house prices — to walk through four tools that often get mentioned in the same breath: **Jenkins, Docker, Kubernetes, and Kubeflow**. Each one solves a different problem, and the best way to understand why each exists is to watch the same project pass through all of them.

## The project we're using

Repository: `uptor_002_ai_ml_api`

```
app.py              # Flask API: loads house_model.pkl, exposes POST /predict
train_model.py       # Trains a scikit-learn LinearRegression model, saves house_model.pkl
house_model.pkl       # The trained model
requirements.txt      # pandas, scikit-learn, flask, joblib, gunicorn
test_prediction.py    # Sends a sample request to /predict and prints the result
Jenkinsfile           # CI pipeline: install deps, retrain, smoke-test the API
```

Today this project already works: it's deployed on Render, and a Jenkinsfile builds and smoke-tests it on every commit. Nothing below replaces that — it adds two more layers underneath it so you can see what Docker and Kubernetes actually contribute.

## The mental model

Before diving in, here's the one-sentence version of each tool, in the order they build on each other:

| Tool | What it actually does | Where it sits |
|---|---|---|
| **Jenkins** | Runs your build/test steps automatically on every commit | CI — "does the code still work?" |
| **Docker** | Packages your app + its dependencies into one portable image | Packaging — "make it run the same everywhere" |
| **Kubernetes** | Runs, scales, and heals many containers across machines | Orchestration — "keep N copies of this running reliably" |
| **Kubeflow** | Adds ML-specific pipelines (train/serve/tune) on top of Kubernetes | ML orchestration — "manage many models and training jobs" |

You already have Jenkins. This guide adds Docker, Kubernetes, and Kubeflow Pipelines hands-on, all using this exact project.

---

## Part 1 — Docker: packaging the app

### Why

Right now, running this app depends on whoever's machine it's running on: do they have Python 3.11 or 3.9? Did they `pip install` the exact same package versions? Docker removes that question by baking the app *and* everything it needs into one image that runs identically anywhere Docker is installed.

### The Dockerfile

Create this file as `Dockerfile` in the project root:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies first so Docker can cache this layer
# whenever only app code changes, not requirements.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy just what the running API needs
COPY app.py house_model.pkl ./

EXPOSE 5000

# gunicorn (already in requirements.txt) instead of Flask's dev server —
# this mirrors how the app is meant to run in production on Render.
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
```

Walking through it line by line:

- `FROM python:3.11-slim` — start from a minimal Linux image that already has Python 3.11 installed. This is the "base layer" everything else stacks on.
- `WORKDIR /app` — every following command runs inside `/app` inside the image.
- `COPY requirements.txt .` then `RUN pip install ...` — installing dependencies *before* copying the rest of the code is a deliberate ordering trick: Docker caches each step, so if you only change `app.py` later, Docker reuses the cached "dependencies installed" layer instead of reinstalling everything.
- `COPY app.py house_model.pkl ./` — copy in only what the running API needs (not `train_model.py` or `test_prediction.py` — those are dev-time files, not part of the running service).
- `EXPOSE 5000` — documents which port the container listens on.
- `CMD [...]` — the command that runs when the container starts: `gunicorn` serving the `app` object from `app.py`, the same production server your `requirements.txt` already includes.

Also add a `.dockerignore` next to it, so build artifacts and local junk never get copied into the image:

```
venv/
__pycache__/
*.pyc
.git
Jenkinsfile
app.log
app.pid
```

### Build and run it

These commands need Docker Desktop (or another Docker daemon) running on **your** machine — a from-scratch container build isn't something that can be verified from inside this chat session, so run these yourself:

```bash
# Build the image (the "1.0" is just a tag/version you choose)
docker build -t house-price-api:1.0 .

# Run it, mapping container port 5000 to your machine's port 5000
docker run -p 5000:5000 house-price-api:1.0
```

Then, in another terminal, hit it exactly the way `test_prediction.py` does:

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"size": 1700}'
```

You should get back something like `{"house_size": 1700, "predicted_price": 85.0}` — the same result the app gives when run directly with Python, just now running from a self-contained image instead of your local Python install.

### What you should notice

The app's behavior didn't change at all. What changed is *portability*: that same `house-price-api:1.0` image will run identically on your laptop, a classmate's laptop, a Jenkins agent, or a cloud server — because everything it needs travels with it.

---

## Part 2 — Kubernetes: running it at scale

### Why

Docker packages one container. It doesn't decide *how many* copies should run, restart one if it crashes, or spread traffic across several. That's Kubernetes' job. It's overkill for a single instance of a toy API — but running through it hands-on, even briefly, is the clearest way to understand what "orchestration" actually means in practice.

### Prerequisites

You need a local Kubernetes cluster and `kubectl` (the Kubernetes command-line tool). Two easy options:

- **Minikube** — runs a single-node cluster in a VM/container on your machine.
- **Kind** ("Kubernetes in Docker") — runs a cluster using Docker containers as nodes.

Either works fine for this exercise; install one plus `kubectl` following their official install docs before continuing.

### The manifests

Kubernetes is configured declaratively — you describe the end state you want in YAML files, and Kubernetes continuously works to match reality to that description. Create a `k8s/` folder with two files.

**`k8s/deployment.yaml`** — describes how many copies of your container should run:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: house-price-api
  labels:
    app: house-price-api
spec:
  replicas: 2
  selector:
    matchLabels:
      app: house-price-api
  template:
    metadata:
      labels:
        app: house-price-api
    spec:
      containers:
        - name: house-price-api
          image: house-price-api:1.0
          imagePullPolicy: IfNotPresent
          ports:
            - containerPort: 5000
          readinessProbe:
            httpGet:
              path: /
              port: 5000
            initialDelaySeconds: 5
            periodSeconds: 10
          livenessProbe:
            httpGet:
              path: /
              port: 5000
            initialDelaySeconds: 10
            periodSeconds: 20
```

Key things to notice:

- `replicas: 2` — Kubernetes will keep **two** copies (Pods) of your container running at all times. Kill one, and it starts a replacement automatically — this is the "self-healing" everyone mentions.
- `image: house-price-api:1.0` — the exact image you built with Docker in Part 1. Kubernetes doesn't build images; it only runs ones that already exist.
- `imagePullPolicy: IfNotPresent` — important for a locally-built image: it tells Kubernetes "don't try to download this from the internet, use the one already loaded."
- `readinessProbe` / `livenessProbe` — Kubernetes periodically checks `GET /` (your Flask API's home route) to know whether a Pod is healthy and ready to receive traffic.

**`k8s/service.yaml`** — gives the set of Pods a stable network address:

```yaml
apiVersion: v1
kind: Service
metadata:
  name: house-price-api-service
spec:
  type: NodePort
  selector:
    app: house-price-api
  ports:
    - port: 5000
      targetPort: 5000
      nodePort: 30500
```

Pods come and go (they get new internal IPs each time), so the Service is what gives your app one stable place to be reached, and automatically load-balances across whichever Pods currently match `app: house-price-api`.

*(These two YAML files were checked for valid syntax and required Kubernetes fields with a YAML parser before being included here.)*

### Adding MLflow to the cluster

The project's `docker-compose.yml` runs three services: `mlflow`, `train`, and `api`. So far the manifests above only cover `api` on its own. Three more files bring MLflow onto the cluster too, mirroring what Compose already does:

**`k8s/mlflow-pvc.yaml`** — a PersistentVolumeClaim, so the tracking database and logged models survive a Pod restart instead of vanishing (the Kubernetes equivalent of Compose's `mlflow_data` named volume):

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: mlflow-data
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 1Gi
```

**`k8s/mlflow-deployment.yaml`** — runs the same image `Dockerfile.mlflow` builds, mounted to that PVC:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mlflow
  labels:
    app: mlflow
spec:
  replicas: 1   # see note below
  selector:
    matchLabels:
      app: mlflow
  template:
    metadata:
      labels:
        app: mlflow
    spec:
      containers:
        - name: mlflow
          image: mlflow-house-price:1.0
          imagePullPolicy: IfNotPresent
          ports:
            - containerPort: 5000
          volumeMounts:
            - name: mlflow-data
              mountPath: /mlflow
          readinessProbe:
            httpGet: { path: /, port: 5000 }
            initialDelaySeconds: 5
            periodSeconds: 10
          livenessProbe:
            httpGet: { path: /, port: 5000 }
            initialDelaySeconds: 10
            periodSeconds: 20
      volumes:
        - name: mlflow-data
          persistentVolumeClaim:
            claimName: mlflow-data
```

Why `replicas: 1` here but `replicas: 2` for the api: this server's backend store is a SQLite file, and SQLite doesn't handle two Pods writing to it at once safely. The api is stateless, so scaling it is just "run more copies." MLflow isn't, unless you swap the backend for something like Postgres — worth noticing as a real, common distinction between "can I just scale this" and "not without changing the storage first."

**`k8s/mlflow-service.yaml`** — a ClusterIP Service (the default type, so no `type:` field), meaning it's reachable from other Pods but *not* exposed outside the cluster the way the api's NodePort Service is:

```yaml
apiVersion: v1
kind: Service
metadata:
  name: mlflow-service
spec:
  selector:
    app: mlflow
  ports:
    - port: 5000
      targetPort: 5000
```

That's a deliberate difference from Compose, where both `mlflow` (port 5001) and `api` (port 5000) are published to your host. On Kubernetes you choose what's reachable from outside per Service — MLflow has no reason to be, so it stays internal-only; to open its UI from your own machine anyway, use `kubectl port-forward` (shown below) instead of a NodePort.

Finally, `k8s/deployment.yaml` (the api Deployment) gets one addition, an `env` block on the container so `api` knows where to find MLflow — the same role `MLFLOW_TRACKING_URI=http://mlflow:5000` plays in `docker-compose.yml`, just resolved by cluster DNS via the Service name instead of Compose's network:

```yaml
          env:
            - name: MLFLOW_TRACKING_URI
              value: http://mlflow-service:5000
```

*(All five manifests were checked for valid syntax and required Kubernetes fields with a YAML parser before being included here.)*

### Run it

```bash
# 1. Build both images (api, and mlflow from its own Dockerfile)
docker build -t house-price-api:1.0 .
docker build -f Dockerfile.mlflow -t mlflow-house-price:1.0 .

# 2. Make both images visible to your local cluster:
#    Minikube:
minikube image load house-price-api:1.0
minikube image load mlflow-house-price:1.0
#    Kind (use instead of the two lines above, if you're using Kind):
kind load docker-image house-price-api:1.0
kind load docker-image mlflow-house-price:1.0

# 3. Apply MLflow's manifests first, then the api's — mirrors Compose's
#    depends_on: mlflow, though api will keep starting fine even if MLflow
#    isn't ready yet, exactly like it does locally (see app.py).
kubectl apply -f k8s/mlflow-pvc.yaml
kubectl apply -f k8s/mlflow-deployment.yaml
kubectl apply -f k8s/mlflow-service.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

# 4. Watch the Pods come up (should settle at 1 mlflow + 2 api = 3 Pods)
kubectl get pods -w
```

Once all Pods show `Running`, reach the api the same way as before:

```bash
# Minikube gives you a direct URL for a NodePort service:
minikube service house-price-api-service --url

# then curl the URL it prints, e.g.:
curl -X POST http://127.0.0.1:PORT/predict \
  -H "Content-Type: application/json" \
  -d '{"size": 1700}'
```

Then check MLflow logged that request — this is the real test that the two Pods are actually talking to each other through the Service, not just that both happen to be `Running`:

```bash
kubectl port-forward svc/mlflow-service 5001:5000
# leave that running, then in your browser:
#   http://localhost:5001
# open the "house-price-predictions" experiment — you should see one run
# per /predict call you made above.
```

### Two things worth demonstrating to a class

**Self-healing** — delete one Pod and watch Kubernetes replace it automatically:

```bash
kubectl get pods
kubectl delete pod <one-of-the-pod-names>
kubectl get pods   # a new one appears within seconds
```

**Scaling** — go from 2 copies to 4 with one command, no new Docker build needed:

```bash
kubectl scale deployment house-price-api --replicas=4
kubectl get pods
```

This is the concrete payoff of Kubernetes: the same declarative file that describes "2 copies" can describe "4 copies," and Kubernetes reconciles reality to match — something you'd otherwise be doing by hand with `docker run` commands one at a time.

### Clean up

```bash
kubectl delete -f k8s/service.yaml
kubectl delete -f k8s/deployment.yaml
kubectl delete -f k8s/mlflow-service.yaml
kubectl delete -f k8s/mlflow-deployment.yaml
kubectl delete -f k8s/mlflow-pvc.yaml
```

---

## Part 3 — What Kubeflow adds on top of plain Kubernetes

Plain Kubernetes (what you just used) knows how to run and scale *containers*, full stop — it has no concept of "this container is a model training job" versus "this one serves predictions." You had to know that yourself, and wire `train_model.py`'s logic and MLflow logging into it by hand. Kubeflow builds ML-specific abstractions on top of Kubernetes so that stops being hand-wiring:

- **Pipelines** — chaining steps like "load data → train → evaluate → deploy" as one reproducible, versioned workflow with a visual graph, instead of you manually running a script and rebuilding an image each time. This is the piece Part 4 below sets up and actually runs.
- **Distributed/GPU training** — spreading a large training job across multiple machines in the cluster.
- **Model serving (KServe)** — a standardized way to deploy trained models behind an API, conceptually similar to what your `Deployment` + `Service` pair does for `api` above, but with ML-specific features like automatic scaling based on inference traffic and side-by-side model version rollouts.
- **Experiment tracking & hyperparameter tuning** — recording which training runs used which settings and results, and automating the search for the best ones. (Your project already does a *simple* version of the tracking half with plain MLflow — Kubeflow's contribution is orchestrating the runs that produce those records, not replacing MLflow.)

A full Kubeflow platform install is genuinely heavy — it's built for teams running many models and pipelines, not a five-row linear regression — so Part 4 below uses only **Kubeflow Pipelines standalone**, the one component that's realistic to run on a laptop-sized Minikube cluster and the one most syllabi mean when they say "Kubeflow."

## Part 4 — Kubeflow Pipelines: hands-on

### Why this component specifically

"Kubeflow" is really a bundle of separate components (Pipelines, Katib for tuning, KServe for serving, notebooks, and more), installed together or piece by piece. **Kubeflow Pipelines (KFP)** is the one that answers the actual gap Part 3 named: right now, "train the model" means someone runs `train_model.py` by hand, or Jenkins runs it as one opaque shell step. KFP turns that into a real pipeline: separate, versioned steps, each in its own container, with the trained model passed between them as a tracked artifact and a graph you can look at afterward — rather than one script's stdout.

### 1. Give Minikube more room

KFP standalone brings its own API server, persistence agent, MySQL, MinIO, and UI — several Pods beyond the 3 you're already running for `api` and `mlflow`. Start (or restart) Minikube with more resources than Part 2 needed:

```bash
minikube delete           # only if you already have a smaller cluster running
minikube start --driver=docker --cpus=4 --memory=8192
```

### 2. Install Kubeflow Pipelines standalone

This deploys KFP into its own `kubeflow` namespace, separate from the `default` namespace `api` and `mlflow` are running in (that separation is exactly why `kubeflow/pipeline.py` below addresses MLflow by its *full* cluster DNS name, not just `mlflow-service`):

```bash
export PIPELINE_VERSION=2.17.0

kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=$PIPELINE_VERSION"
kubectl wait --for condition=established --timeout=60s crd/applications.app.k8s.io
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/dev?ref=$PIPELINE_VERSION"

# takes a few minutes — wait until everything in the kubeflow namespace is Running
kubectl get pods -n kubeflow -w
```

*(Commands current as of the official Kubeflow Pipelines installation docs at the time this guide was written. If `kubectl apply -k` fails, check `kubeflow.org/docs/components/pipelines/operator-guides/installation` for whatever the current command is — these manifests do move between releases.)*

Once the Pods settle, open the UI:

```bash
kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80
# then browse to http://localhost:8080
```

### 3. Install the SDK and compile the pipeline

On your own machine (not inside the cluster):

```bash
pip install kfp
```

`kubeflow/pipeline.py` is already in the project. It expresses the same training logic as `train_model.py` as two pipeline steps instead of one script:

```python
@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["pandas==2.2.2", "scikit-learn==1.5.1", "joblib==1.4.2", "mlflow-skinny==2.16.0"],
)
def train_house_price_model(mlflow_tracking_uri: str, model: Output[Model], metrics: Output[Metrics]):
    ...  # same 5-row DataFrame, same LinearRegression fit as train_model.py
    joblib.dump(reg, model.path)         # written as a pipeline artifact, not a loose .pkl
    metrics.log_metric("r2", r2)
    ...                                   # also logs to MLflow, same as train_model.py

@dsl.component(base_image="python:3.11-slim")
def check_quality_gate(metrics: Input[Metrics], min_r2: float):
    if metrics.metadata["r2"] < min_r2:
        raise ValueError(...)             # fails the run — the pipeline's version of a smoke test

@dsl.pipeline(name="house-price-training-pipeline")
def house_price_pipeline(
    mlflow_tracking_uri: str = "http://mlflow-service.default.svc.cluster.local:5000",
    min_r2: float = 0.90,
):
    train_task = train_house_price_model(mlflow_tracking_uri=mlflow_tracking_uri)
    check_quality_gate(metrics=train_task.outputs["metrics"], min_r2=min_r2)
```

A few things worth noticing in that code: `packages_to_install` means each step's container is built for you from a plain `python:3.11-slim` base at run time — you don't need to write and push a custom Dockerfile the way Parts 1 and 2 required (a real production pipeline usually would, for faster and more reproducible runs, but this keeps the exercise focused). `model.path` and `metrics` are how KFP tracks artifacts between steps and shows them in the UI's run graph — that lineage is the thing a bare shell script or Jenkins stage doesn't give you for free.

Compile it:

```bash
cd kubeflow
python pipeline.py
```

*(This was run in this session's sandbox against kfp 2.17.0 to confirm it compiles cleanly — it produces `house_price_pipeline.yaml`, a couple hundred lines of pipeline spec, without needing a cluster to do that part.)*

### 4. Run it

In the KFP UI (`http://localhost:8080`): **Pipelines → Upload pipeline**, select `kubeflow/house_price_pipeline.yaml`, then **Create run**. Watch the two-node graph execute — `Train model` then `Quality gate` — and click either node's **Artifacts** tab to see the logged metrics and the model file itself.

Then check MLflow the same way as in Part 2:

```bash
kubectl port-forward svc/mlflow-service 5001:5000
# http://localhost:5001 → "house-price-training" experiment →
# a run named "kubeflow-pipeline", logged by the pipeline step, not by you running a script
```

### What actually changed versus Jenkins running the same training step

Your `Jenkinsfile` already retrains the model on every build — so what did this buy you? Jenkins runs `train_model.py` as one opaque shell command on one agent; if it fails, you get a log to scroll through. Kubeflow Pipelines runs the same logic as two separate, independently retryable containers, keeps the model and metrics as addressable artifacts instead of print statements, blocks automatically on the quality gate before anything downstream (like a deploy step) would run, and gives you a visual history of every past run to compare against — the difference between "a script ran" and "a versioned, inspectable ML workflow ran."

### Clean up

Kubeflow Pipelines standalone is the heaviest thing in this whole guide — don't leave it running on a laptop:

```bash
kubectl delete -k "github.com/kubeflow/pipelines/manifests/kustomize/env/dev?ref=$PIPELINE_VERSION"
kubectl delete -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=$PIPELINE_VERSION"
```

---

## Putting the whole picture together

```
   Your code (app.py, train_model.py, ...)
          │
          ▼
   Jenkins  ── installs deps, retrains the model, runs a smoke test on every commit
          │
          ▼
   Docker  ── packages the app + dependencies into one portable image
          │
          ▼
   Kubernetes ── runs N copies of api + mlflow, restarts failed Pods, scales up/down
          │
          ▼
   Kubeflow Pipelines ── orchestrates the train → gate steps as a tracked workflow,
                          running *on* the same Kubernetes cluster, in its own
                          kubeflow namespace, instead of one hand-run script
```

Render, where this project is actually deployed today, is a managed platform that does a simplified version of the Docker + Kubernetes steps for you behind the scenes — it builds and runs your app in a container and keeps it up, without you writing a Dockerfile or Kubernetes manifest at all. That's exactly why the project worked fine before any of this: Render was already doing that job. What this guide adds is seeing, explicitly and by hand, what each layer underneath a platform like Render is actually responsible for.

## Suggested hands-on exercise for a student

1. Clone the project and run it locally with plain Python — confirm `/predict` works.
2. Write the `Dockerfile` above, build the image, and run it with `docker run` — confirm the same request gives the same answer.
3. Install Minikube (or Kind) + `kubectl`, apply the api and mlflow manifests, and confirm the API responds through the Kubernetes Service instead of `docker run` directly.
4. Kill the mlflow Pod and watch it get replaced; scale the api Deployment to 4 replicas and watch new Pods appear.
5. Install Kubeflow Pipelines standalone, compile and run `kubeflow/pipeline.py`, and confirm a run shows up in both the KFP UI and MLflow's `house-price-training` experiment.
6. Write one paragraph, in their own words, on what changed at each step and what problem it solved — that's the real test of whether the concept landed, more than getting the commands to run.
