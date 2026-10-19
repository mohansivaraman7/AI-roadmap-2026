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

You already have Jenkins. This guide adds Docker and Kubernetes hands-on, using this exact project, and explains where Kubeflow would come in without requiring you to set it up.

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

### Run it

```bash
# 1. Build the image (same as Part 1)
docker build -t house-price-api:1.0 .

# 2. Make the image visible to your local cluster:
#    Minikube:
minikube image load house-price-api:1.0
#    Kind (use instead of the line above, if you're using Kind):
kind load docker-image house-price-api:1.0

# 3. Apply the manifests
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

# 4. Watch the Pods come up
kubectl get pods -w
```

Once both Pods show `Running`, reach the service:

```bash
# Minikube gives you a direct URL for a NodePort service:
minikube service house-price-api-service --url

# then curl the URL it prints, e.g.:
curl -X POST http://127.0.0.1:PORT/predict \
  -H "Content-Type: application/json" \
  -d '{"size": 1700}'
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
```

---

## Part 3 — Where Kubeflow fits (conceptual)

Kubeflow is **not** something to set up for this project — it needs a full Kubernetes cluster with ML pipeline tooling installed on top, and it's built for a scale this project doesn't have. But now that you've run real Kubernetes commands above, it's worth explaining precisely where Kubeflow would enter the picture if this project grew.

Plain Kubernetes (what you just used) knows how to run and scale *containers*, full stop — it has no concept of "this container is a model training job" versus "this one serves predictions." Kubeflow builds ML-specific abstractions on top of Kubernetes:

- **Pipelines** — chaining steps like "load data → train → evaluate → deploy" as a single reproducible workflow, instead of you manually running `train_model.py` and rebuilding an image each time.
- **Distributed/GPU training** — spreading a large training job across multiple machines in the cluster.
- **Model serving (KServe)** — a standardized way to deploy trained models behind an API, conceptually similar to what your `Deployment` + `Service` pair does above, but with ML-specific features like automatic scaling based on inference traffic and side-by-side model version rollouts.
- **Experiment tracking & hyperparameter tuning** — recording which training runs used which settings and results, and automating the search for the best ones.

You'd reach for Kubeflow when `train_model.py` stops being five lines of pandas/scikit-learn and becomes a multi-hour distributed training job, when you have many models to manage instead of one, or when a team needs shared, repeatable ML infrastructure instead of one person running scripts by hand. That's a meaningfully different scale of problem than a linear regression over five data points.

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
   Kubernetes ── runs N copies of that image, restarts failed ones, scales up/down
          │
          ▼
   Kubeflow ── (only at ML-team scale) orchestrates training pipelines and model
                serving *using* Kubernetes underneath, instead of hand-run scripts
```

Render, where this project is actually deployed today, is a managed platform that does a simplified version of the Docker + Kubernetes steps for you behind the scenes — it builds and runs your app in a container and keeps it up, without you writing a Dockerfile or Kubernetes manifest at all. That's exactly why the project worked fine before any of this: Render was already doing that job. What this guide adds is seeing, explicitly and by hand, what each layer underneath a platform like Render is actually responsible for.

## Suggested hands-on exercise for a student

1. Clone the project and run it locally with plain Python — confirm `/predict` works.
2. Write the `Dockerfile` above, build the image, and run it with `docker run` — confirm the same request gives the same answer.
3. Install Minikube (or Kind) + `kubectl`, apply the two manifests, and confirm the API responds through the Kubernetes Service instead of `docker run` directly.
4. Kill a Pod and watch it get replaced; scale to 4 replicas and watch new Pods appear.
5. Write one paragraph, in their own words, on what changed at each step and what problem it solved — that's the real test of whether the concept landed, more than getting the commands to run.
