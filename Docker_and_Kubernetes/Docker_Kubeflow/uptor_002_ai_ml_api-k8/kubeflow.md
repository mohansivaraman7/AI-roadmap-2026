# Docker → Kubernetes → Kubeflow: 2-Hour Class Prep

Your own working notes for teaching this using `uptor_002_ai_ml_api` (the house-price API + MLflow project). Built from a session where I walked through this exact material step by step and hit the same confusions a student will — the "questions students will likely ask" sections below are pulled directly from real sticking points, not guesses.

---

## 1. The one-sentence mental model (say this out loud early, repeat it at each transition)

| Tool | What it actually does | Where it sits |
|---|---|---|
| **Docker** | Packages your app + its dependencies into one portable image | Packaging — "make it run the same everywhere" |
| **Docker Compose** | Runs several containers together on one machine, wired up by one config file | Local multi-container convenience — a wrapper around `docker build`/`docker run` |
| **Kubernetes** | Runs, restarts, and scales containers — possibly across many machines | Orchestration — "keep N copies of this running reliably" |
| **Kubeflow Pipelines** | Runs a training workflow as tracked, versioned steps — on top of Kubernetes | ML workflow orchestration — "a job that happens and is recorded," not a service that stays up |

The through-line for the whole 2 hours: **each tool solves a problem the previous one doesn't.** Docker solves "it works on my machine but not yours." Kubernetes solves "it's running, but if it crashes at 2am nobody's there to restart it." Kubeflow solves "training happened, but there's no record of what ran, with what data, and whether it passed quality checks." If a student asks "why do I need the next tool," that's the honest answer every time — not "because it's more powerful."

---

## 2. Before class — do this the night before, not live

Two hours disappears fast if any of this happens live. Pre-build and pre-verify so class time goes to *explaining*, not *waiting*:

- [ ] `docker build` both images (`house-price-api:1.0`, `mlflow-house-price:1.0`) ahead of time — a cold build pulling `python:3.11-slim` and installing pandas/scikit-learn can take a couple of minutes on conference wifi.
- [ ] `minikube start --driver=docker` already run once on the training machine, confirmed working, *before* students are watching. First-start issues (Docker Desktop not running, WSL2 not enabled on Windows) eat 10+ minutes if hit live.
- [ ] Decide now: **are you actually installing Kubeflow Pipelines live, or only showing it?** KFP standalone needs `--cpus=4 --memory=8192` and ~3 minutes to deploy even when nothing goes wrong. In a 2-hour class, recommend pre-installing it before class and just walking through the already-running UI + a fresh pipeline run live — don't burn 15+ minutes of a 2-hour class on a from-scratch KFP install unless this is the one thing you want them to see happen.
- [ ] Have `kubectl get pods -A` and `docker images` already run once so you know current state before you start narrating.
- [ ] Know your **image name ↔ manifest name** pairing cold. This is the single most common live-demo failure — see the gotcha below.

---

## 3. Suggested 2-hour timeline

| Time | Segment |
|---|---|
| 0:00–0:10 | Mental model + why this project (the one table above, plus "we're doing this hands-on on one real small app, not slides") |
| 0:10–0:35 | **Docker**: image vs container, build vs run, walk the Dockerfile |
| 0:35–1:00 | **Docker Compose**: why 3 containers need Compose, walk docker-compose.yml, live `docker compose up` |
| 1:00–1:10 | *Break* |
| 1:10–1:45 | **Kubernetes**: Deployment vs Service, apply the manifests live, **live self-healing + scaling demo** (this is the moment that actually sells the whole segment — don't cut it) |
| 1:45–2:00 | **Kubeflow**: conceptual framing + walk a pre-installed pipeline run in the UI, tie back to "what MLflow already does vs. what this adds" |
| last 5 min | Quick check questions (Section 6) |

If you're tight on time, the segment to compress is Kubeflow, not Kubernetes — the self-healing/scaling demo is the payoff moment of the whole class and is worth protecting.

---

## 4. Section-by-section teaching notes

### Docker (0:10–0:35)

**Core idea to land**: an image is an inert blueprint; a container is a running instance of it. One image → many containers.

**Live demo**:
```bash
docker build -t house-price-api:1.0 .
docker images                                    # show it now exists, not running
docker run -p 5000:5000 house-price-api:1.0
docker ps                                        # now show it running, in a second terminal
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{"size": 1700}'
```

**Questions students will likely ask, and the answer that lands:**

- *"We never had a file called `house-price-api` — where does that come from, and what's `1.0`?"* — `-t house-price-api:1.0` isn't referencing an existing file at all; it's you *naming* the image you're about to create, the way "Save As" names a new file. `1.0` is a version tag you're choosing, not something Docker enforces meaning on. Show `docker images` right after the build so they see the name land there, freshly created.
- *"Is `house-price-api:1.0` the image or the container?"* — the image. Every `docker run` of it creates a new container with its own ID; the image itself never runs, it's what gets copied to create something that runs. `docker images` vs `docker ps` is the clean way to show this side by side live.

### Docker Compose (0:35–1:00)

**Core idea to land**: Compose is not a different technology from `docker build`/`docker run` — it's the same primitives, driven by a YAML file instead of typed flags, so multiple containers can be defined and networked together in one command instead of typed out by hand per container.

**Live demo**:
```bash
docker compose build
docker compose up -d mlflow
docker compose run --rm train
docker compose up -d api
curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d '{"size": 2100}'
# open http://localhost:5001 — show the MLflow UI, the logged run
docker compose down -v
```

**Questions students will likely ask:**

- *"What's actually different between the `docker build`/`docker run` we just did and this?"* — walk `docker-compose.yml` and point at where each Compose command's flags are already written down: `ports:`, `environment:`, `depends_on:`. Compose reads that file so you don't retype `-p 5001:5000 -e MLFLOW_TRACKING_URI=...` by hand, and it creates one shared network automatically so `api` can reach `mlflow` by container name.
- *"Why `docker compose run --rm train` instead of `up`?"* — `up` is for things that should stay running (`api`, `mlflow`); `run --rm` is for a one-off job that finishes and exits (training the model once), then deletes itself.

### Kubernetes (1:10–1:45) — the core of the class

**Core idea to land**: Kubernetes doesn't change what the app does — a working `curl` looks *identical* to the Compose version. What it adds only shows up when something breaks or when you need more capacity. Say this explicitly, before a student says it skeptically (one did, verbatim: *"looks like docker only, no difference... if it's kubernetes we need to know the use of it"*) — naming the objection yourself lands better than a student discovering it and feeling like the class oversold it.

**Deployment vs Service, in one sentence each**: a Deployment keeps N copies of a container *running*; a Service gives those copies one *stable address*, because individual Pods get replaced and get new IPs each time.

**Live demo — build, load, apply**:
```bash
docker build -t house-price-api:1.0 .
docker build -f Dockerfile.mlflow -t mlflow-house-price:1.0 .
minikube image load house-price-api:1.0
minikube image load mlflow-house-price:1.0

kubectl apply -f k8s/mlflow-pvc.yaml
kubectl apply -f k8s/mlflow-deployment.yaml
kubectl apply -f k8s/mlflow-service.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

kubectl get pods -w
minikube service house-price-api-service --url
```

**The payoff demo — do not skip this part:**
```bash
kubectl get pods
kubectl delete pod <one-of-the-api-pod-names>
kubectl get pods              # a replacement is already there — this is the moment to pause and let it land

kubectl scale deployment house-price-api --replicas=4
kubectl get pods              # 4 pods, one command, no new docker build
```

**Questions students will likely ask:**

- *"We never had `house-price-api` as a real file — same confusion as Docker, now with the manifest too."* — reinforce: `image:` in the YAML has to match, character for character, whatever you tagged with `docker build -t`. This is **the single most common live-fail point**: a student renames their build tag (e.g. following a personal naming convention) and forgets the manifest still says the old name, and gets `ErrImageNeverPull`. Flag this before it happens, not after.
- *"So build → load → apply is the actual order?"* — yes, and it's worth being precise on stage: build creates images (nothing running), load makes the cluster aware of them (still nothing running), `kubectl apply` on the *Deployment* files is what actually starts containers; applying the *Service* files doesn't start anything, it just makes what's already running reachable.
- *"Why two Service files (`service.yaml`, `mlflow-service.yaml`)?"* — a Service can only front one set of Pods (matched by one label selector); two separate running apps need two separate stable addresses, same reason Compose needed two separate service blocks.
- *"Does this mean the code is *always* running now?"* — that's the goal, but be precise: it's a reconciliation loop (Kubernetes keeps fixing the gap between "what should exist" and "what does exist"), not an absolute guarantee. A genuinely broken app will `CrashLoopBackOff` forever, not "become fixed." And on a single-node Minikube cluster specifically, there's no real redundancy — if the one node goes down, everything on it goes down together; multi-node failover is the part Minikube can't actually demonstrate.
- *"What's this PVC file for — do we need it?"* — without it, MLflow's tracking database lives on the Pod's own disposable filesystem, and gets wiped the instant that Pod is replaced (including by the self-healing demo you just ran). The PVC gives it storage that survives a Pod being replaced. Good moment to connect it back to the self-healing demo they just watched: "if we'd deleted the mlflow pod instead of the api pod, without this file every run in that database would already be gone."
- *"Can Kubernetes run on other clusters, like real cloud ones?"* — yes, and it's worth being concrete about the two mechanisms rather than hand-waving: `kubectl` contexts (which cluster your commands currently target — `kubectl config use-context ...`) and image registries (a real remote cluster can't `minikube image load` from your laptop; images have to be `docker push`ed somewhere reachable, like Docker Hub). The YAML files themselves don't change between local and cloud — what changes is how the image gets there and which context is active.

### Kubeflow (1:45–2:00)

**Core idea to land**: `kubectl` is a general-purpose remote control — it has no concept of "this is a training job," it just runs whatever container you point it at, forever. Kubeflow Pipelines is a specific ML application that happens to also run on Kubernetes (installed via `kubectl` itself) — it's job-shaped, not service-shaped: steps run once, finish, and get recorded, rather than staying up.

**Walk, don't rebuild live** (assuming you pre-installed per Section 2): open the KFP UI, upload/re-run `kubeflow/house_price_pipeline.yaml`, click into a completed run, show the two-node graph (`Train model` → `Quality gate`), click the artifacts tab to show the logged model + metrics.

**Questions students will likely ask:**

- *"This still just trains the same 5-row model we had in `train_model.py` — what's actually different?"* — nothing about the model. What's different is *how the training run is represented*: two separate, independently retryable containers instead of one opaque script; the model and metrics as addressable, browsable artifacts instead of print statements; a quality gate that automatically blocks progress if `r2` drops below threshold, before you'd otherwise find out by accident; and a visual history of every past run to compare against.
- *"We already have Jenkins retraining on every commit — why do we need this too?"* — Jenkins runs the retrain as one opaque shell step; if it fails, you get a log to scroll through. This is the same underlying training logic, but broken into inspectable, artifact-tracked steps with a gate — the difference between "a script ran" and "a versioned, inspectable ML workflow ran." Good place to be honest: for a 5-row toy dataset, this is overkill in practice — the value only becomes real once training is genuinely expensive, or there are many models to track instead of one.
- *"That compiled YAML file looks terrifying — do we write that?"* — no, and say this clearly: it's a build output, generated by `python kubeflow/pipeline.py`, the same relationship `house_model.pkl` has to `train_model.py`. You edit the Python, you never hand-edit the compiled YAML.

---

## 5. The golden-path command sheet (keep this open in a terminal pane during class)

```bash
# --- Docker ---
docker build -t house-price-api:1.0 .
docker run -p 5000:5000 house-price-api:1.0

# --- Compose ---
docker compose build
docker compose up -d mlflow
docker compose run --rm train
docker compose up -d api
docker compose down -v

# --- Kubernetes ---
minikube start --driver=docker
docker build -t house-price-api:1.0 .
docker build -f Dockerfile.mlflow -t mlflow-house-price:1.0 .
minikube image load house-price-api:1.0
minikube image load mlflow-house-price:1.0
kubectl apply -f k8s/mlflow-pvc.yaml
kubectl apply -f k8s/mlflow-deployment.yaml
kubectl apply -f k8s/mlflow-service.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl get pods -w
minikube service house-price-api-service --url

# self-healing + scaling payoff demo
kubectl delete pod <pod-name>
kubectl get pods
kubectl scale deployment house-price-api --replicas=4
kubectl get pods

# --- Kubeflow (pre-installed before class) ---
kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80
# browse http://localhost:8080, walk an existing run
```

---

## 6. Five-minute check at the end — ask these out loud, don't just say "any questions?"

1. What's the difference between an image and a container?
2. Why does `docker-compose.yml` need a separate service block for `mlflow` and `api`, instead of one block for both?
3. If you delete a running Pod by hand, what happens next, and why?
4. What's the actual job of a Kubernetes *Service*, in one sentence — and why can't the Deployment do that job itself?
5. Name one thing Kubeflow Pipelines gives you that plain Kubernetes doesn't.

If most of the room can answer #3 and #4 without hesitating, the Kubernetes segment landed — those two are the actual concepts the whole class is built to teach; everything else is scaffolding to get there.
