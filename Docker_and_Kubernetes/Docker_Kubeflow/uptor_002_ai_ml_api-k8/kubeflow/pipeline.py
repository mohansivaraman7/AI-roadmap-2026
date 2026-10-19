"""Kubeflow Pipelines version of train_model.py.

Same LinearRegression training logic as ../train_model.py, but expressed as a
two-step Kubeflow pipeline instead of a script you run by hand:

  1. train_house_price_model  -- trains the model, logs the run to MLflow,
     and writes the model out as a pipeline artifact (not just a local .pkl).
  2. check_quality_gate       -- fails the pipeline run if r2 drops below a
     threshold, the pipeline equivalent of the Jenkinsfile's smoke test.

Compile it with:
    python kubeflow/pipeline.py
which writes house_price_pipeline.yaml -- upload that file in the Kubeflow
Pipelines UI (or submit it with the kfp SDK) to run it on the cluster.
See ../docker-kubernetes-kubeflow-guide.md, Part 4, for the full walkthrough.
"""

from kfp import compiler, dsl
from kfp.dsl import Input, Metrics, Model, Output


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=[
        "pandas==2.2.2",
        "scikit-learn==1.5.1",
        "joblib==1.4.2",
        "mlflow-skinny==2.16.0",
    ],
)
def train_house_price_model(
    mlflow_tracking_uri: str,
    model: Output[Model],
    metrics: Output[Metrics],
):
    """Trains the model and logs the run, mirroring train_model.py exactly."""
    import joblib
    import pandas as pd
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score

    # Same five rows train_model.py uses.
    df = pd.DataFrame({
        "size": [1000, 1200, 1500, 1800, 2000],
        "price": [50, 60, 75, 90, 100],
    })
    x = df[["size"]]
    y = df["price"]

    reg = LinearRegression()
    reg.fit(x, y)

    preds = reg.predict(x)
    mse = mean_squared_error(y, preds)
    r2 = r2_score(y, preds)

    # model.path is a location Kubeflow manages for this run -- writing here
    # (rather than a plain "house_model.pkl") is what lets the next pipeline
    # step, and the Kubeflow UI's lineage graph, find this exact artifact.
    joblib.dump(reg, model.path)

    metrics.log_metric("mse", mse)
    metrics.log_metric("r2", r2)

    # Log to the same MLflow server the Docker Compose / Kubernetes api and
    # train services use -- never let this block the pipeline if MLflow is
    # unreachable, same policy as train_model.py and app.py.
    try:
        import mlflow

        if mlflow_tracking_uri:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment("house-price-training")
        with mlflow.start_run(run_name="kubeflow-pipeline"):
            mlflow.log_param("model_type", "LinearRegression")
            mlflow.log_param("training_rows", len(df))
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("r2", r2)
        print("Logged to MLflow.")
    except Exception as exc:
        print(f"MLflow logging skipped: {exc}")

    print(f"Trained model: r2={r2:.3f}, mse={mse:.3f}")


@dsl.component(base_image="python:3.11-slim")
def check_quality_gate(metrics: Input[Metrics], min_r2: float):
    """Fails the run if r2 is below the threshold -- the pipeline's smoke test."""
    r2 = metrics.metadata["r2"]
    print(f"r2={r2:.4f}, required >= {min_r2}")
    if r2 < min_r2:
        raise ValueError(
            f"Quality gate failed: r2 {r2:.4f} is below the required {min_r2}"
        )
    print("Quality gate passed.")


@dsl.pipeline(
    name="house-price-training-pipeline",
    description=(
        "Trains the house-price LinearRegression model, logs the run to "
        "MLflow, and gates on a minimum r2 -- the Kubeflow equivalent of "
        "running train_model.py by hand."
    ),
)
def house_price_pipeline(
    mlflow_tracking_uri: str = "http://mlflow-service.default.svc.cluster.local:5000",
    min_r2: float = 0.90,
):
    train_task = train_house_price_model(mlflow_tracking_uri=mlflow_tracking_uri)
    train_task.set_display_name("Train model")

    gate_task = check_quality_gate(
        metrics=train_task.outputs["metrics"],
        min_r2=min_r2,
    )
    gate_task.set_display_name("Quality gate")


if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=house_price_pipeline,
        package_path="house_price_pipeline.yaml",
    )
    print("Compiled house_price_pipeline.yaml")
