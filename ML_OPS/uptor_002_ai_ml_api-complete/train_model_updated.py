import os

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Sample data
df = pd.DataFrame({
    "size": [1000, 1200, 1500, 1800, 2000],
    "price": [50, 60, 75, 90, 100]
})

x = df[["size"]]
y = df["price"]

model = LinearRegression()
model.fit(x, y)

preds = model.predict(x)
mse = mean_squared_error(y, preds)
r2 = r2_score(y, preds)

# Save the model first — this always succeeds, whether or not MLflow
# is reachable. Regenerating house_model.pkl should never depend on an
# MLflow server being up.
joblib.dump(model, "house_model.pkl")
print(f"Model saved successfully (r2={r2:.3f}, mse={mse:.3f})")

# Log the run to MLflow if a tracking server is reachable — a bonus for
# experiment tracking, not a requirement. If MLFLOW_TRACKING_URI isn't
# set, or no server is listening, we just skip logging instead of
# crashing (mirrors how /predict in app.py handles MLflow being down).
try:
    import mlflow
    import mlflow.sklearn

    mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment("house-price-training")

    with mlflow.start_run():
        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_param("training_rows", len(df))
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2", r2)
        mlflow.sklearn.log_model(model, name="model")
        run_id = mlflow.active_run().info.run_id

    print(f"Logged to MLflow (run: {run_id})")
except Exception as exc:
    print(f"MLflow logging skipped: {exc}")
