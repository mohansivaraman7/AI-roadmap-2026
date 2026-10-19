import os

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import mlflow
import mlflow.sklearn

# Point at the MLflow tracking server. When running inside docker-compose,
# MLFLOW_TRACKING_URI is set to the "mlflow" service; when running this
# locally against `docker compose up mlflow`, it defaults to the port
# that service is published on (127.0.0.1:5001).
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001"))
mlflow.set_experiment("house-price-training")

# Sample data
df = pd.DataFrame({
    "size": [1000, 1200, 1500, 1800, 2000],
    "price": [50, 60, 75, 90, 100]
})

x = df[["size"]]
y = df["price"]

with mlflow.start_run():
    model = LinearRegression()
    model.fit(x, y)

    preds = model.predict(x)
    mse = mean_squared_error(y, preds)
    r2 = r2_score(y, preds)

    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_param("training_rows", len(df))
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2", r2)
    mlflow.sklearn.log_model(model, name="model")

    joblib.dump(model, "house_model_latest.pkl")

    run_id = mlflow.active_run().info.run_id
    print(f"Model saved successfully (MLflow run: {run_id}, r2={r2:.3f}, mse={mse:.3f})")
