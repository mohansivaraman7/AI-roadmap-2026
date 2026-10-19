import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Step 1: Create time series data
date_rng = pd.date_range(start='2023-01-01', end='2023-01-30', freq='D')
np.random.seed(140)
data = np.cumsum(np.random.randn(len(date_rng))) + 50
df = pd.Series(data, index=date_rng)

# Step 2: Split into train and test
train = df[:-5]
test = df[-5:]

# Step 3: Fit Moving Average model (MA with q=2)
model = ARIMA(train, order=(0, 0, 2))
model_fit = model.fit()

# Step 4: Forecast
predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)

# Step 5: Plot
plt.figure(figsize=(8,4))
plt.plot(train, label="Train Data")
plt.plot(test, label="Test Data", color='orange')
plt.plot(predictions, label="MA(2) Predictions", color='green', marker='o')
plt.title("Moving Average (MA) Model Forecast")
plt.xlabel("Date")
plt.ylabel("Value")
plt.legend()
plt.show()

print("Predicted values:")
print(predictions)
