import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg

# Step 1: Create time series data with datetime index
date_rng = pd.date_range(start='2023-01-01', end='2023-01-30', freq='D')
np.random.seed(140)
data = np.cumsum(np.random.randn(len(date_rng))) + 50   # cumulative to create trend
df = pd.Series(data, index=date_rng)
print(df)

# Step 2: Plot the data
plt.figure(figsize=(8,4))
plt.plot(df, label="Original Data")
plt.title("Daily Time Series Data")
plt.xlabel("Date")
plt.ylabel("Value")
plt.legend()
plt.show()
#
# Step 3: Train-Test Split
train = df[:-5] #0 :
test = df[-5:]
#
# # Step 4: Fit AR model (using 3 lags)
# model = AutoReg(train, lags=3)
# model_fit = model.fit()
#
# # Step 5: Forecast
# predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)
#
# # Step 6: Visualize
# plt.figure(figsize=(8,4))
# plt.plot(train, label="Train Data")
# plt.plot(test, label="Test Data", color='orange')
# plt.plot(predictions, label="Predictions", color='green', marker='o')
# plt.title("AutoRegressive (AR) Model Forecast")
# plt.xlabel("Date")
# plt.ylabel("Value")
# plt.legend()
# plt.show()
#
# print("Predicted values:")
# print(predictions)
