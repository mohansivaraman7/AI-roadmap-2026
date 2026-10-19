import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Step 1: Create non-stationary time series (with upward trend)
date_rng = pd.date_range(start='2023-01-01', end='2023-01-30', freq='D')
np.random.seed(140)
trend = np.linspace(50, 80, len(date_rng))  # linear upward trend
noise = np.random.randn(len(date_rng)) * 2
data = trend + noise
df = pd.Series(data, index=date_rng)

# Plot original (non-stationary) data
plt.figure(figsize=(8,4))
plt.plot(df, label="Original Data (Non-stationary)", color='orange')
plt.title("Original Series with Trend")
plt.xlabel("Date")
plt.ylabel("Value")
plt.legend()
plt.show()

# Step 2: Apply 1st order differencing (d=1)
diff_df = df.diff().dropna()

plt.figure(figsize=(8,4))
plt.plot(diff_df, label="After 1st Differencing (Stationary)", color='green')
plt.title("Differenced Series (d=1)")
plt.xlabel("Date")
plt.ylabel("Differenced Value")
plt.legend()
plt.show()

# Step 3: Fit ARIMA model with integration
# Here p=0, d=1, q=0 — means only Integrated part
model = ARIMA(df, order=(0, 1, 0))
model_fit = model.fit()

# Step 4: Forecast next 5 days
forecast = model_fit.forecast(steps=5)
print("Forecasted Values:")
print(forecast)

# Step 5: Visualize forecast
plt.figure(figsize=(8,4))
plt.plot(df, label="Original Data")
plt.plot(pd.date_range(df.index[-1], periods=6, freq='D')[1:], forecast, label="Forecast (Integrated)", marker='o')
plt.title("ARIMA(0,1,0) - Integrated Model Forecast")
plt.xlabel("Date")
plt.ylabel("Value")
plt.legend()
plt.show()
