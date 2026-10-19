import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

"""ARIMA - Auto Regressive, Integrated, Moving Average"""

# Example data
data = [120, 132, 101, 134, 90, 120, 110, 125, 130, 140]
df = pd.Series(data)

# Fit ARIMA model (p=1, d=1, q=1)
model = ARIMA(df, order=(1, 1, 1))
model_fit = model.fit()

# Summary of the model
print(model_fit.summary())

# Forecast next 3 values
forecast = model_fit.forecast(steps=3)
print("Forecasted values:", forecast)

# Plot
plt.plot(df, label="Original")
plt.plot(range(len(df), len(df)+3), forecast, label="Forecast", marker='o')
plt.legend()
plt.show()
