import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Sample data
df = pd.DataFrame({
    "size":[1000,1200,1500,1800,2000],
    "price":[50,60,75,90,100]
})

x = df[["size"]]
y = df["price"]

model = LinearRegression()
model.fit(x, y)

joblib.dump(model, "house_model.pkl")

print("Model saved successfully")