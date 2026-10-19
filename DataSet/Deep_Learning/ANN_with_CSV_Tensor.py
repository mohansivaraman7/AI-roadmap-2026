# ann_csv_keras.py — TRUE DEEP LEARNING with Keras
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

# 1. Load CSV
df = pd.read_csv("customer_churn.csv")
X = df.drop("churn", axis=1).values
y = df["churn"].values

# 2. Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 3. Build the model (this is the DEEP LEARNING style)
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')  # Binary classification
])

# 4. COMPILE the model (this is what you mentioned!)
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',  # or 'mse' for some tasks [web:62]
    metrics=['accuracy']
)

# 5. Train with epochs and batch_size
model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# 6. Predict (this is model.predict() you mentioned!)
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)

print("Accuracy:", np.mean(y_pred == y_test))