# ann_csv.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Load CSV dataset
df = pd.read_csv("../Architecture_Building_Blocks/customer_churn.csv")  # Example: predict customer churn
# Dataset format: columns like age, tenure, monthly_charges, total_charges, churn (0/1)

df.drop(columns=["CustomerID","Gender","Subscription Type","Contract Length"],inplace=True)
# 2. Separate features (X) and target (y)
X = df.drop("Churn", axis=1).values
y = df["Churn"].values

# 3. Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Scale features (important for ANN)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. Build ANN model
clf = MLPClassifier(
    hidden_layer_sizes=(128, 64, 32),  # Two hidden layers
    activation="relu",
    max_iter=1000,
    random_state=42
)

# 6. Train
clf.fit(X_train, y_train)

# 7. Evaluate
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification report:\n", classification_report(y_test, y_pred))
