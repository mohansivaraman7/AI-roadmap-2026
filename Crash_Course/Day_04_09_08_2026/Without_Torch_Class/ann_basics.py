# ==========================================
# 1. IMPORT LIBRARIES
# ==========================================

import pandas as pd
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ==========================================
# 2. LOAD DATA
# ==========================================

df = pd.read_csv("Social_Network_Ads.csv")

X = df[['Age', 'EstimatedSalary']]
y = df['Purchased']


# ==========================================
# 3. TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ==========================================
# 4. SCALING
# ==========================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ==========================================
# 5. CONVERT TO TENSORS
# ==========================================

X_train = torch.tensor(
    X_train,
    dtype=torch.float32
)

X_test = torch.tensor(
    X_test,
    dtype=torch.float32
)

y_train = torch.tensor(
    y_train.values.reshape(-1,1),
    dtype=torch.float32
)

y_test = torch.tensor(
    y_test.values.reshape(-1,1),
    dtype=torch.float32
)


# ==========================================
# 6. DEFINE ANN
# ==========================================

model = nn.Sequential(

    nn.Linear(2,8),
    nn.ReLU(),

    nn.Linear(8,4),
    nn.ReLU(),

    nn.Linear(4,1),
    nn.Sigmoid()
)

print(model)


# ==========================================
# 7. LOSS FUNCTION
# ==========================================

criterion = nn.BCELoss()


# ==========================================
# 8. OPTIMIZER
# ==========================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# ==========================================
# 9. TRAINING LOOP
# ==========================================

epochs = 100

for epoch in range(epochs):

    # Forward Propagation
    outputs = model(X_train)

    # Loss Calculation
    loss = criterion(
        outputs,
        y_train
    )

    # Clear Old Gradients
    optimizer.zero_grad()

    # Backpropagation
    loss.backward()

    # Weight Update
    optimizer.step()

    if (epoch + 1) % 10 == 0:

        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Loss = {loss.item():.4f}"
        )


# ==========================================
# 10. TESTING
# ==========================================

with torch.no_grad():

    predictions = model(X_test)

    predictions = (
        predictions > 0.5
    ).float()


# ==========================================
# 11. ACCURACY
# ==========================================

accuracy = (
    predictions.eq(y_test)
    .sum()
    .item()
    /
    len(y_test)
)

print("\nAccuracy:", accuracy)