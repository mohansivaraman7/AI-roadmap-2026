import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier

from sklearn.metrics import accuracy_score

# ----------------------------------------------------
# Load Dataset
# ----------------------------------------------------
data = load_breast_cancer()

X = data.data
y = data.target

# ----------------------------------------------------
# Train Test Split
# ----------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ----------------------------------------------------
# Base Models
# ----------------------------------------------------
base_models = [

    ("dt", DecisionTreeClassifier()),

    ("rf", RandomForestClassifier()),

    ("svm", SVC(probability=True))

]

# ----------------------------------------------------
# Meta Model
# ----------------------------------------------------
meta_model = LogisticRegression()

# ----------------------------------------------------
# Stacking Classifier
# ----------------------------------------------------
stack_model = StackingClassifier(

    estimators=base_models,

    final_estimator=meta_model

)

# ----------------------------------------------------
# Train
# ----------------------------------------------------
stack_model.fit(X_train, y_train)

# ----------------------------------------------------
# Predict
# ----------------------------------------------------
y_pred = stack_model.predict(X_test)

# ----------------------------------------------------
# Accuracy
# ----------------------------------------------------
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy :", accuracy)