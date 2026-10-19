# Install XGBoost if needed:
# pip install xgboost
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (accuracy_score,confusion_matrix, classification_report,
                             roc_auc_score)
from xgboost import XGBClassifier
from xgboost import plot_importance

import numpy as np
np.set_printoptions(suppress=True, precision=4)

data = load_breast_cancer()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,random_state=42)
model = XGBClassifier(n_estimators=200,learning_rate=0.1,max_depth=4,
                      subsample=0.8,colsample_bytree=0.8,random_state=42
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(X_test)
probability = model.predict_proba(X_test) #only for classification
print("\nFirst 5 Prediction Probabilities\n")
print(probability[:5])
# accuracy = accuracy_score(y_test, y_pred)
# print("\nAccuracy =", accuracy)
# cm = confusion_matrix(y_test, y_pred)
# print("\nConfusion Matrix\n")
# print(cm)
# print("\nClassification Report\n")
# print(classification_report(y_test, y_pred))
# auc = roc_auc_score(y_test,probability[:,1]) #binary classifier
# print("\nROC AUC Score =", auc)
# scores = cross_val_score(model,X,y,cv=5)
# print("\nCross Validation Scores")
# print(scores)
# print("\nAverage CV Accuracy =", scores.mean())
# print("\nDisplaying Feature Importance...")
# plot_importance(model)
# plt.show()
# model.save_model("xgboost_model.json")
# print("\nModel Saved Successfully")
# new_model = XGBClassifier()
# new_model.load_model("xgboost_model.json")
# print("Model Loaded Successfully")
#
# sample = X_test[0].reshape(1, -1)
#
# prediction = new_model.predict(sample)
#
# print("\nPrediction for First Test Sample =", prediction[0])
#
# print("\nProgram Completed Successfully")