from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

load_iris_input = load_iris()
print(load_iris_input)

x = load_iris_input.data
y = load_iris_input.target

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, random_state=42)

model = DecisionTreeClassifier()
model.fit(x_train, y_train)

y_prediciton = model.predict(x_test)
print(y_prediciton)

metric_evaluation = confusion_matrix(y_test, y_prediciton)
print(metric_evaluation)