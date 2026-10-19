import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

hello = load_iris()

df = pd.DataFrame(hello.data, columns=hello.feature_names)

df["target"] = hello.target

plt.scatter(df["petal length (cm)"],
            df["petal width (cm)"],
            c=df['target'],
            cmap="magma")
plt.colorbar()

plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.show()