import pandas as pd
from sklearn.datasets import load_iris

# hello = load_iris()
# df = pd.DataFrame(hello.data, columns=hello.feature_names)
# correlation_output = df.corr()
# print(correlation_output)

# data = {"Year":[2000,2001,2002,2003,2004],
#         "Price":[1000,2000,3000,1900,5000]}
# df = pd.DataFrame(data)
# print(df)
# print(df.corr())


# hello = load_iris()
# df = pd.DataFrame(hello.data, columns=["a","b","c","d"])
# correlation_output = df.corr()
# print(correlation_output)

# hello = load_iris()
#
# df = pd.DataFrame(hello.data, columns=["a","b","c","d"])
#
# correlation_output = df["a"].corr(df["c"])
# print(correlation_output)


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load dataset
hello = load_iris()

# Create DataFrame
df = pd.DataFrame(hello.data, columns=["a", "b", "c", "d"])

# Correlation matrix
correlation_output = df.corr()
print(correlation_output)

# Heatmap
sns.heatmap(correlation_output,
            annot=True,      # Show correlation values
            cmap="coolwarm", # Color theme
            fmt=".2f")       # 2 decimal places

plt.title("Correlation Heatmap")
plt.show()