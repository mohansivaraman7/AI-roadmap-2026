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
            cmap="rainbow", # Color theme
            fmt=".2f")       # 2 decimal places
plt.title("Correlation Heatmap")
plt.show()