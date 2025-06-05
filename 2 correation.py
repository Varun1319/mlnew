import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

data = fetch_california_housing(as_frame=True)
df = data["data"]
df["MedHouseVal"] = data["target"]

correlation_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, fmt=".2f", cbar=True, cmap="coolwarm", annot=True)
plt.title("Heatmap of the correlation matrix")

sns.pairplot(df, diag_kind="kde", corner=True)
plt.show()
