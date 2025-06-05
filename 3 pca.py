import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import seaborn as sns

data = load_iris(as_frame=True)
df = data["data"]
target = data.target

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

pca = PCA(n_components=2)
df_scaled_pca = pca.fit_transform(df_scaled)

df_new_pca = pd.DataFrame(df_scaled_pca, columns=["PC 1", "PC 2"])
df_new_pca["Target"] = target
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df_new_pca, x="PC 1", y="PC 2", s=100, hue="Target", palette="viridis"
)
handles, _ = plt.gca().get_legend_handles_labels()
plt.legend(handles=handles, labels=data.target_names.tolist(), title="Target")
plt.show()
