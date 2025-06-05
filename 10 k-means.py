import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

data = load_breast_cancer(as_frame=True)
x, y = data.data, data.target

x_scaled = StandardScaler().fit_transform(x)
x_pca_scaled = PCA().fit_transform(x_scaled)

x_train, x_test, y_train, y_test = train_test_split(
    x_pca_scaled, y, test_size=0.2, random_state=42
)
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(x_train, y_train)
y_pred = kmeans.predict(x_test)

plt.figure(figsize=(15, 10))
plt.scatter(x_pca_scaled[:, 0], x_pca_scaled[:, 1], label="Data Points")
plt.scatter(
    kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], label="Centroids"
)
plt.title("K Means Clustering")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend()
plt.grid(True)
plt.show()
