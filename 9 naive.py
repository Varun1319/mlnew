import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

faces = fetch_olivetti_faces()
x, y = faces["data"], faces["target"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=42, test_size=0.2
)

y_pred = GaussianNB().fit(x_train, y_train).predict(x_test)
print(f"Accuracy Score: {accuracy_score(y_test, y_pred)}")

fig, axes = plt.subplots(2, 5, figsize=(12, 10))
for i, axis in enumerate(axes.flat):
    axis.imshow(x_test[i].reshape(64, 64), cmap="grey")
    axis.set_title(f"Predicted: {y_pred[i]}, Actual: {y_test[i]}")
    axis.axis("off")

plt.tight_layout()
plt.show()
