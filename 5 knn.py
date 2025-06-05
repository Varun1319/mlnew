import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

np.random.seed(42)
x = np.random.rand(100).reshape(-1, 1)

x_train = np.array([1 if xi <= 0.5 else 2 for xi in x[:50]])
x_train_true = np.array([1 if xi <= 0.5 else 2 for xi in x[50:]])

k_values = [1, 2, 3, 4, 5, 20, 30]
classified_labels = {}
accuracies = {}

print("Accuracy for different k values")
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x[:50], x_train)
    predicted_labels = knn.predict(x[50:])
    classified_labels[k] = predicted_labels
    acc = accuracy_score(x_train_true, predicted_labels)
    accuracies[k] = acc
    print(f"Accuracy Score for {k} neighbors: {acc}")

    plt.figure(figsize=(10, 8))
    plt.scatter(x[:50], x_train_true, label="Actual Values", color="blue", marker="o")
    plt.scatter(
        x[50:], classified_labels[k], label="Predicted Values", color="red", marker="x"
    )
    plt.xlabel("X Values")
    plt.ylabel("Class")
    plt.legend()
    plt.grid()
    plt.show()
