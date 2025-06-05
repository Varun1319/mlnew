import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


data = load_breast_cancer()
x, y = data["data"], data["target"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(x_train, y_train)

print(f"Decision Tree Accuracy: {accuracy_score(y_test, model.predict(x_test)):.2f}")
plt.figure(figsize=(16, 10))
plot_tree(
    model,
    feature_names=data.feature_names,
    class_names=data.target_names,
    filled=True,
)
plt.title("Decision Tree Visualization")
plt.show()
print(
    "Decision Tree Rules:", export_text(model, feature_names=list(data.feature_names))
)
sample = np.array([1] * 30).reshape(1, -1)
print(f"New Sample Predicted Class: {data.target_names[model.predict(sample)[0]]}")
